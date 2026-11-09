"""Tests for the detection/tracking format and the box-level tracking scores.

These use small synthetic tables, so they need neither the dataset nor the
detector and tracker libraries.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.tracking import (
    DETECTION_COLUMNS,
    TRACK_COLUMNS,
    annotated_frames,
    box_iou,
    count_id_switches,
    detection_rows,
    match_frame,
    match_tracks_to_gt,
    run_tracker,
    save_csv,
    track_to_person_map,
    tracking_summary,
    validate_tracks,
)

BOX_A = [100.0, 100.0, 150.0, 250.0]
BOX_B = [400.0, 120.0, 460.0, 280.0]


def _gt(frames: int) -> pd.DataFrame:
    """Two people standing still for some frames."""
    rows = []
    for frame_idx in range(frames):
        rows.append({"frame_idx": frame_idx, "person_id": 7, **dict(zip(["xmin", "ymin", "xmax", "ymax"], BOX_A))})
        rows.append({"frame_idx": frame_idx, "person_id": 9, **dict(zip(["xmin", "ymin", "xmax", "ymax"], BOX_B))})
    return pd.DataFrame(rows)


def _tracks(ids_a: list[int], ids_b: list[int]) -> pd.DataFrame:
    """Predicted boxes on top of the ground truth, with chosen track IDs."""
    rows = []
    for frame_idx, (id_a, id_b) in enumerate(zip(ids_a, ids_b)):
        for track_id, box in ((id_a, BOX_A), (id_b, BOX_B)):
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "frame_number": frame_idx * 5,
                    "time_sec": frame_idx / 2.0,
                    "track_id": track_id,
                    "xmin": box[0],
                    "ymin": box[1],
                    "xmax": box[2],
                    "ymax": box[3],
                    "score": 0.9,
                }
            )
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def test_iou_of_identical_and_disjoint_boxes() -> None:
    iou = box_iou(np.array([BOX_A]), np.array([BOX_A, BOX_B]))
    assert iou.shape == (1, 2)
    assert np.isclose(iou[0, 0], 1.0)
    assert iou[0, 1] == 0.0


def test_iou_of_half_overlap() -> None:
    """Two 10 x 10 boxes shifted by 5 px share 50 of 150 square pixels."""
    iou = box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]]))
    assert np.isclose(iou[0, 0], 50 / 150)


def test_match_frame_is_one_to_one_and_respects_threshold() -> None:
    pred = np.array([BOX_B, BOX_A, [900, 900, 950, 1000]])
    gt = np.array([BOX_A, BOX_B])
    assert sorted(match_frame(pred, gt, 0.5)) == [(0, 1), (1, 0)]
    shifted = np.array([[130.0, 100.0, 180.0, 250.0]])  # IoU 0.25 with BOX_A
    assert match_frame(shifted, np.array([BOX_A]), 0.5) == []


def test_perfect_tracks_score_perfectly() -> None:
    gt = _gt(6)
    tracks = _tracks([1] * 6, [2] * 6)
    summary = tracking_summary(tracks, gt)
    assert summary["det_precision"] == 1.0
    assert summary["det_recall"] == 1.0
    assert summary["id_switches"] == 0
    assert summary["mota"] == 1.0
    mapping = track_to_person_map(match_tracks_to_gt(tracks, gt), tracks)
    assert dict(zip(mapping["track_id"], mapping["person_id"])) == {1: 7, 2: 9}
    assert np.allclose(mapping["purity"], 1.0)


def test_swapping_ids_counts_two_switches() -> None:
    """IDs swap at frame 3: each person changes track once."""
    gt = _gt(6)
    tracks = _tracks([1, 1, 1, 2, 2, 2], [2, 2, 2, 1, 1, 1])
    matches = match_tracks_to_gt(tracks, gt)
    assert count_id_switches(matches) == 2
    mapping = track_to_person_map(matches, tracks)
    assert np.allclose(mapping["purity"], 0.5)


def test_lost_and_found_with_same_id_is_not_a_switch() -> None:
    gt = _gt(5)
    tracks = _tracks([1] * 5, [2] * 5)
    tracks = tracks.loc[~((tracks["frame_idx"] == 2) & (tracks["track_id"] == 1))]
    summary = tracking_summary(tracks, gt)
    assert summary["id_switches"] == 0
    assert summary["det_recall"] == pytest.approx(9 / 10)


def test_false_positive_lowers_precision() -> None:
    gt = _gt(2)
    tracks = _tracks([1, 1], [2, 2])
    extra = tracks.iloc[[0]].assign(track_id=5, xmin=900.0, xmax=950.0)
    summary = tracking_summary(pd.concat([tracks, extra], ignore_index=True), gt)
    assert summary["det_precision"] == pytest.approx(4 / 5)
    mapping = track_to_person_map(match_tracks_to_gt(pd.concat([tracks, extra]), gt), pd.concat([tracks, extra]))
    assert int(mapping.loc[mapping["track_id"] == 5, "person_id"].iloc[0]) == -1


def test_validate_tracks_rejects_bad_tables() -> None:
    good = _tracks([1, 1], [2, 2])
    validate_tracks(good)
    with pytest.raises(ValueError, match="repeat a track_id"):
        validate_tracks(pd.concat([good, good.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="outside"):
        validate_tracks(good.assign(xmax=3000.0))
    with pytest.raises(ValueError, match="missing columns"):
        validate_tracks(good.drop(columns=["score"]))


class _EchoTracker:
    """Gives detection i in each frame track ID i + 1. Records what it was fed."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, str]] = []

    def update(self, boxes: np.ndarray, scores: np.ndarray, image_path: str):
        self.seen.append((len(boxes), image_path))
        return [(index + 1, box, float(score)) for index, (box, score) in enumerate(zip(boxes, scores))]


def test_run_tracker_feeds_every_frame_including_empty_ones() -> None:
    frames = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "frame_number": [0, 5, 10],
            "time_sec": [0.0, 0.5, 1.0],
            "image_path": ["a.png", "b.png", "c.png"],
        }
    )
    frame0 = frames.iloc[0].to_dict()
    frame2 = frames.iloc[2].to_dict()
    detections = pd.DataFrame(
        detection_rows(frame0, np.array([BOX_A, BOX_B]), np.array([0.9, 0.8]))
        + detection_rows(frame2, np.array([BOX_A]), np.array([0.7])),
        columns=DETECTION_COLUMNS,
    )
    tracker = _EchoTracker()
    tracks = run_tracker(detections, frames, tracker)
    assert tracker.seen == [(2, "a.png"), (0, "b.png"), (1, "c.png")]
    assert list(tracks.columns) == TRACK_COLUMNS
    assert tracks["frame_idx"].tolist() == [0, 0, 2]
    assert tracks["time_sec"].tolist() == [0.0, 0.0, 1.0]
    validate_tracks(tracks)


def test_save_csv_round_trip(tmp_path) -> None:
    tracks = _tracks([3, 3], [1, 1]).sample(frac=1.0, random_state=0)
    path = save_csv(tracks, TRACK_COLUMNS, tmp_path / "tracks.csv")
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == TRACK_COLUMNS
    assert loaded[["frame_idx", "track_id"]].values.tolist() == [[0, 1], [0, 3], [1, 1], [1, 3]]


def test_annotated_frames_follow_annotation_files(tmp_path) -> None:
    """frame_idx counts annotation files; images without one are skipped."""
    root = tmp_path / "Wildtrack_dataset"
    (root / "annotations_positions").mkdir(parents=True)
    (root / "Image_subsets" / "C1").mkdir(parents=True)
    for stem in ("00000000", "00000005", "00000010"):
        (root / "annotations_positions" / f"{stem}.json").write_text(json.dumps([]))
        (root / "Image_subsets" / "C1" / f"{stem}.png").write_bytes(b"")
    (root / "Image_subsets" / "C1" / "00002000.png").write_bytes(b"")
    frames = annotated_frames(root, "C1")
    assert frames["frame_idx"].tolist() == [0, 1, 2]
    assert frames["frame_number"].tolist() == [0, 5, 10]
    assert frames["time_sec"].tolist() == [0.0, 0.5, 1.0]
    assert frames["image_path"].str.endswith(".png").all()
