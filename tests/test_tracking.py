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
    classify_distant_feet,
    classify_missed_people,
    drop_outside_ground_area,
    ground_error_components,
    in_ground_area,
    match_frame,
    match_frame_ground,
    match_tracks_to_gt,
    nearest_gt_distance,
    recall_by_camera_distance,
    tracking_summary_ground,
    relabel_tracks,
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
    assert int(mapping.loc[mapping["track_id"] == 5, "person_id"].iloc[0]) == -5


def test_ground_area_filter_drops_a_point_at_x_20() -> None:
    tracks = pd.DataFrame({"X_m": [1.0, 20.0, 9.0], "Y_m": [0.0, 0.0, 27.0]})
    kept, dropped = drop_outside_ground_area(tracks)
    assert dropped == 1
    assert kept["X_m"].tolist() == [1.0, 9.0]


def test_boundary_margin_ignores_unmatched_edge_feet() -> None:
    """A false foot on the X = -3 m edge is ignored once the 1 m band is applied."""
    gt = pd.DataFrame(
        {
            "frame_idx": [0],
            "person_id": [1],
            "X_m": [5.0],
            "Y_m": [10.0],
        }
    )
    tracks = pd.DataFrame(
        {
            "frame_idx": [0, 0],
            "track_id": [1, 2],
            "X_m": [5.1, -2.9],
            "Y_m": [10.0, 10.0],
            "score": [0.9, 0.9],
        }
    )
    full = tracking_summary_ground(tracks, gt, max_dist_m=0.5, boundary_margin_m=0.0)
    assert full["det_precision"] == pytest.approx(0.5)
    assert full["det_recall"] == pytest.approx(1.0)
    assert full["ignored_pred"] == 0

    inset = tracking_summary_ground(tracks, gt, max_dist_m=0.5, boundary_margin_m=1.0)
    assert inset["det_precision"] == pytest.approx(1.0)
    assert inset["det_recall"] == pytest.approx(1.0)
    assert inset["ignored_pred"] == 1
    assert inset["mota"] == pytest.approx(1.0)
    assert in_ground_area([-2.9], [10.0], margin_m=0.0)[0]
    assert not in_ground_area([-2.9], [10.0], margin_m=1.0)[0]


def test_match_frame_ground_keeps_0_3_m_and_rejects_0_7_m() -> None:
    pred = np.array([[0.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.3, 0.0], [2.7, 0.0]])
    assert match_frame_ground(pred, gt, 0.5) == [(0, 0)]


def test_nearest_gt_distance_stays_inside_its_frame() -> None:
    tracks = pd.DataFrame(
        {
            "frame_idx": [0, 0, 1],
            "X_m": [0.0, 3.0, 0.0],
            "Y_m": [0.0, 0.0, 0.0],
        }
    )
    gt = pd.DataFrame(
        {
            "frame_idx": [0, 0, 1],
            "X_m": [0.2, 10.0, 5.0],
            "Y_m": [0.0, 0.0, 0.0],
        }
    )
    assert np.allclose(nearest_gt_distance(tracks, gt), [0.2, 2.8, 5.0])


def test_along_view_is_positive_when_the_foot_is_closer_to_the_camera() -> None:
    """Camera at the origin, truth straight ahead. A foot in between is positive."""
    camera = np.array([0.0, 0.0])
    truth = np.array([[0.0, 10.0], [0.0, 10.0]])
    pred = np.array([[0.0, 8.0], [1.0, 10.0]])
    along, sideways = ground_error_components(pred, truth, camera)
    assert along[0] == pytest.approx(2.0)
    assert sideways[0] == pytest.approx(0.0)
    assert along[1] == pytest.approx(0.0)
    assert sideways[1] == pytest.approx(-1.0)


def test_recall_by_camera_distance_uses_three_bins() -> None:
    gt = pd.DataFrame(
        {
            "frame_idx": [0, 0, 0],
            "person_id": [1, 2, 3],
            "X_m": [0.0, 0.0, 0.0],
            "Y_m": [5.0, 15.0, 25.0],
        }
    )
    matches = pd.DataFrame({"frame_idx": [0], "person_id": [1], "track_id": [7]})
    rows = recall_by_camera_distance(gt, matches, np.array([0.0, 0.0]))
    assert [row["bin"] for row in rows] == ["0-10", "10-20", ">20"]
    assert [row["recall"] for row in rows] == [1.0, 0.0, 0.0]
    assert [row["gt"] for row in rows] == [1, 1, 1]


def _box_row(frame: int, xmin: float, ymin: float, xmax: float, ymax: float, **extra: float) -> dict[str, float]:
    row = {"frame_idx": frame, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
    row.update(extra)
    return row


def test_distant_foot_categories_follow_the_priority_order() -> None:
    """cut_off wins over a good overlap, and a wide box is partial rather than foot_wrong."""
    tracks = pd.DataFrame(
        [
            _box_row(0, 100, 100, 160, 1076),  # overlaps the person and touches the bottom
            _box_row(0, 80, 140, 200, 220),  # wide overlap: partial wins over foot_wrong
            _box_row(0, 100, 100, 150, 250),  # tall overlap, foot is still far
            _box_row(0, 800, 100, 850, 260),  # no overlap
            _box_row(0, 140, 100, 200, 260),  # IoU 0.2 lands in other
            _box_row(0, 100, 100, 150, 250),  # close foot, not classified
        ]
    )
    gt = pd.DataFrame([_box_row(0, 100, 100, 160, 260)])
    nearest = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 0.2])
    labels = classify_distant_feet(tracks, gt, nearest, far_m=1.0)
    assert labels.tolist() == ["cut_off", "partial", "foot_wrong", "no_person", "other", ""]


def test_left_edge_box_is_cut_off() -> None:
    tracks = pd.DataFrame([_box_row(0, 0, 100, 40, 220)])
    gt = pd.DataFrame([_box_row(0, 500, 100, 540, 240)])
    assert classify_distant_feet(tracks, gt, np.array([3.0])).tolist() == ["cut_off"]


def test_missed_people_split_by_box_overlap() -> None:
    pred = pd.DataFrame([_box_row(0, 100, 100, 150, 250), _box_row(0, 400, 100, 440, 240)])
    gt = pd.DataFrame(
        [
            _box_row(0, 100, 100, 150, 250, person_id=1, X_m=1.0, Y_m=1.0),
            _box_row(0, 700, 100, 750, 260, person_id=2, X_m=8.0, Y_m=8.0),
            _box_row(0, 400, 100, 440, 240, person_id=3, X_m=4.0, Y_m=4.0),
        ]
    )
    matches = pd.DataFrame({"frame_idx": [0], "person_id": [3], "track_id": [1]})
    missed = classify_missed_people(pred, gt, matches)
    reasons = dict(zip(missed["person_id"], missed["reason"]))
    assert reasons == {1: "detected_foot_wrong", 2: "not_detected"}
    assert 3 not in reasons


def test_unmatched_tracks_get_different_negative_ids() -> None:
    tracks = _tracks([4, 4], [6, 6])
    matches = pd.DataFrame(columns=["frame_idx", "track_id", "person_id"])
    mapping = track_to_person_map(matches, tracks)
    ids = dict(zip(mapping["track_id"], mapping["person_id"]))
    assert ids == {4: -4, 6: -6}


def test_relabel_tracks_keeps_one_row_per_person_per_frame() -> None:
    tracks = _tracks([1, 1], [2, 2])
    tracks.loc[tracks["track_id"] == 1, "score"] = 0.95
    tracks.loc[tracks["track_id"] == 2, "score"] = 0.40
    idmap = pd.DataFrame({"track_id": [1, 2], "person_id": [7, 7]})
    relabeled, dropped = relabel_tracks(tracks, idmap)
    assert dropped == 2
    assert relabeled.groupby(["frame_idx", "person_id"]).size().max() == 1
    assert relabeled["track_id"].tolist() == [1, 1]
    assert relabeled["person_id"].tolist() == [7, 7]


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
