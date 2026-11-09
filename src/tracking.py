"""Person detection and tracking on the annotated WILDTRACK frames.

The detectors and trackers are off-the-shelf. This module only keeps their
inputs and outputs in one shared format so the event code can read them:

    detections CSV: frame_idx, frame_number, time_sec, xmin, ymin, xmax, ymax, score
    tracks CSV:     frame_idx, frame_number, time_sec, track_id, xmin, ymin, xmax, ymax, score

Boxes are in pixels of the original 1920 x 1080 image. frame_idx, frame_number
and time_sec follow src.data.load_annotations, so a track row and a ground-truth
row with the same frame_idx describe the same image.

Heavy libraries (ultralytics, supervision, deep_sort_realtime) are imported
inside the functions that use them, so the rest of the repository and the
tests run without them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from src.data import ANNOTATED_FPS, annotation_dir, normalize_cam_id, resolve_dataset_root

DETECTION_COLUMNS = [
    "frame_idx",
    "frame_number",
    "time_sec",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
]
TRACK_COLUMNS = [
    "frame_idx",
    "frame_number",
    "time_sec",
    "track_id",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "score",
]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# COCO class 0 is "person".
PERSON_CLASS_ID = 0


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def annotated_frames(root: str | Path, cam_id: str | int) -> pd.DataFrame:
    """List the annotated frames of one camera, in time order.

    Frames come from the annotation JSON names, exactly as in
    load_annotations, so frame_idx 0..399 lines up with the ground truth.
    Images without an annotation (for example 00002000.png) are skipped.
    Returns frame_idx, frame_number, time_sec, and image_path.
    """
    name = normalize_cam_id(cam_id)
    folder = resolve_dataset_root(root) / "Image_subsets" / name
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    stems = sorted(path.stem for path in annotation_dir(root).glob("*.json"))
    if not stems:
        raise FileNotFoundError(f"No annotation JSON files in {annotation_dir(root)}")
    rows = []
    for frame_idx, stem in enumerate(stems):
        image = next(
            (folder / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES if (folder / f"{stem}{suffix}").is_file()),
            None,
        )
        if image is None:
            raise FileNotFoundError(f"No image for annotated frame {stem} in {folder}")
        rows.append(
            {
                "frame_idx": frame_idx,
                "frame_number": int(stem),
                "time_sec": frame_idx / ANNOTATED_FPS,
                "image_path": str(image),
            }
        )
    return pd.DataFrame(rows, columns=["frame_idx", "frame_number", "time_sec", "image_path"])


# ---------------------------------------------------------------------------
# CSV schema
# ---------------------------------------------------------------------------


def detection_rows(frame: dict[str, Any] | pd.Series, xyxy: np.ndarray, scores: np.ndarray) -> list[dict[str, Any]]:
    """Turn one frame's boxes (N, 4) and scores (N,) into detection CSV rows."""
    boxes = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    confidences = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(confidences):
        raise ValueError(f"{len(boxes)} boxes but {len(confidences)} scores")
    return [
        {
            "frame_idx": int(frame["frame_idx"]),
            "frame_number": int(frame["frame_number"]),
            "time_sec": float(frame["time_sec"]),
            "xmin": float(box[0]),
            "ymin": float(box[1]),
            "xmax": float(box[2]),
            "ymax": float(box[3]),
            "score": float(score),
        }
        for box, score in zip(boxes, confidences)
    ]


def _check_columns(table: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{what} is missing columns: {missing}")


def validate_boxes(table: pd.DataFrame, width: int = 1920, height: int = 1080, tol: float = 2.0) -> None:
    """Raise if any box is empty, flipped, or far outside the image."""
    if table.empty:
        return
    bad = table.loc[(table["xmax"] <= table["xmin"]) | (table["ymax"] <= table["ymin"])]
    if not bad.empty:
        raise ValueError(f"{len(bad)} boxes have xmax <= xmin or ymax <= ymin")
    outside = table.loc[
        (table["xmin"] < -tol)
        | (table["ymin"] < -tol)
        | (table["xmax"] > width + tol)
        | (table["ymax"] > height + tol)
    ]
    if not outside.empty:
        raise ValueError(
            f"{len(outside)} boxes lie outside the {width} x {height} image. "
            "Boxes must be in original image pixels, not resized ones."
        )


def validate_tracks(tracks: pd.DataFrame, width: int = 1920, height: int = 1080) -> None:
    """Check a tracks table before it is saved or handed to the event code."""
    _check_columns(tracks, TRACK_COLUMNS, "tracks")
    validate_boxes(tracks, width, height)
    duplicated = tracks.duplicated(["frame_idx", "track_id"])
    if duplicated.any():
        raise ValueError(f"{int(duplicated.sum())} rows repeat a track_id within one frame")
    if (tracks["score"] < 0).any() or (tracks["score"] > 1).any():
        raise ValueError("score must lie in [0, 1]")


def save_csv(table: pd.DataFrame, columns: list[str], path: str | Path) -> Path:
    """Sort by frame, then track, and write with a fixed column order."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["frame_idx", "track_id"] if "track_id" in columns else ["frame_idx"]
    ordered = table.loc[:, columns].sort_values(keys, kind="mergesort").reset_index(drop=True)
    ordered.to_csv(output, index=False, float_format="%.3f")
    return output


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------


class Tracker(Protocol):
    """One frame in, the confirmed tracks seen in that frame out."""

    def update(self, boxes: np.ndarray, scores: np.ndarray, image_path: str) -> list[tuple[int, np.ndarray, float]]:
        """Return (track_id, xyxy box, score) for tracks matched in this frame."""
        ...


class ByteTrackTracker:
    """ByteTrack from the supervision package. Uses boxes and scores only.

    ByteTrack keeps low-score detections for a second matching round, so it
    should receive every detection down to the detector's low threshold.
    """

    def __init__(self, fps: float = ANNOTATED_FPS, max_lost_seconds: float = 2.0) -> None:
        import supervision as sv

        self._sv = sv
        # supervision keeps a lost track for int(frame_rate / 30 * lost_track_buffer)
        # frames, so this buffer keeps it for max_lost_seconds at any frame rate.
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=int(round(30 * max_lost_seconds)),
            minimum_matching_threshold=0.8,
            frame_rate=int(round(fps)),
        )

    def update(self, boxes: np.ndarray, scores: np.ndarray, image_path: str) -> list[tuple[int, np.ndarray, float]]:
        detections = self._sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(scores, dtype=np.float32).reshape(-1),
            class_id=np.full(len(scores), PERSON_CLASS_ID, dtype=int),
        )
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None:
            return []
        return [
            (int(track_id), np.asarray(box, dtype=np.float64), float(score))
            for box, score, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id)
        ]


class DeepSortTracker:
    """DeepSORT from deep_sort_realtime. Adds an appearance embedding per box.

    DeepSORT has no low-score matching round, so weak detections are dropped
    before they reach it (min_score). Only confirmed tracks that were matched
    to a detection in this frame are returned, with that detection's box.
    """

    def __init__(
        self,
        fps: float = ANNOTATED_FPS,
        max_lost_seconds: float = 2.0,
        n_init: int = 2,
        min_score: float = 0.3,
        use_gpu: bool | None = None,
    ) -> None:
        import cv2
        from deep_sort_realtime.deepsort_tracker import DeepSort

        if use_gpu is None:
            try:
                import torch

                use_gpu = bool(torch.cuda.is_available())
            except ImportError:
                use_gpu = False
        self._cv2 = cv2
        self._min_score = float(min_score)
        self._tracker = DeepSort(
            max_age=max(1, int(round(max_lost_seconds * fps))),
            n_init=int(n_init),
            embedder="mobilenet",
            half=use_gpu,
            bgr=True,
            embedder_gpu=use_gpu,
        )

    def update(self, boxes: np.ndarray, scores: np.ndarray, image_path: str) -> list[tuple[int, np.ndarray, float]]:
        image = self._cv2.imread(image_path, self._cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        raw = [
            ([float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])], float(score), PERSON_CLASS_ID)
            for box, score in zip(np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1))
            if score >= self._min_score
        ]
        out = []
        for track in self._tracker.update_tracks(raw, frame=image):
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            box = track.to_ltrb(orig=True, orig_strict=True)
            if box is None:
                continue
            score = track.get_det_conf()
            out.append((int(track.track_id), np.asarray(box, dtype=np.float64), float(score if score is not None else 0.0)))
        return out


TRACKERS = {
    "bytetrack": ByteTrackTracker,
    "deepsort": DeepSortTracker,
}


def make_tracker(name: str, **kwargs: Any) -> Tracker:
    """Build a tracker by name: 'bytetrack' or 'deepsort'."""
    key = name.lower()
    if key not in TRACKERS:
        raise ValueError(f"Unknown tracker {name!r}. Expected one of: {', '.join(TRACKERS)}")
    return TRACKERS[key](**kwargs)


def run_tracker(detections: pd.DataFrame, frames: pd.DataFrame, tracker: Tracker) -> pd.DataFrame:
    """Feed every annotated frame, in order, to one tracker.

    Frames with no detections are still passed in, so lost tracks age
    correctly. Returns a table with TRACK_COLUMNS.
    """
    _check_columns(detections, DETECTION_COLUMNS, "detections")
    by_frame = {int(key): group for key, group in detections.groupby("frame_idx", sort=False)}
    rows = []
    for frame in frames.sort_values("frame_idx").itertuples(index=False):
        group = by_frame.get(int(frame.frame_idx))
        if group is None:
            boxes = np.empty((0, 4), dtype=np.float64)
            scores = np.empty(0, dtype=np.float64)
        else:
            boxes = group[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
            scores = group["score"].to_numpy(dtype=np.float64)
        for track_id, box, score in tracker.update(boxes, scores, str(frame.image_path)):
            rows.append(
                {
                    "frame_idx": int(frame.frame_idx),
                    "frame_number": int(frame.frame_number),
                    "time_sec": float(frame.time_sec),
                    "track_id": int(track_id),
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3]),
                    "score": float(np.clip(score, 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


# ---------------------------------------------------------------------------
# Evaluation against ground-truth boxes
# ---------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N, 4) and b (M, 4), both xyxy."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_frame(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one matching of predicted to ground-truth boxes by IoU.

    Uses the Hungarian algorithm on 1 - IoU and keeps pairs with
    IoU >= iou_threshold. Returns (pred_index, gt_index) pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    iou = box_iou(pred_boxes, gt_boxes)
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]


def gt_boxes(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible ground-truth boxes from load_annotations, one row per person and frame."""
    _check_columns(annotations, ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax", "visible"], "annotations")
    visible = annotations.loc[annotations["visible"]]
    return visible.loc[:, ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax"]].reset_index(drop=True)


def match_tracks_to_gt(tracks: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    """Match every frame and return one row per matched (track, person) pair.

    Columns: frame_idx, track_id, person_id, iou.
    """
    box_columns = ["xmin", "ymin", "xmax", "ymax"]
    pred_by_frame = {int(key): group for key, group in tracks.groupby("frame_idx", sort=False)}
    rows = []
    for frame_idx, gt_frame in gt.groupby("frame_idx", sort=True):
        pred_frame = pred_by_frame.get(int(frame_idx))
        if pred_frame is None:
            continue
        pred_xyxy = pred_frame[box_columns].to_numpy(dtype=np.float64)
        gt_xyxy = gt_frame[box_columns].to_numpy(dtype=np.float64)
        iou = box_iou(pred_xyxy, gt_xyxy)
        for p, g in match_frame(pred_xyxy, gt_xyxy, iou_threshold):
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "track_id": int(pred_frame["track_id"].iloc[p]),
                    "person_id": int(gt_frame["person_id"].iloc[g]),
                    "iou": float(iou[p, g]),
                }
            )
    return pd.DataFrame(rows, columns=["frame_idx", "track_id", "person_id", "iou"])


def count_id_switches(matches: pd.DataFrame) -> int:
    """CLEAR-MOT ID switches: a person's matched track_id differs from the last one.

    Frames where the person is unmatched are skipped, so losing someone and
    picking them up again under the same track_id is not a switch.
    """
    switches = 0
    for _person, group in matches.sort_values("frame_idx").groupby("person_id", sort=False):
        ids = group["track_id"].to_numpy()
        switches += int(np.sum(ids[1:] != ids[:-1]))
    return switches


def track_to_person_map(matches: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """For each track, the ground-truth person it matched most often.

    Columns: track_id, person_id (-1 if never matched), matched_frames,
    track_frames, purity = matched_frames / track_frames. Event matching in
    src.evaluation compares identities, so predicted track IDs need a link to
    ground-truth person IDs; this table is that link.
    """
    lengths = tracks.groupby("track_id").size().rename("track_frames")
    if matches.empty:
        best = pd.DataFrame(columns=["track_id", "person_id", "matched_frames"])
    else:
        counts = matches.groupby(["track_id", "person_id"]).size().rename("matched_frames").reset_index()
        counts = counts.sort_values(["track_id", "matched_frames", "person_id"], ascending=[True, False, True])
        best = counts.drop_duplicates("track_id", keep="first")
    table = lengths.reset_index().merge(best, on="track_id", how="left")
    table["person_id"] = table["person_id"].fillna(-1).astype(int)
    table["matched_frames"] = table["matched_frames"].fillna(0).astype(int)
    table["purity"] = table["matched_frames"] / table["track_frames"]
    return table[["track_id", "person_id", "matched_frames", "track_frames", "purity"]]


def tracking_summary(tracks: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5) -> dict[str, Any]:
    """Box-level detection and tracking scores for one tracks table.

    det_precision / det_recall: matched boxes over predicted / ground-truth boxes.
    id_switches: CLEAR-MOT ID switches.
    mota: 1 - (misses + false positives + ID switches) / ground-truth boxes.
    """
    matches = match_tracks_to_gt(tracks, gt, iou_threshold)
    tp = len(matches)
    fp = len(tracks) - tp
    fn = len(gt) - tp
    id_switches = count_id_switches(matches)
    mapping = track_to_person_map(matches, tracks) if len(tracks) else pd.DataFrame(columns=["purity"])
    return {
        "pred_boxes": int(len(tracks)),
        "gt_boxes": int(len(gt)),
        "matched": int(tp),
        "det_precision": tp / len(tracks) if len(tracks) else float("nan"),
        "det_recall": tp / len(gt) if len(gt) else float("nan"),
        "mean_iou": float(matches["iou"].mean()) if tp else float("nan"),
        "id_switches": int(id_switches),
        "mota": 1.0 - (fn + fp + id_switches) / len(gt) if len(gt) else float("nan"),
        "pred_tracks": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        "gt_people": int(gt["person_id"].nunique()),
        "mean_track_purity": float(mapping["purity"].mean()) if len(mapping) else float("nan"),
    }


def iter_track_files(folder: str | Path, cam_id: str) -> Iterable[Path]:
    """Track CSVs for one camera, named tracks_<cam>_<detector>_<tracker>.csv."""
    return sorted(Path(folder).glob(f"tracks_{cam_id}_*.csv"))
