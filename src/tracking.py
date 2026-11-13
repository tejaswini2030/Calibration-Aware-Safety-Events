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

from src.data import (
    ANNOTATED_FPS,
    CELL_SIZE_M,
    DATA_ROOT,
    GRID_HEIGHT,
    GRID_WIDTH,
    ORIGIN_X_M,
    ORIGIN_Y_M,
    annotation_dir,
    normalize_cam_id,
    resolve_dataset_root,
)

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
# Annotated ground rectangle: origin (-3, -9) m and a 12 m x 36 m plane.
# A foot on the boundary still counts as inside.
GROUND_X_MIN_M = ORIGIN_X_M
GROUND_X_MAX_M = ORIGIN_X_M + GRID_WIDTH * CELL_SIZE_M
GROUND_Y_MIN_M = ORIGIN_Y_M
GROUND_Y_MAX_M = ORIGIN_Y_M + GRID_HEIGHT * CELL_SIZE_M
# A predicted foot farther than this from every ground-truth foot is unmatched.
GROUND_MATCH_DISTANCE_M = 0.5
# Shrink the ground rectangle by this many meters on every side before counting
# TP / FP / FN. Matching still uses the full rectangle; unmatched feet in the
# band between the full and inner rectangles are ignored.
DEFAULT_BOUNDARY_MARGIN_M = 1.0
# Original WILDTRACK frame. A box bottom within 5 px of 1080 is treated as cut off.
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CUT_OFF_BOTTOM_Y = IMAGE_HEIGHT - 5
# Distant feet (more than 1 m from every person) are labeled in this order.
DISTANT_FOOT_M = 1.0
PARTIAL_ASPECT_MAX = 1.5
FOOT_WRONG_IOU = 0.3
NO_PERSON_IOU = 0.1
DISTANT_FOOT_CATEGORIES = ("cut_off", "partial", "foot_wrong", "no_person", "other")

_H_IMG_TO_WORLD: dict[str, np.ndarray] = {}


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

    def __init__(
        self,
        fps: float = ANNOTATED_FPS,
        max_lost_seconds: float = 2.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        import supervision as sv

        self._sv = sv
        # supervision keeps a lost track for int(frame_rate / 30 * lost_track_buffer)
        # frames, so this buffer keeps it for max_lost_seconds at any frame rate.
        import inspect

        kwargs: dict[str, Any] = {
            "track_activation_threshold": float(track_activation_threshold),
            "lost_track_buffer": int(round(30 * max_lost_seconds)),
            "minimum_matching_threshold": 0.8,
            "frame_rate": int(round(fps)),
        }
        # Older supervision builds omit minimum_consecutive_frames.
        if "minimum_consecutive_frames" in inspect.signature(sv.ByteTrack).parameters:
            kwargs["minimum_consecutive_frames"] = int(minimum_consecutive_frames)
        self._tracker = sv.ByteTrack(**kwargs)

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
# Evaluation against ground-truth boxes and ground-plane feet
# ---------------------------------------------------------------------------


def _h_img_to_world(cam_id: str | int) -> np.ndarray:
    """Ground-truth image-to-world homography, the inverse of H*."""
    name = normalize_cam_id(cam_id)
    cached = _H_IMG_TO_WORLD.get(name)
    if cached is not None:
        return cached
    from src.geometry import ground_homography, invert_homography, load_camera_calibration

    K, R, t = load_camera_calibration(DATA_ROOT, name)
    homography = invert_homography(ground_homography(K, R, t))
    _H_IMG_TO_WORLD[name] = homography
    return homography


def foot_points_world(tracks: pd.DataFrame, cam_id: str | int) -> pd.DataFrame:
    """Project each box's bottom-center onto the ground plane.

    The foot point is ((xmin + xmax) / 2, ymax) in pixels. It is mapped with
    image_to_world and H*, the ground-truth homography. Adds X_m and Y_m in
    meters and does not modify the caller's table.
    """
    _check_columns(tracks, ["xmin", "xmax", "ymax"], "tracks")
    out = tracks.copy()
    if out.empty:
        out["X_m"] = np.array([], dtype=np.float64)
        out["Y_m"] = np.array([], dtype=np.float64)
        return out
    from src.geometry import image_to_world

    feet = np.column_stack(
        [
            (out["xmin"].to_numpy(dtype=np.float64) + out["xmax"].to_numpy(dtype=np.float64)) / 2.0,
            out["ymax"].to_numpy(dtype=np.float64),
        ]
    )
    world = image_to_world(_h_img_to_world(cam_id), feet)
    out["X_m"] = world[:, 0]
    out["Y_m"] = world[:, 1]
    return out


def ground_area_bounds(margin_m: float = 0.0) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) of the ground rectangle after an inset.

    margin_m = 0 is the full annotated area. A positive margin shrinks every
    side. If the margin would empty the rectangle, the bounds collapse to an
    empty interval so every point is outside.
    """
    margin = float(margin_m)
    if margin < 0:
        raise ValueError(f"margin_m must be >= 0, got {margin_m}")
    xmin = GROUND_X_MIN_M + margin
    xmax = GROUND_X_MAX_M - margin
    ymin = GROUND_Y_MIN_M + margin
    ymax = GROUND_Y_MAX_M - margin
    if xmin > xmax or ymin > ymax:
        return xmin, xmin, ymin, ymin
    return xmin, xmax, ymin, ymax


def in_ground_area(x_m: np.ndarray, y_m: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
    """True where (X, Y) lies inside the (possibly inset) ground rectangle.

    margin_m = 0 is the full annotated area. A positive margin is the inner
    "care" region used after matching.
    """
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    xmin, xmax, ymin, ymax = ground_area_bounds(margin_m)
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def drop_outside_ground_area(tracks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop predicted rows whose foot point is outside the annotated ground area.

    The area is X from -3 m to 9 m and Y from -9 m to 27 m. Ground truth
    already lies inside it, so only predictions are filtered. Returns the
    kept rows and how many were dropped.
    """
    _check_columns(tracks, ["X_m", "Y_m"], "tracks")
    if tracks.empty:
        return tracks.copy(), 0
    keep = in_ground_area(tracks["X_m"].to_numpy(), tracks["Y_m"].to_numpy())
    return tracks.loc[keep].reset_index(drop=True), int((~keep).sum())


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


def match_frame_ground(
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    max_dist_m: float = GROUND_MATCH_DISTANCE_M,
) -> list[tuple[int, int]]:
    """One-to-one matching of predicted feet to ground-truth feet.

    Uses the Hungarian algorithm on Euclidean distance in meters and keeps
    pairs no farther than max_dist_m. Returns (pred_index, gt_index) pairs.
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    distance = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distance)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if distance[r, c] <= max_dist_m]


def gt_boxes(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible ground-truth boxes from load_annotations, one row per person and frame."""
    _check_columns(annotations, ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax", "visible"], "annotations")
    visible = annotations.loc[annotations["visible"]]
    return visible.loc[:, ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax"]].reset_index(drop=True)


def gt_positions(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible ground-truth feet from load_annotations, in meters.

    X_m and Y_m are the annotated ground positions, not a projection of the box.
    """
    _check_columns(annotations, ["frame_idx", "person_id", "X_m", "Y_m", "visible"], "annotations")
    visible = annotations.loc[annotations["visible"]]
    return visible.loc[:, ["frame_idx", "person_id", "X_m", "Y_m"]].reset_index(drop=True)


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


def match_tracks_ground(
    tracks: pd.DataFrame,
    gt: pd.DataFrame,
    max_dist_m: float = GROUND_MATCH_DISTANCE_M,
) -> pd.DataFrame:
    """Match every frame by foot position and return one row per pair.

    gt uses the ground-truth X_m, Y_m of people visible in this camera.
    Columns: frame_idx, track_id, person_id, distance_m.
    """
    _check_columns(tracks, ["frame_idx", "track_id", "X_m", "Y_m"], "tracks")
    _check_columns(gt, ["frame_idx", "person_id", "X_m", "Y_m"], "ground truth")
    pred_by_frame = {int(key): group for key, group in tracks.groupby("frame_idx", sort=False)}
    rows = []
    for frame_idx, gt_frame in gt.groupby("frame_idx", sort=True):
        pred_frame = pred_by_frame.get(int(frame_idx))
        if pred_frame is None:
            continue
        pred_xy = pred_frame[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
        gt_xy = gt_frame[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
        distance = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
        for p, g in match_frame_ground(pred_xy, gt_xy, max_dist_m):
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "track_id": int(pred_frame["track_id"].iloc[p]),
                    "person_id": int(gt_frame["person_id"].iloc[g]),
                    "distance_m": float(distance[p, g]),
                }
            )
    return pd.DataFrame(rows, columns=["frame_idx", "track_id", "person_id", "distance_m"])


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

    Columns: track_id, person_id, matched_frames, track_frames, and
    purity = matched_frames / track_frames. A track that never matches gets
    its own negative identity, -track_id, so two unmatched tracks are not
    treated as the same person. Event matching compares identities, so
    predicted track IDs need this link to ground-truth person IDs.
    """
    lengths = tracks.groupby("track_id").size().rename("track_frames")
    if matches.empty:
        best = pd.DataFrame(columns=["track_id", "person_id", "matched_frames"])
    else:
        counts = matches.groupby(["track_id", "person_id"]).size().rename("matched_frames").reset_index()
        counts = counts.sort_values(["track_id", "matched_frames", "person_id"], ascending=[True, False, True])
        best = counts.drop_duplicates("track_id", keep="first")
    table = lengths.reset_index().merge(best, on="track_id", how="left")
    missing = table["person_id"].isna()
    table.loc[missing, "person_id"] = -table.loc[missing, "track_id"].astype(int)
    table["person_id"] = table["person_id"].astype(int)
    table["matched_frames"] = table["matched_frames"].fillna(0).astype(int)
    table["purity"] = table["matched_frames"] / table["track_frames"]
    return table[["track_id", "person_id", "matched_frames", "track_frames", "purity"]]


def relabel_tracks(tracks: pd.DataFrame, idmap: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return tracks with a person_id on every row.

    person_id comes from idmap. A track that is missing from the map gets
    -track_id. When two rows in one frame would share a person_id, the row
    with the higher score is kept (the smaller track_id wins a tie). Returns
    the relabeled table and how many rows were dropped.
    """
    _check_columns(tracks, ["frame_idx", "track_id", "score"], "tracks")
    _check_columns(idmap, ["track_id", "person_id"], "id map")
    out = tracks.copy()
    person = out["track_id"].map(idmap.drop_duplicates("track_id").set_index("track_id")["person_id"])
    fallback = -out["track_id"].astype(int)
    out["person_id"] = person.fillna(pd.Series(fallback, index=out.index)).astype(int)
    ranked = out.sort_values(
        ["frame_idx", "person_id", "score", "track_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    kept = ranked.drop_duplicates(["frame_idx", "person_id"], keep="first")
    dropped = len(out) - len(kept)
    kept = kept.sort_values(["frame_idx", "track_id"], kind="mergesort").reset_index(drop=True)
    return kept, dropped


def _summary_from_matches(
    tracks: pd.DataFrame,
    gt: pd.DataFrame,
    matches: pd.DataFrame,
    quality_column: str,
    quality_name: str,
) -> dict[str, Any]:
    """Detection and identity scores shared by the IoU and ground-plane matchers.

    det_precision / det_recall: matched rows over predicted / ground-truth rows.
    id_switches: CLEAR-MOT ID switches.
    mota: 1 - (misses + false positives + ID switches) / ground-truth rows.
    """
    tp = len(matches)
    fp = len(tracks) - tp
    fn = len(gt) - tp
    id_switches = count_id_switches(matches)
    mapping = track_to_person_map(matches, tracks) if len(tracks) else pd.DataFrame(columns=["purity"])
    quality = float(matches[quality_column].mean()) if tp else float("nan")
    return {
        "pred_boxes": int(len(tracks)),
        "gt_boxes": int(len(gt)),
        "matched": int(tp),
        "det_precision": tp / len(tracks) if len(tracks) else float("nan"),
        "det_recall": tp / len(gt) if len(gt) else float("nan"),
        quality_name: quality,
        "id_switches": int(id_switches),
        "mota": 1.0 - (fn + fp + id_switches) / len(gt) if len(gt) else float("nan"),
        "pred_tracks": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        "gt_people": int(gt["person_id"].nunique()) if len(gt) else 0,
        "mean_track_purity": float(mapping["purity"].mean()) if len(mapping) else float("nan"),
    }


def tracking_summary(tracks: pd.DataFrame, gt: pd.DataFrame, iou_threshold: float = 0.5) -> dict[str, Any]:
    """Box-level detection and tracking scores for one tracks table.

    Pairs are boxes with IoU >= iou_threshold. mean_iou is the mean IoU of
    the matched pairs.
    """
    matches = match_tracks_to_gt(tracks, gt, iou_threshold)
    return _summary_from_matches(tracks, gt, matches, "iou", "mean_iou")


def _ground_summary_with_margin(
    tracks: pd.DataFrame,
    gt: pd.DataFrame,
    matches: pd.DataFrame,
    boundary_margin_m: float,
) -> dict[str, Any]:
    """Apply the boundary don't-care band after a full-set ground-plane match.

    Matching is unchanged. A match counts as a TP when the predicted foot or
    the ground-truth foot lies in the inner rectangle (annotated area shrunk
    by boundary_margin_m). An unmatched prediction or ground-truth person
    counts as FP / FN only when it is inside that inner rectangle. Unmatched
    points in the margin band are ignored.
    """
    margin = float(boundary_margin_m)
    pred_inner = in_ground_area(tracks["X_m"].to_numpy(), tracks["Y_m"].to_numpy(), margin)
    gt_inner = in_ground_area(gt["X_m"].to_numpy(), gt["Y_m"].to_numpy(), margin)
    if matches.empty:
        counted = matches.copy()
        matched_pred = np.zeros(len(tracks), dtype=bool)
        matched_gt = np.zeros(len(gt), dtype=bool)
    else:
        pred_keys = list(zip(tracks["frame_idx"].astype(int), tracks["track_id"].astype(int)))
        gt_keys = list(zip(gt["frame_idx"].astype(int), gt["person_id"].astype(int)))
        pred_index = {key: index for index, key in enumerate(pred_keys)}
        gt_index = {key: index for index, key in enumerate(gt_keys)}
        keep_rows = []
        matched_pred = np.zeros(len(tracks), dtype=bool)
        matched_gt = np.zeros(len(gt), dtype=bool)
        for row_index, row in enumerate(matches.itertuples(index=False)):
            p = pred_index[(int(row.frame_idx), int(row.track_id))]
            g = gt_index[(int(row.frame_idx), int(row.person_id))]
            matched_pred[p] = True
            matched_gt[g] = True
            if pred_inner[p] or gt_inner[g]:
                keep_rows.append(row_index)
        counted = matches.iloc[keep_rows].reset_index(drop=True)
    tp = len(counted)
    fp = int((~matched_pred & pred_inner).sum())
    fn = int((~matched_gt & gt_inner).sum())
    ignored_pred = int((~matched_pred & ~pred_inner).sum())
    ignored_gt = int((~matched_gt & ~gt_inner).sum())
    id_switches = count_id_switches(counted)
    mapping = track_to_person_map(matches, tracks) if len(tracks) else pd.DataFrame(columns=["purity"])
    quality = float(counted["distance_m"].mean()) if tp else float("nan")
    care_gt = int(gt_inner.sum())
    return {
        "boundary_margin_m": margin,
        "pred_boxes": int(len(tracks)),
        "gt_boxes": int(len(gt)),
        "care_gt_boxes": care_gt,
        "matched": int(tp),
        "ignored_pred": ignored_pred,
        "ignored_gt": ignored_gt,
        "det_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "det_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "mean_distance_m": quality,
        "id_switches": int(id_switches),
        "mota": 1.0 - (fn + fp + id_switches) / care_gt if care_gt else float("nan"),
        "pred_tracks": int(tracks["track_id"].nunique()) if len(tracks) else 0,
        "gt_people": int(gt["person_id"].nunique()) if len(gt) else 0,
        "mean_track_purity": float(mapping["purity"].mean()) if len(mapping) else float("nan"),
    }


def tracking_summary_ground(
    tracks: pd.DataFrame,
    gt: pd.DataFrame,
    max_dist_m: float = GROUND_MATCH_DISTANCE_M,
    boundary_margin_m: float = DEFAULT_BOUNDARY_MARGIN_M,
) -> dict[str, Any]:
    """Ground-plane detection and tracking scores for one tracks table.

    Pairs are feet within max_dist_m on the full ground area. Scores then use
    a don't-care band: the annotated rectangle is shrunk by boundary_margin_m
    on every side. margin 0 recovers the old behaviour (every unmatched foot
    counts). Default margin is 1 m.
    """
    matches = match_tracks_ground(tracks, gt, max_dist_m)
    if float(boundary_margin_m) == 0.0:
        summary = _summary_from_matches(tracks, gt, matches, "distance_m", "mean_distance_m")
        summary["boundary_margin_m"] = 0.0
        summary["care_gt_boxes"] = int(len(gt))
        summary["ignored_pred"] = 0
        summary["ignored_gt"] = 0
        return summary
    return _ground_summary_with_margin(tracks, gt, matches, boundary_margin_m)


def camera_ground_xy(cam_id: str | int) -> np.ndarray:
    """Camera center on the ground plane, in meters.

    The center is C = -R^T t. Only the X and Y coordinates are returned;
    height is not part of a ground-plane distance.
    """
    from src.geometry import load_camera_calibration

    _K, R, t = load_camera_calibration(DATA_ROOT, normalize_cam_id(cam_id))
    center = (-R.T @ np.asarray(t, dtype=np.float64).reshape(3, 1)).ravel()
    return center[:2].copy()


def nearest_gt_distance(tracks: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    """Distance from each predicted foot to the nearest ground-truth foot in that frame.

    This is not one-to-one: several predictions may share the same neighbor.
    A frame with no ground truth gets +inf. The array follows the row order
    of tracks.
    """
    _check_columns(tracks, ["frame_idx", "X_m", "Y_m"], "tracks")
    _check_columns(gt, ["frame_idx", "X_m", "Y_m"], "ground truth")
    distances = np.full(len(tracks), np.inf, dtype=np.float64)
    if tracks.empty:
        return distances
    gt_by_frame = {
        int(frame): group[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
        for frame, group in gt.groupby("frame_idx", sort=False)
    }
    frames = tracks["frame_idx"].to_numpy()
    xy = tracks[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
    for frame_idx in np.unique(frames):
        gt_xy = gt_by_frame.get(int(frame_idx))
        if gt_xy is None or len(gt_xy) == 0:
            continue
        rows = np.flatnonzero(frames == frame_idx)
        diff = xy[rows, None, :] - gt_xy[None, :, :]
        distances[rows] = np.linalg.norm(diff, axis=2).min(axis=1)
    return distances


def max_frame_iou(boxes: pd.DataFrame, others: pd.DataFrame) -> np.ndarray:
    """Max IoU of each box with the other boxes in the same frame.

    Frames with no other boxes score 0. The array follows the row order of boxes.
    """
    _check_columns(boxes, ["frame_idx", "xmin", "ymin", "xmax", "ymax"], "boxes")
    _check_columns(others, ["frame_idx", "xmin", "ymin", "xmax", "ymax"], "other boxes")
    scores = np.zeros(len(boxes), dtype=np.float64)
    if boxes.empty:
        return scores
    other_by_frame = {
        int(frame): group[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
        for frame, group in others.groupby("frame_idx", sort=False)
    }
    frames = boxes["frame_idx"].to_numpy()
    xyxy = boxes[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float64)
    for frame_idx in np.unique(frames):
        other_xyxy = other_by_frame.get(int(frame_idx))
        if other_xyxy is None or len(other_xyxy) == 0:
            continue
        rows = np.flatnonzero(frames == frame_idx)
        scores[rows] = box_iou(xyxy[rows], other_xyxy).max(axis=1)
    return scores


def classify_distant_feet(
    tracks: pd.DataFrame,
    gt_boxes: pd.DataFrame,
    nearest_m: np.ndarray,
    far_m: float = DISTANT_FOOT_M,
) -> np.ndarray:
    """Label in-area feet that are more than far_m from every ground-truth person.

    The first matching rule wins: cut_off (bottom within 5 px of the image
    bottom, or the box touches the left or right edge), partial (height/width
    below 1.5), foot_wrong (IoU >= 0.3 with a ground-truth box), no_person
    (IoU < 0.1 with every ground-truth box), otherwise other. Feet within
    far_m are labeled "".
    """
    _check_columns(tracks, ["xmin", "ymin", "xmax", "ymax"], "tracks")
    nearest = np.asarray(nearest_m, dtype=np.float64).reshape(-1)
    if len(nearest) != len(tracks):
        raise ValueError(f"{len(tracks)} tracks but {len(nearest)} distances")
    labels = np.full(len(tracks), "", dtype=object)
    distant = nearest > far_m
    if not np.any(distant):
        return labels
    max_iou = max_frame_iou(tracks, gt_boxes)
    width = np.maximum(tracks["xmax"].to_numpy(dtype=np.float64) - tracks["xmin"].to_numpy(dtype=np.float64), 1e-9)
    height = tracks["ymax"].to_numpy(dtype=np.float64) - tracks["ymin"].to_numpy(dtype=np.float64)
    cut_off = (
        (tracks["ymax"].to_numpy(dtype=np.float64) >= CUT_OFF_BOTTOM_Y)
        | (tracks["xmin"].to_numpy(dtype=np.float64) <= 0.0)
        | (tracks["xmax"].to_numpy(dtype=np.float64) >= IMAGE_WIDTH)
    )
    partial = (height / width) < PARTIAL_ASPECT_MAX
    labels[distant & ~cut_off & ~partial & (max_iou < NO_PERSON_IOU)] = "no_person"
    labels[distant & ~cut_off & ~partial & (max_iou >= FOOT_WRONG_IOU)] = "foot_wrong"
    labels[distant & ~cut_off & partial] = "partial"
    labels[distant & cut_off] = "cut_off"
    labels[distant & (labels == "")] = "other"
    return labels


def classify_missed_people(
    pred_boxes: pd.DataFrame,
    gt: pd.DataFrame,
    matches: pd.DataFrame,
    iou_threshold: float = FOOT_WRONG_IOU,
) -> pd.DataFrame:
    """Label ground-truth people who have no ground-plane match.

    detected_foot_wrong: some predicted box in that frame has IoU >= iou_threshold
    with their box. not_detected: no predicted box overlaps them that much.
    People who were matched are omitted. X_m and Y_m are copied when present.
    """
    _check_columns(gt, ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax"], "ground truth")
    if matches.empty:
        matched: set[tuple[int, int]] = set()
    else:
        _check_columns(matches, ["frame_idx", "person_id"], "matches")
        matched = set(zip(matches["frame_idx"].astype(int), matches["person_id"].astype(int)))
    keys = list(zip(gt["frame_idx"].astype(int), gt["person_id"].astype(int)))
    missed_rows = [index for index, key in enumerate(keys) if key not in matched]
    missed = gt.iloc[missed_rows].reset_index(drop=True)
    columns = ["frame_idx", "person_id", "max_iou", "reason"]
    if missed.empty:
        return pd.DataFrame(columns=columns + ["X_m", "Y_m"])
    max_iou = max_frame_iou(missed, pred_boxes)
    out = missed[["frame_idx", "person_id"]].copy()
    if "X_m" in missed.columns and "Y_m" in missed.columns:
        out["X_m"] = missed["X_m"].to_numpy(dtype=np.float64)
        out["Y_m"] = missed["Y_m"].to_numpy(dtype=np.float64)
    out["max_iou"] = max_iou
    out["reason"] = np.where(max_iou >= iou_threshold, "detected_foot_wrong", "not_detected")
    return out


def ground_error_components(
    pred_xy: np.ndarray,
    true_xy: np.ndarray,
    camera_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a foot error into along-view and sideways meters.

    The view direction is the ground ray from the camera center to the true
    position. along_m is positive when the predicted foot is closer to the
    camera than the truth. sideways_m is positive to the left of that ray
    (90 degrees counter-clockwise). A true position on the camera center
    has no direction and yields NaN.
    """
    pred = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    true = np.asarray(true_xy, dtype=np.float64).reshape(-1, 2)
    if len(pred) != len(true):
        raise ValueError(f"{len(pred)} predicted feet but {len(true)} true feet")
    camera = np.asarray(camera_xy, dtype=np.float64).reshape(2)
    view = true - camera
    norm = np.linalg.norm(view, axis=1)
    direction = view / np.maximum(norm, 1e-12)[:, None]
    delta = pred - true
    along = -(delta * direction).sum(axis=1)
    left = np.column_stack([-direction[:, 1], direction[:, 0]])
    sideways = (delta * left).sum(axis=1)
    invalid = norm < 1e-9
    along = np.where(invalid, np.nan, along)
    sideways = np.where(invalid, np.nan, sideways)
    return along, sideways


def matched_foot_coordinates(
    tracks: pd.DataFrame,
    gt: pd.DataFrame,
    max_dist_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """World coordinates of one-to-one ground matches, predicted then true."""
    matches = match_tracks_ground(tracks, gt, max_dist_m)
    if matches.empty:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    paired = matches.merge(
        tracks[["frame_idx", "track_id", "X_m", "Y_m"]],
        on=["frame_idx", "track_id"],
        how="left",
    )
    paired = paired.merge(
        gt[["frame_idx", "person_id", "X_m", "Y_m"]],
        on=["frame_idx", "person_id"],
        how="left",
        suffixes=("_pred", "_gt"),
    )
    pred_xy = paired[["X_m_pred", "Y_m_pred"]].to_numpy(dtype=np.float64)
    true_xy = paired[["X_m_gt", "Y_m_gt"]].to_numpy(dtype=np.float64)
    if np.isnan(pred_xy).any() or np.isnan(true_xy).any():
        raise ValueError("A ground-plane match has no foot coordinates.")
    return pred_xy, true_xy


def recall_by_camera_distance(
    gt: pd.DataFrame,
    matches: pd.DataFrame,
    camera_xy: np.ndarray,
    edges_m: tuple[float, float] = (10.0, 20.0),
) -> list[dict[str, Any]]:
    """Recall of ground-truth people in range bins from the camera.

    Distance is horizontal, from the camera's ground position to the
    annotated (X, Y). Bins are [0, near), [near, far), and [far, inf).
    A person counts as recalled when the matcher paired them with a foot.
    """
    _check_columns(gt, ["frame_idx", "person_id", "X_m", "Y_m"], "ground truth")
    camera = np.asarray(camera_xy, dtype=np.float64).reshape(2)
    distance = np.hypot(gt["X_m"].to_numpy(dtype=np.float64) - camera[0], gt["Y_m"].to_numpy(dtype=np.float64) - camera[1])
    if matches.empty:
        hit = np.zeros(len(gt), dtype=bool)
    else:
        matched = set(zip(matches["frame_idx"].astype(int), matches["person_id"].astype(int)))
        hit = np.array(
            [(int(frame), int(person)) in matched for frame, person in zip(gt["frame_idx"], gt["person_id"])],
            dtype=bool,
        )
    bin_id = np.digitize(distance, np.asarray(edges_m, dtype=np.float64))
    near, far = edges_m
    labels = (f"0-{near:g}", f"{near:g}-{far:g}", f">{far:g}")
    rows = []
    for index, label in enumerate(labels):
        chosen = bin_id == index
        count = int(chosen.sum())
        matched_count = int(hit[chosen].sum()) if count else 0
        rows.append(
            {
                "bin": label,
                "gt": count,
                "matched": matched_count,
                "recall": matched_count / count if count else float("nan"),
            }
        )
    return rows


def iter_track_files(folder: str | Path, cam_id: str) -> Iterable[Path]:
    """Track CSVs for one camera, named tracks_<cam>_<detector>_<tracker>.csv.

    Diagnostic files that end in _loose.csv are skipped so the main scores
    stay on the default tracker settings.
    """
    return sorted(
        path
        for path in Path(folder).glob(f"tracks_{cam_id}_*.csv")
        if not path.stem.endswith("_loose")
    )
