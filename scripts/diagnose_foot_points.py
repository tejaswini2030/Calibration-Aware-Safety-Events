#!/usr/bin/env python3
"""Diagnose box bottom-center feet. The main match distance stays 0.5 m.

1. Oracle: project the ground-truth boxes themselves and match at 0.5 m.
2. Distance from every in-area foot to the nearest ground-truth position.
3. Along-view and sideways bias of pairs matched at a looser 1.0 m.
4. Precision, recall, and MOTA at 0.5, 0.75, and 1.0 m.
5. Recall in 0-10 m, 10-20 m, and beyond 20 m from the camera.
6. Raw detections versus the trackers built on them.
7. Why feet farther than 1 m from every person are wrong, and why people are missed.
8. Whether "no_person" feet sit on people annotated but not visible in this camera.
9. Looser tracker settings for yolo11x (diagnostic files only; main tracks stay put).
10. For yolo11x + DeepSORT: a top-down no_person map and a grid of 24 random crops.

The histogram is outputs/tracks/nearest_gt_distance_<cam>.png.
Example frames are outputs/tracks/diagnose_<cam>_yolo11x_deepsort_*.png.
no_person map/crops are outputs/tracks/no_person_*_<cam>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ANNOTATED_FPS, DATA_ROOT, load_annotations, normalize_cam_id, project_root, resolve_dataset_root
from src.events import load_event_config
from src.tracking import (
    CUT_OFF_BOTTOM_Y,
    DISTANT_FOOT_CATEGORIES,
    DISTANT_FOOT_M,
    GROUND_MATCH_DISTANCE_M,
    GROUND_X_MAX_M,
    GROUND_X_MIN_M,
    GROUND_Y_MAX_M,
    GROUND_Y_MIN_M,
    TRACK_COLUMNS,
    ByteTrackTracker,
    DeepSortTracker,
    annotated_frames,
    camera_ground_xy,
    classify_distant_feet,
    classify_missed_people,
    drop_outside_ground_area,
    foot_points_world,
    ground_error_components,
    gt_boxes,
    gt_positions,
    in_ground_area,
    iter_track_files,
    match_tracks_ground,
    matched_foot_coordinates,
    max_frame_iou,
    nearest_gt_distance,
    recall_by_camera_distance,
    run_tracker,
    save_csv,
    tracking_summary_ground,
    validate_tracks,
)

MATCH_DISTANCE_M = GROUND_MATCH_DISTANCE_M
BIAS_DISTANCE_M = 1.0
THRESHOLDS_M = (0.5, 0.75, 1.0)
RANGE_EDGES_M = (10.0, 20.0)
HIST_MAX_M = 2.0
HIST_BIN_M = 0.05
ORACLE_NAME = "oracle"
DETECTION_CONFIDENCE = (0.1, 0.25, 0.3, 0.5)
EXAMPLE_MODEL = "yolo11x_deepsort"
EXAMPLE_CATEGORIES = ("cut_off", "foot_wrong", "no_person")
NO_PERSON_CROP_COUNT = 24
NO_PERSON_CROP_SEED = 0
NO_PERSON_CROP_SIZE = 128
ZONE_COLORS = {
    "near": (0.95, 0.45, 0.10),
    "medium": (0.15, 0.65, 0.30),
    "far": (0.20, 0.40, 0.85),
}
SENSITIVITY_DETECTOR = "yolo11x"
# Diagnostic trackers only. Main tracks_* files are left alone.
LOOSE_BYTE_TRACK = {"track_activation_threshold": 0.1, "minimum_consecutive_frames": 1}
LOOSE_DEEP_SORT = {"n_init": 1, "min_score": 0.1}
MAX_LOST_SECONDS = 2.0
LINE_COLORS = {
    "oracle": "#222222",
    "yolo11n_bytetrack": "#e07a2f",
    "yolo11n_deepsort": "#2e9b4f",
    "yolo11x_bytetrack": "#3d6fd8",
    "yolo11x_deepsort": "#7a4ea3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--tracks-dir", default=str(project_root() / "results" / "tracks"))
    parser.add_argument("--det-dir", default=str(project_root() / "results" / "detections"))
    return parser.parse_args()


def _oracle_boxes(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible ground-truth boxes, used as if a detector had found every person."""
    visible = annotations.loc[annotations["visible"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "frame_idx": visible["frame_idx"].astype(int),
            "frame_number": visible["frame_number"].astype(int),
            "time_sec": visible["time_sec"].astype(float),
            "track_id": visible["person_id"].astype(int),
            "xmin": visible["xmin"].astype(float),
            "ymin": visible["ymin"].astype(float),
            "xmax": visible["xmax"].astype(float),
            "ymax": visible["ymax"].astype(float),
            "score": 1.0,
        }
    )


def _prepare(tracks: pd.DataFrame, cam_id: str) -> tuple[pd.DataFrame, int]:
    """Project feet and drop those outside the annotated ground rectangle."""
    world = foot_points_world(tracks, cam_id)
    return drop_outside_ground_area(world)


def _distance_shares(distances: np.ndarray) -> dict[str, float]:
    """Share of feet within 0.5 m, from 0.5 to 1.0 m, and beyond 1.0 m."""
    count = len(distances)
    if count == 0:
        return {"n": 0, "within_0.5": float("nan"), "mid_0.5_1": float("nan"), "beyond_1": float("nan")}
    return {
        "n": count,
        "within_0.5": float(np.mean(distances <= 0.5)),
        "mid_0.5_1": float(np.mean((distances > 0.5) & (distances <= 1.0))),
        "beyond_1": float(np.mean(distances > 1.0)),
    }


def _histogram(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of feet in each 0.05 m bin from 0 to 2 m. The last bin includes 2 m."""
    edges = np.linspace(0.0, HIST_MAX_M, int(round(HIST_MAX_M / HIST_BIN_M)) + 1)
    counts, edges = np.histogram(distances[np.isfinite(distances)], bins=edges)
    share = counts / len(distances) if len(distances) else counts.astype(float)
    return edges, share


def _save_histogram(series: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [ORACLE_NAME] + [name for name in series if name != ORACLE_NAME]
    for name in names:
        edges, share = _histogram(series[name])
        centers = (edges[:-1] + edges[1:]) / 2.0
        ax.plot(
            centers,
            share,
            color=LINE_COLORS.get(name, None),
            linestyle="--" if name == ORACLE_NAME else "-",
            linewidth=2.0 if name == ORACLE_NAME else 1.5,
            label=name,
        )
    for gate in (0.5, 1.0):
        ax.axvline(gate, color="0.7", linewidth=0.8, zorder=0)
    ax.set_xlim(0.0, HIST_MAX_M)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("distance to nearest ground-truth foot (m)")
    ax.set_ylabel("share of feet")
    ax.set_title("Nearest ground-truth foot, 0.05 m bins")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bias(on_ground: pd.DataFrame, positions: pd.DataFrame, camera_xy: np.ndarray) -> dict[str, float]:
    """Median along-view and sideways error of pairs within 1.0 m."""
    pred_xy, true_xy = matched_foot_coordinates(on_ground, positions, BIAS_DISTANCE_M)
    along, sideways = ground_error_components(pred_xy, true_xy, camera_xy)
    return {
        "n": int(len(along)),
        "along_m": float(np.nanmedian(along)) if len(along) else float("nan"),
        "sideways_m": float(np.nanmedian(sideways)) if len(sideways) else float("nan"),
    }


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def main() -> None:
    """Print the five foot-point diagnostics and save the nearest-distance histogram."""
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    tracks_dir = Path(args.tracks_dir)
    files = list(iter_track_files(tracks_dir, cam_id))
    if not files:
        raise FileNotFoundError(f"No tracks_{cam_id}_*.csv in {tracks_dir}. Run scripts/run_tracking.py first.")

    annotations = load_annotations(resolve_dataset_root(args.data_root), cam_id)
    positions = gt_positions(annotations)
    camera_xy = camera_ground_xy(cam_id)
    print(f"{cam_id} camera ground position ({camera_xy[0]:.3f}, {camera_xy[1]:.3f}) m")
    print(f"Main ground-plane match stays at {MATCH_DISTANCE_M:g} m. Bias pairs use {BIAS_DISTANCE_M:g} m.")

    oracle_boxes = _oracle_boxes(annotations)
    oracle_feet, oracle_dropped = _prepare(oracle_boxes, cam_id)
    oracle_summary = tracking_summary_ground(oracle_feet, positions, MATCH_DISTANCE_M)
    print(
        f"\nOracle (ground-truth boxes, bottom-center, H*, {MATCH_DISTANCE_M:g} m): "
        f"recall {oracle_summary['det_recall']:.3f}, "
        f"mean matched distance {oracle_summary['mean_distance_m']:.3f} m, "
        f"dropped {oracle_dropped} feet outside the ground area"
    )

    models: list[dict[str, object]] = []
    for path in files:
        name = path.stem.removeprefix(f"tracks_{cam_id}_")
        tracks = pd.read_csv(path)
        validate_tracks(tracks)
        last_frame = int(tracks["frame_idx"].max())
        covered = positions.loc[positions["frame_idx"] <= last_frame].reset_index(drop=True)
        on_ground, dropped = _prepare(tracks, cam_id)
        print(f"{name}: dropped {dropped} predicted rows outside the ground area")
        models.append({"name": name, "feet": on_ground, "boxes": tracks, "covered": covered})

    distances = {ORACLE_NAME: nearest_gt_distance(oracle_feet, positions)}
    distances.update({str(model["name"]): nearest_gt_distance(model["feet"], model["covered"]) for model in models})
    shares = pd.DataFrame(
        [{"model": name, **_distance_shares(values)} for name, values in distances.items()]
    )
    _print_table("Share of in-area feet by distance to the nearest ground-truth position", shares)
    histogram_path = project_root() / "outputs" / "tracks" / f"nearest_gt_distance_{cam_id}.png"
    _save_histogram(distances, histogram_path)
    print(f"Saved {histogram_path.name}")

    bias_rows = [{"model": ORACLE_NAME, **_bias(oracle_feet, positions, camera_xy)}]
    bias_rows.extend({"model": model["name"], **_bias(model["feet"], model["covered"], camera_xy)} for model in models)
    _print_table(
        "Median error of pairs within 1.0 m. along > 0 means the foot is closer to the camera than the truth",
        pd.DataFrame(bias_rows),
    )

    threshold_rows = []
    for model in models:
        row: dict[str, float | str] = {"model": str(model["name"])}
        for threshold in THRESHOLDS_M:
            summary = tracking_summary_ground(model["feet"], model["covered"], threshold)
            label = f"{threshold:.2f}"
            row[f"P_{label}"] = summary["det_precision"]
            row[f"R_{label}"] = summary["det_recall"]
            row[f"MOTA_{label}"] = summary["mota"]
        threshold_rows.append(row)
    _print_table("Ground-plane precision, recall, and MOTA at three match distances", pd.DataFrame(threshold_rows))

    range_rows = []
    oracle_matches = match_tracks_ground(oracle_feet, positions, MATCH_DISTANCE_M)
    for item in recall_by_camera_distance(positions, oracle_matches, camera_xy, RANGE_EDGES_M):
        range_rows.append({"model": ORACLE_NAME, **item})
    for model in models:
        matches = match_tracks_ground(model["feet"], model["covered"], MATCH_DISTANCE_M)
        for item in recall_by_camera_distance(model["covered"], matches, camera_xy, RANGE_EDGES_M):
            range_rows.append({"model": model["name"], **item})
    _print_table(
        f"Recall at {MATCH_DISTANCE_M:g} m by horizontal distance from the camera to the true position",
        pd.DataFrame(range_rows),
    )
    _detector_vs_tracker(args, cam_id, models, positions)
    _error_breakdown(args, cam_id, models, annotations, positions, camera_xy)
    _tracker_sensitivity(args, cam_id, models, positions)


def _detector_name(track_name: str, detectors: list[str]) -> str | None:
    """Return the detector a track file was built from, matching the longest name."""
    matches = [detector for detector in detectors if track_name.startswith(detector + "_")]
    if not matches:
        return None
    return max(matches, key=len)


def _score_detections(detections: pd.DataFrame, cam_id: str, positions: pd.DataFrame) -> dict[str, float]:
    """Ground-plane precision and recall for one confidence subset, at 0.5 m."""
    labeled = detections.copy()
    labeled["track_id"] = np.arange(len(labeled), dtype=int)
    feet, _dropped = _prepare(labeled, cam_id)
    covered = positions.loc[positions["frame_idx"] <= int(detections["frame_idx"].max())]
    summary = tracking_summary_ground(feet, covered, MATCH_DISTANCE_M)
    return {"det_P": summary["det_precision"], "det_R": summary["det_recall"]}


def _detector_vs_tracker(
    args: argparse.Namespace,
    cam_id: str,
    models: list[dict[str, object]],
    positions: pd.DataFrame,
) -> None:
    """Compare raw detection recall with the trackers built on each detector."""
    det_dir = Path(args.det_dir)
    files = sorted(det_dir.glob(f"det_{cam_id}_*.csv"))
    files = [path for path in files if not path.stem.endswith("_summary")]
    if not files:
        raise FileNotFoundError(f"No det_{cam_id}_*.csv in {det_dir}. Run scripts/run_detection.py first.")
    tracker_recall = {
        str(model["name"]): tracking_summary_ground(model["feet"], model["covered"], MATCH_DISTANCE_M)["det_recall"]
        for model in models
    }
    detectors = [path.stem.removeprefix(f"det_{cam_id}_") for path in files]
    rows = []
    for path, detector in zip(files, detectors):
        detections = pd.read_csv(path)
        built = [name for name in tracker_recall if _detector_name(name, detectors) == detector]
        for threshold in DETECTION_CONFIDENCE:
            kept = detections.loc[detections["score"] >= threshold]
            scores = _score_detections(kept, cam_id, positions) if len(kept) else {"det_P": float("nan"), "det_R": float("nan")}
            row: dict[str, float | str] = {"detector": detector, "conf": threshold, **scores}
            for name in built:
                tracker = name.removeprefix(detector + "_")
                row[f"R_{tracker}"] = tracker_recall[name]
                row[f"gap_{tracker}"] = scores["det_R"] - tracker_recall[name]
            rows.append(row)
    _print_table(
        "Raw detections at 0.5 m, beside tracker recall. Gap = detection recall − tracker recall",
        pd.DataFrame(rows),
    )


def _category_row(model: str, labels: np.ndarray) -> dict[str, float | str]:
    distant = labels[labels != ""]
    count = len(distant)
    row: dict[str, float | str] = {"model": model, "n": count}
    for category in DISTANT_FOOT_CATEGORIES:
        row[category] = float(np.mean(distant == category)) if count else float("nan")
    return row


def _range_labels(x_m: np.ndarray, y_m: np.ndarray, camera_xy: np.ndarray) -> np.ndarray:
    distance = np.hypot(x_m - camera_xy[0], y_m - camera_xy[1])
    bin_id = np.digitize(distance, np.asarray(RANGE_EDGES_M, dtype=np.float64))
    names = (f"0-{RANGE_EDGES_M[0]:g}", f"{RANGE_EDGES_M[0]:g}-{RANGE_EDGES_M[1]:g}", f">{RANGE_EDGES_M[1]:g}")
    return np.asarray(names, dtype=object)[bin_id]


def _miss_rows(model: str, missed: pd.DataFrame, camera_xy: np.ndarray) -> list[dict[str, float | str]]:
    if missed.empty:
        return [{"model": model, "bin": "all", "n": 0, "detected_foot_wrong": float("nan"), "not_detected": float("nan")}]
    bins = _range_labels(missed["X_m"].to_numpy(dtype=float), missed["Y_m"].to_numpy(dtype=float), camera_xy)
    groups = [("all", np.ones(len(missed), dtype=bool))]
    groups.extend((name, bins == name) for name in (f"0-{RANGE_EDGES_M[0]:g}", f"{RANGE_EDGES_M[0]:g}-{RANGE_EDGES_M[1]:g}", f">{RANGE_EDGES_M[1]:g}"))
    rows = []
    for name, chosen in groups:
        reasons = missed.iloc[np.flatnonzero(chosen)]["reason"]
        count = int(len(reasons))
        rows.append(
            {
                "model": model,
                "bin": name,
                "n": count,
                "detected_foot_wrong": float(np.mean(reasons == "detected_foot_wrong")) if count else float("nan"),
                "not_detected": float(np.mean(reasons == "not_detected")) if count else float("nan"),
            }
        )
    return rows


def _nearby_gt(frame_gt: pd.DataFrame, box: pd.Series, foot_xy: np.ndarray) -> pd.DataFrame:
    """Ground-truth boxes near this prediction, in the image or on the ground."""
    if frame_gt.empty:
        return frame_gt
    center_x = (frame_gt["xmin"].to_numpy(dtype=float) + frame_gt["xmax"].to_numpy(dtype=float)) / 2.0
    center_y = (frame_gt["ymin"].to_numpy(dtype=float) + frame_gt["ymax"].to_numpy(dtype=float)) / 2.0
    pred_x = (float(box["xmin"]) + float(box["xmax"])) / 2.0
    pred_y = (float(box["ymin"]) + float(box["ymax"])) / 2.0
    near_pixels = np.hypot(center_x - pred_x, center_y - pred_y) <= 250.0
    near_ground = np.hypot(frame_gt["X_m"].to_numpy(dtype=float) - foot_xy[0], frame_gt["Y_m"].to_numpy(dtype=float) - foot_xy[1]) <= 4.0
    prediction = pd.DataFrame([{column: box[column] for column in ("frame_idx", "xmin", "ymin", "xmax", "ymax")}])
    overlap = max_frame_iou(frame_gt, prediction) > 0.0
    keep = near_pixels | near_ground | overlap
    if not np.any(keep):
        ground = np.hypot(frame_gt["X_m"].to_numpy(dtype=float) - foot_xy[0], frame_gt["Y_m"].to_numpy(dtype=float) - foot_xy[1])
        keep = np.zeros(len(frame_gt), dtype=bool)
        keep[int(np.argmin(ground))] = True
    return frame_gt.iloc[np.flatnonzero(keep)]


def _crop_around(image: np.ndarray, regions: np.ndarray, margin: int = 80) -> np.ndarray:
    """Crop to the boxes being shown, with a margin, so the example stays readable."""
    height, width = image.shape[:2]
    x0 = max(0, int(np.min(regions[:, 0])) - margin)
    y0 = max(0, int(np.min(regions[:, 1])) - margin)
    x1 = min(width, int(np.max(regions[:, 2])) + margin)
    y1 = min(height, int(np.max(regions[:, 3])) + margin)
    if x1 - x0 < 240:
        extra = (240 - (x1 - x0)) // 2
        x0 = max(0, x0 - extra)
        x1 = min(width, x0 + 240)
    if y1 - y0 < 240:
        extra = (240 - (y1 - y0)) // 2
        y0 = max(0, y0 - extra)
        y1 = min(height, y0 + 240)
    return image[y0:y1, x0:x1]


def _draw_example(image_path: str, box: pd.Series, nearby: pd.DataFrame, title: str, path: Path) -> None:
    """One predicted box, its bottom-center foot, and the nearby ground-truth boxes."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    for row in nearby.itertuples(index=False):
        cv2.rectangle(image, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (255, 255, 255), 1)
    color = (0, 180, 255)
    top_left = (int(box["xmin"]), int(box["ymin"]))
    cv2.rectangle(image, top_left, (int(box["xmax"]), int(box["ymax"])), color, 2)
    foot = (int(round((float(box["xmin"]) + float(box["xmax"])) / 2.0)), int(round(float(box["ymax"]))))
    cv2.circle(image, foot, 7, (0, 0, 0), -1)
    cv2.circle(image, foot, 5, (0, 255, 255), -1)
    label = f"{float(box['score']):.2f}"
    origin = (top_left[0], max(16, top_left[1] - 8))
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    regions = [[float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])]]
    if len(nearby):
        regions.extend(nearby[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float).tolist())
    image = _crop_around(image, np.asarray(regions, dtype=float), margin=120)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (text_width, text_height), baseline = cv2.getTextSize(title, font, scale, thickness)
    width = max(image.shape[1], text_width + 16)
    if image.shape[1] < width:
        padded = np.zeros((image.shape[0], width, 3), dtype=np.uint8)
        padded[:, : image.shape[1]] = image
        image = padded
    banner = np.zeros((text_height + baseline + 14, width, 3), dtype=np.uint8)
    image = np.vstack([banner, image])
    cv2.putText(image, title, (8, text_height + 8), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def _pick_example(cases: pd.DataFrame, category: str) -> pd.Series:
    """A high-score example. foot_wrong prefers the box that overlaps a person most."""
    chosen = cases.loc[cases["category"] == category]
    if chosen.empty:
        raise ValueError(f"No {category} foot for {EXAMPLE_MODEL}")
    if category == "cut_off":
        bottom = chosen.loc[chosen["ymax"] >= CUT_OFF_BOTTOM_Y]
        if len(bottom):
            chosen = bottom
    sort_by = ["max_iou", "score"] if category == "foot_wrong" else ["score"]
    return chosen.sort_values(sort_by, ascending=False).iloc[0]


def _save_examples(
    cam_id: str,
    cases: pd.DataFrame,
    people: pd.DataFrame,
    image_by_frame: pd.Series,
    output_dir: Path,
) -> None:
    for category in EXAMPLE_CATEGORIES:
        example = _pick_example(cases, category)
        frame_gt = people.loc[people["frame_idx"] == int(example["frame_idx"])]
        foot_xy = np.array([float(example["X_m"]), float(example["Y_m"])])
        nearby = _nearby_gt(frame_gt, example, foot_xy)
        title = f"{cam_id} {EXAMPLE_MODEL} {category} frame {int(example['frame_idx'])}"
        path = output_dir / f"diagnose_{cam_id}_{EXAMPLE_MODEL}_{category}.png"
        _draw_example(str(image_by_frame.loc[int(example["frame_idx"])]), example, nearby, title, path)
        print(f"Saved {path.name}")


def _save_no_person_map(
    cam_id: str,
    no_person: pd.DataFrame,
    positions: pd.DataFrame,
    zones: list[dict],
    path: Path,
) -> None:
    """Top-down ground area: grey GT feet, red no_person feet, outlined zones."""
    fig, ax = plt.subplots(figsize=(7, 9))
    if len(positions):
        ax.scatter(
            positions["X_m"],
            positions["Y_m"],
            s=2,
            c="0.75",
            alpha=0.35,
            linewidths=0,
            label="ground truth",
            rasterized=True,
            zorder=1,
        )
    if len(no_person):
        ax.scatter(
            no_person["X_m"],
            no_person["Y_m"],
            s=8,
            c="#d62728",
            alpha=0.55,
            linewidths=0,
            label=f"no_person ({len(no_person)})",
            rasterized=True,
            zorder=3,
        )
    for zone in zones:
        polygon = np.asarray(zone["polygon"], dtype=float)
        color = ZONE_COLORS.get(str(zone["id"]), (0.2, 0.2, 0.2))
        closed = np.vstack([polygon, polygon[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.0, zorder=2)
        center = polygon.mean(axis=0)
        ax.text(center[0], center[1], str(zone["name"]), ha="center", va="center", fontsize=9, zorder=4)
    ax.set_xlim(GROUND_X_MIN_M, GROUND_X_MAX_M)
    ax.set_ylim(GROUND_Y_MIN_M, GROUND_Y_MAX_M)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{cam_id} {EXAMPLE_MODEL}: no_person feet on the ground plane")
    ax.legend(loc="upper left", markerscale=2, frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def _box_crop(image: np.ndarray, row: pd.Series, pad: int = 4) -> np.ndarray:
    """Crop one box from the frame and resize it to a square cell."""
    height, width = image.shape[:2]
    x0 = max(0, int(np.floor(float(row["xmin"]))) - pad)
    y0 = max(0, int(np.floor(float(row["ymin"]))) - pad)
    x1 = min(width, int(np.ceil(float(row["xmax"]))) + pad)
    y1 = min(height, int(np.ceil(float(row["ymax"]))) + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((NO_PERSON_CROP_SIZE, NO_PERSON_CROP_SIZE, 3), dtype=np.uint8)
    crop = image[y0:y1, x0:x1]
    return cv2.resize(crop, (NO_PERSON_CROP_SIZE, NO_PERSON_CROP_SIZE), interpolation=cv2.INTER_AREA)


def _save_no_person_crops(
    cam_id: str,
    no_person: pd.DataFrame,
    image_by_frame: pd.Series,
    path: Path,
) -> None:
    """Grid of random no_person boxes, labeled with score and frame_idx."""
    if no_person.empty:
        raise ValueError(f"No no_person rows for {EXAMPLE_MODEL}")
    sample = no_person.sample(
        n=min(NO_PERSON_CROP_COUNT, len(no_person)),
        random_state=NO_PERSON_CROP_SEED,
    ).reset_index(drop=True)
    cols = 6
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes_flat = np.atleast_1d(axes).ravel()
    cache: dict[int, np.ndarray] = {}
    for index, axis in enumerate(axes_flat):
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= len(sample):
            axis.axis("off")
            continue
        row = sample.iloc[index]
        frame_idx = int(row["frame_idx"])
        if frame_idx not in cache:
            image = cv2.imread(str(image_by_frame.loc[frame_idx]), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read frame {frame_idx}")
            cache[frame_idx] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = _box_crop(cache[frame_idx], row)
        axis.imshow(crop)
        axis.set_title(f"f{frame_idx}  {float(row['score']):.2f}", fontsize=8)
        for spine in axis.spines.values():
            spine.set_color("#d62728")
            spine.set_linewidth(1.5)
    fig.suptitle(f"{cam_id} {EXAMPLE_MODEL}: {len(sample)} random no_person crops (seed {NO_PERSON_CROP_SEED})")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def _invisible_in_area(annotations: pd.DataFrame) -> pd.DataFrame:
    """Annotated people inside the ground rectangle who are not visible in this camera."""
    hidden = annotations.loc[~annotations["visible"], ["frame_idx", "person_id", "X_m", "Y_m"]].copy()
    keep = in_ground_area(hidden["X_m"].to_numpy(), hidden["Y_m"].to_numpy())
    return hidden.loc[keep].reset_index(drop=True)


def _print_invisible_counts(invisible: pd.DataFrame, n_frames: int) -> None:
    """How many annotated people sit in the area but are marked not visible in this camera."""
    counts = invisible.groupby("frame_idx").size()
    if n_frames <= 0:
        print("No frames to count not-visible people against.")
        return
    per_frame = np.zeros(n_frames, dtype=int)
    for frame_idx, count in counts.items():
        if 0 <= int(frame_idx) < n_frames:
            per_frame[int(frame_idx)] = int(count)
    print(
        f"\nAnnotated people inside the ground area but not visible in this camera: "
        f"mean {per_frame.mean():.2f} per frame "
        f"(min {per_frame.min()}, max {per_frame.max()}, "
        f"frames with at least one {int((per_frame > 0).sum())}/{n_frames})"
    )


def _no_person_near_invisible(
    models: list[dict[str, object]],
    annotations: pd.DataFrame,
    boxes: pd.DataFrame,
) -> None:
    """Ask whether no_person feet land on people who are annotated but not visible here."""
    invisible = _invisible_in_area(annotations)
    all_people = annotations.loc[:, ["frame_idx", "person_id", "X_m", "Y_m"]].copy()
    n_frames = int(annotations["frame_idx"].max()) + 1 if len(annotations) else 0
    _print_invisible_counts(invisible, n_frames)
    rows = []
    for model in models:
        name = str(model["name"])
        feet = model["feet"]
        nearest_visible = nearest_gt_distance(feet, model["covered"])
        labels = classify_distant_feet(feet, boxes, nearest_visible, DISTANT_FOOT_M)
        no_person = feet.loc[labels == "no_person"].reset_index(drop=True)
        nearest_any = nearest_gt_distance(no_person, all_people)
        nearest_hidden = nearest_gt_distance(no_person, invisible)
        count = len(no_person)
        rows.append(
            {
                "model": name,
                "n": count,
                "within_0.5_any": float(np.mean(nearest_any <= 0.5)) if count else float("nan"),
                "within_1.0_any": float(np.mean(nearest_any <= 1.0)) if count else float("nan"),
                "within_0.5_not_visible": float(np.mean(nearest_hidden <= 0.5)) if count else float("nan"),
                "within_1.0_not_visible": float(np.mean(nearest_hidden <= 1.0)) if count else float("nan"),
            }
        )
    _print_table(
        "no_person feet vs annotated people (any = visible or not; not_visible = marked invisible in this camera)",
        pd.DataFrame(rows),
    )


def _score_track_table(tracks: pd.DataFrame, cam_id: str, positions: pd.DataFrame) -> dict[str, float]:
    """Ground-plane summary used for the default-versus-loose comparison."""
    feet, _dropped = _prepare(tracks, cam_id)
    covered = positions.loc[positions["frame_idx"] <= int(tracks["frame_idx"].max())]
    return tracking_summary_ground(feet, covered, MATCH_DISTANCE_M)


def _tracker_sensitivity(
    args: argparse.Namespace,
    cam_id: str,
    models: list[dict[str, object]],
    positions: pd.DataFrame,
) -> None:
    """Re-track yolo11x with looser settings and score next to the defaults.

    Writes tracks_<cam>_yolo11x_*_loose.csv so the main track files stay unchanged.
    """
    det_path = Path(args.det_dir) / f"det_{cam_id}_{SENSITIVITY_DETECTOR}.csv"
    if not det_path.is_file():
        print(f"\nNo {det_path.name}; skipping tracker sensitivity.")
        return
    detections = pd.read_csv(det_path)
    frames = annotated_frames(args.data_root, cam_id)
    used_frames = frames.loc[frames["frame_idx"] <= detections["frame_idx"].max()]
    out_dir = Path(args.tracks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_scores = {
        str(model["name"]): tracking_summary_ground(model["feet"], model["covered"], MATCH_DISTANCE_M)
        for model in models
        if str(model["name"]).startswith(SENSITIVITY_DETECTOR + "_")
    }
    loose_runners = [
        (
            "bytetrack",
            "bytetrack_loose",
            ByteTrackTracker(fps=ANNOTATED_FPS, max_lost_seconds=MAX_LOST_SECONDS, **LOOSE_BYTE_TRACK),
        ),
        (
            "deepsort",
            "deepsort_loose",
            DeepSortTracker(fps=ANNOTATED_FPS, max_lost_seconds=MAX_LOST_SECONDS, **LOOSE_DEEP_SORT),
        ),
    ]
    rows = []
    for default_name, loose_name, tracker in loose_runners:
        default_key = f"{SENSITIVITY_DETECTOR}_{default_name}"
        default = default_scores.get(default_key)
        if default is None:
            print(f"Missing default tracks for {default_key}; skipping that pair.")
            continue
        out = out_dir / f"tracks_{cam_id}_{SENSITIVITY_DETECTOR}_{loose_name}.csv"
        if out.is_file():
            print(f"Reusing {out.name}")
            tracks = pd.read_csv(out)
            validate_tracks(tracks)
        else:
            print(f"Running {SENSITIVITY_DETECTOR} + {loose_name}...")
            tracks = run_tracker(detections, used_frames, tracker)
            validate_tracks(tracks)
            save_csv(tracks, TRACK_COLUMNS, out)
            print(f"  -> {out.name} ({len(tracks)} rows, {tracks['track_id'].nunique()} track IDs)")
        loose = _score_track_table(tracks, cam_id, positions)
        for label, summary in (("default", default), ("loose", loose)):
            rows.append(
                {
                    "tracker": default_name,
                    "setting": label,
                    "precision": summary["det_precision"],
                    "recall": summary["det_recall"],
                    "id_switches": summary["id_switches"],
                    "mota": summary["mota"],
                    "pred_tracks": summary["pred_tracks"],
                }
            )
    _print_table(
        f"{SENSITIVITY_DETECTOR} tracker sensitivity at {MATCH_DISTANCE_M:g} m "
        f"(loose ByteTrack activation=0.1 consecutive=1; loose DeepSORT n_init=1 min_score=0.1)",
        pd.DataFrame(rows),
    )


def _error_breakdown(
    args: argparse.Namespace,
    cam_id: str,
    models: list[dict[str, object]],
    annotations: pd.DataFrame,
    positions: pd.DataFrame,
    camera_xy: np.ndarray,
) -> None:
    """Classify distant feet and missed people, and save three example frames."""
    boxes = gt_boxes(annotations)
    people = annotations.loc[annotations["visible"], ["frame_idx", "person_id", "xmin", "ymin", "xmax", "ymax", "X_m", "Y_m"]].reset_index(drop=True)
    distant_rows = []
    miss_rows = []
    example_cases: pd.DataFrame | None = None
    for model in models:
        name = str(model["name"])
        feet = model["feet"]
        nearest = nearest_gt_distance(feet, model["covered"])
        labels = classify_distant_feet(feet, boxes, nearest, DISTANT_FOOT_M)
        distant_rows.append(_category_row(name, labels))
        last_frame = int(model["covered"]["frame_idx"].max()) if len(model["covered"]) else -1
        covered_people = people.loc[people["frame_idx"] <= last_frame]
        matches = match_tracks_ground(feet, model["covered"], MATCH_DISTANCE_M)
        missed = classify_missed_people(model["boxes"], covered_people, matches)
        miss_rows.extend(_miss_rows(name, missed, camera_xy))
        if name == EXAMPLE_MODEL:
            cases = feet.copy()
            cases["category"] = labels
            cases["max_iou"] = max_frame_iou(feet, boxes)
            example_cases = cases.loc[cases["category"].isin(EXAMPLE_CATEGORIES)].reset_index(drop=True)
    _print_table(f"Share of in-area feet more than {DISTANT_FOOT_M:g} m from every ground-truth person", pd.DataFrame(distant_rows))
    _print_table("Share of ground-truth people missed at 0.5 m", pd.DataFrame(miss_rows))
    _no_person_near_invisible(models, annotations, boxes)
    if example_cases is None:
        print(f"No {EXAMPLE_MODEL} tracks, so no example frames were saved.")
        return
    frames = annotated_frames(args.data_root, cam_id)
    image_by_frame = frames.set_index("frame_idx")["image_path"]
    output_dir = project_root() / "outputs" / "tracks"
    _save_examples(cam_id, example_cases, people, image_by_frame, output_dir)
    no_person = example_cases.loc[example_cases["category"] == "no_person"].reset_index(drop=True)
    zones = list(load_event_config(project_root() / "configs" / "events.yaml")["zones"])
    _save_no_person_map(cam_id, no_person, positions, zones, output_dir / f"no_person_map_{cam_id}.png")
    _save_no_person_crops(cam_id, no_person, image_by_frame, output_dir / f"no_person_crops_{cam_id}.png")


if __name__ == "__main__":
    main()
