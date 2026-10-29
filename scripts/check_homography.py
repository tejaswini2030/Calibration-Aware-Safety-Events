#!/usr/bin/env python3
"""Overlay each camera's ground-truth homography on its first annotated frame."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    CAMERA_TO_CALIB,
    DATA_ROOT,
    position_id_to_world,
    project_root,
    resolve_dataset_root,
    view_num_for_camera,
)
from src.geometry import ground_homography, load_camera_calibration, world_to_image

# Ground rectangle from README.txt: origin (-3, -9) m, 12 m by 36 m.
X_MIN_M, X_MAX_M = -3.0, 9.0
Y_MIN_M, Y_MAX_M = -9.0, 27.0
GRID_STEP_M = 1.0
SAMPLE_STEP_M = 0.1
MIN_DEPTH_M = 0.05
# A correct camera agrees with the box bottoms to roughly 10 px on average.
# A wrong calibration mapping misses by a large fraction of the frame.
LARGE_MEAN_ERROR_PX = 50.0
GRID_COLOR = (220, 170, 40)
GREEN = (0, 220, 0)
RED = (0, 0, 255)


def first_annotation(dataset: Path) -> tuple[Path, list[dict]]:
    """Load the earliest annotation JSON."""
    files = sorted((dataset / "annotations_positions").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No annotation JSON files in {dataset / 'annotations_positions'}")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of people in {files[0]}")
    return files[0], payload


def frame_image(dataset: Path, cam_id: str, frame_stem: str) -> Path:
    """Return the image whose filename stem matches an annotation frame."""
    folder = dataset / "Image_subsets" / cam_id
    matches = [
        path
        for path in sorted(folder.glob(f"{frame_stem}.*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not matches:
        raise FileNotFoundError(f"No image for frame {frame_stem} in {folder}")
    return matches[0]


def visible_ground_points(people: list[dict], view_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ground positions and box bottom-centers for one camera view.

    Ground positions are (N, 2) meters. Bottom-centers are (N, 2) pixels.
    """
    world: list[list[float]] = []
    feet: list[list[float]] = []
    for person in people:
        view = next((item for item in person.get("views", []) if item.get("viewNum") == view_num), None)
        if view is None:
            continue
        xmin, ymin = int(view["xmin"]), int(view["ymin"])
        xmax, ymax = int(view["xmax"]), int(view["ymax"])
        if min(xmin, ymin, xmax, ymax) < 0:
            continue
        x_m, y_m = position_id_to_world(person["positionID"])
        world.append([x_m, y_m])
        feet.append([(xmin + xmax) / 2.0, float(ymax)])
    if not world:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(world, dtype=np.float64), np.asarray(feet, dtype=np.float64)


def camera_depth(R: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Return camera-frame Z for ground points. Positive is in front of the camera."""
    world = np.column_stack([xy, np.zeros(len(xy))])
    return (world @ R.T + t.ravel())[:, 2]


def project_masked(
    H: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    xy: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground points and mark those behind the camera or far outside the image."""
    margin = float(max(width, height))
    depth = camera_depth(R, t, xy)
    homogeneous = np.column_stack([xy, np.ones(len(xy))])
    mapped = homogeneous @ np.asarray(H, dtype=np.float64).T
    scale = mapped[:, 2]
    uv = np.full((len(xy), 2), np.nan, dtype=np.float64)
    finite = np.abs(scale) > 1e-8
    uv[finite] = mapped[finite, :2] / scale[finite, None]
    inside = (
        (uv[:, 0] >= -margin)
        & (uv[:, 0] <= width - 1 + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] <= height - 1 + margin)
    )
    valid = (depth > MIN_DEPTH_M) & finite & np.isfinite(uv).all(axis=1) & inside
    return uv, valid


def grid_polylines(x_min: float, x_max: float, y_min: float, y_max: float) -> list[np.ndarray]:
    """Sample a 1 m ground grid. Each item is an (N, 2) polyline in meters."""
    xs = np.arange(x_min, x_max + GRID_STEP_M * 0.5, GRID_STEP_M)
    ys = np.arange(y_min, y_max + GRID_STEP_M * 0.5, GRID_STEP_M)
    lines: list[np.ndarray] = []
    for x in xs:
        y = np.arange(y_min, y_max + SAMPLE_STEP_M * 0.5, SAMPLE_STEP_M)
        lines.append(np.column_stack([np.full(y.shape, x), y]))
    for y in ys:
        x = np.arange(x_min, x_max + SAMPLE_STEP_M * 0.5, SAMPLE_STEP_M)
        lines.append(np.column_stack([x, np.full(x.shape, y)]))
    return lines


def draw_polyline(image: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> None:
    """Draw each contiguous run of valid samples. A ground line is straight in the image."""
    segment: list[tuple[int, int]] = []
    for point, ok in zip(uv, valid):
        if ok:
            segment.append((int(round(float(point[0]))), int(round(float(point[1])))))
            continue
        _stroke(image, segment)
        segment = []
    _stroke(image, segment)


def _stroke(image: np.ndarray, segment: list[tuple[int, int]]) -> None:
    for start, end in zip(segment, segment[1:]):
        cv2.line(image, start, end, GRID_COLOR, 1, cv2.LINE_AA)


def draw_dots(image: np.ndarray, projected: np.ndarray, feet: np.ndarray) -> None:
    """Draw green box-bottom dots and red projected ground-truth dots."""
    for foot, point in zip(feet, projected):
        green = (int(round(float(foot[0]))), int(round(float(foot[1]))))
        red = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(image, green, 6, GREEN, -1, cv2.LINE_AA)
        cv2.circle(image, red, 3, RED, -1, cv2.LINE_AA)


def draw_label(image: np.ndarray, lines: list[str]) -> None:
    """Put a short white label in the top-left corner."""
    y = 32
    for line in lines:
        origin = (16, y)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 30


def pixel_errors(projected: np.ndarray, feet: np.ndarray) -> np.ndarray:
    """Return the distance in pixels between each projected point and box bottom."""
    if len(projected) == 0:
        return np.zeros(0)
    return np.linalg.norm(projected - feet, axis=1)


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    """Print the per-camera error table and flag large mean errors."""
    print()
    print(f"{'camera':<8} {'calibration':<12} {'visible':>8} {'mean_px':>10} {'max_px':>10}  flag")
    flagged = False
    for row in rows:
        flag = ""
        mean_px = float(row["mean_px"])
        if np.isfinite(mean_px) and mean_px > LARGE_MEAN_ERROR_PX:
            flag = "POSSIBLE MAPPING PROBLEM"
            flagged = True
        print(
            f"{row['camera']:<8} {row['calibration']:<12} {int(row['visible']):8d} "
            f"{mean_px:10.2f} {float(row['max_px']):10.2f}  {flag}"
        )
    print(f"Flag rule: mean error above {LARGE_MEAN_ERROR_PX:.0f} px.")
    if not flagged:
        print("No camera was flagged.")


def main() -> None:
    """Save one overlay per camera and print projection errors."""
    dataset = resolve_dataset_root(DATA_ROOT)
    annotation_path, people = first_annotation(dataset)
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Frame: {annotation_path.name}")

    summary: list[dict[str, float | str | int]] = []
    for cam_id, calib_name in CAMERA_TO_CALIB.items():
        K, R, t = load_camera_calibration(dataset, cam_id)
        H = ground_homography(K, R, t)
        image_path = frame_image(dataset, cam_id, annotation_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        height, width = image.shape[:2]
        world, feet = visible_ground_points(people, view_num_for_camera(cam_id))
        projected = world_to_image(H, world) if len(world) else np.zeros((0, 2))
        errors = pixel_errors(projected, feet)
        mean_px = float(np.mean(errors)) if len(errors) else float("nan")
        max_px = float(np.max(errors)) if len(errors) else float("nan")

        for polyline in grid_polylines(X_MIN_M, X_MAX_M, Y_MIN_M, Y_MAX_M):
            uv, valid = project_masked(H, R, t, polyline, width, height)
            draw_polyline(image, uv, valid)
        draw_dots(image, projected, feet)
        draw_label(
            image,
            [
                f"{cam_id}  {calib_name}  frame {annotation_path.stem}",
                "red: projected ground truth   green: box bottom center",
            ],
        )
        destination = output_dir / f"check_homography_{cam_id}.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Could not write {destination}")
        print(
            f"{cam_id} {calib_name}: {len(errors)} visible, "
            f"mean {mean_px:.2f} px, max {max_px:.2f} px -> {destination.name}"
        )
        summary.append(
            {
                "camera": cam_id,
                "calibration": calib_name,
                "visible": len(errors),
                "mean_px": mean_px,
                "max_px": max_px,
            }
        )
    print_summary(summary)


if __name__ == "__main__":
    main()
