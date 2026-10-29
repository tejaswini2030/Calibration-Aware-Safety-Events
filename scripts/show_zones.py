#!/usr/bin/env python3
"""Draw the configured ground zones in a top-down view and on the first frame."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.events import load_event_config
from src.geometry import ground_homography, load_camera_calibration, world_to_image

# RGB colors shared by the bird's-eye plot. Image drawing converts to BGR.
ZONE_COLORS = {
    "near": (0.95, 0.45, 0.10),
    "medium": (0.15, 0.65, 0.30),
    "far": (0.20, 0.40, 0.85),
}


def _zone_color(zone_id: str, index: int) -> tuple[float, float, float]:
    if zone_id in ZONE_COLORS:
        return ZONE_COLORS[zone_id]
    cycle = list(ZONE_COLORS.values())
    return cycle[index % len(cycle)]


def _bgr(color: tuple[float, float, float]) -> tuple[int, int, int]:
    red, green, blue = color
    return int(round(blue * 255)), int(round(green * 255)), int(round(red * 255))


def _camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return the camera center in world meters, shape (3,)."""
    return (-R.T @ t).ravel()


def _corners_inside(uv: np.ndarray, depth: np.ndarray, width: int, height: int) -> bool:
    return bool(
        np.all(
            (depth > 0.0)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < height)
        )
    )


def first_frame_image(dataset: Path, cam_id: str) -> tuple[Path, np.ndarray]:
    """Load the image that matches the earliest annotation file."""
    annotations = sorted((dataset / "annotations_positions").glob("*.json"))
    if not annotations:
        raise FileNotFoundError(f"No annotations in {dataset / 'annotations_positions'}")
    stem = annotations[0].stem
    folder = dataset / "Image_subsets" / cam_id
    matches = [
        path
        for path in sorted(folder.glob(f"{stem}.*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not matches:
        raise FileNotFoundError(f"No image for frame {stem} in {folder}")
    image = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {matches[0]}")
    return matches[0], image


def save_birdseye(
    path: Path,
    positions: np.ndarray,
    zones: list[dict],
    camera_xy: np.ndarray,
    cam_id: str,
) -> None:
    """Save a top-down plot of zones and visible ground-truth positions."""
    fig, ax = plt.subplots(figsize=(7, 9))
    if len(positions):
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=2,
            c="0.55",
            alpha=0.25,
            linewidths=0,
            label="Visible ground truth",
            rasterized=True,
        )
    for index, zone in enumerate(zones):
        polygon = zone["polygon"]
        color = _zone_color(str(zone["id"]), index)
        closed = np.vstack([polygon, polygon[0]])
        ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.35, label=str(zone["name"]))
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        center = polygon.mean(axis=0)
        ax.text(center[0], center[1], str(zone["name"]), ha="center", va="center", fontsize=9)
    ax.scatter(
        [camera_xy[0]],
        [camera_xy[1]],
        c="black",
        marker="^",
        s=50,
        zorder=5,
        label=f"{cam_id} camera",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{cam_id} zones and visible ground-truth positions")
    ax.legend(loc="upper left", markerscale=3, frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_image_overlay(
    path: Path,
    image: np.ndarray,
    zones: list[dict],
    H: np.ndarray,
    cam_id: str,
) -> None:
    """Project each zone onto the frame and save the overlay."""
    overlay = image.copy()
    for index, zone in enumerate(zones):
        polygon = zone["polygon"]
        uv = world_to_image(H, polygon)
        pixels = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        color = _bgr(_zone_color(str(zone["id"]), index))
        cv2.fillPoly(overlay, [pixels], color)
    blended = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    for index, zone in enumerate(zones):
        polygon = zone["polygon"]
        uv = world_to_image(H, polygon)
        pixels = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        color = _bgr(_zone_color(str(zone["id"]), index))
        cv2.polylines(blended, [pixels], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        center = world_to_image(H, polygon.mean(axis=0).reshape(1, 2))[0]
        origin = (int(round(float(center[0]))), int(round(float(center[1]))))
        cv2.putText(
            blended,
            str(zone["name"]),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            blended,
            str(zone["name"]),
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        blended,
        f"{cam_id} zones",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        blended,
        f"{cam_id} zones",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), blended):
        raise OSError(f"Could not write {path}")


def main() -> None:
    """Load the event config, print zone distances, and save both figures."""
    config_path = project_root() / "configs" / "events.yaml"
    config = load_event_config(config_path)
    cam_id = str(config["camera"])
    dataset = resolve_dataset_root(DATA_ROOT)
    K, R, t = load_camera_calibration(dataset, cam_id)
    H = ground_homography(K, R, t)
    camera = _camera_center(R, t)
    frame_path, image = first_frame_image(dataset, cam_id)
    height, width = image.shape[:2]

    print(f"Camera {cam_id} center: ({camera[0]:.3f}, {camera[1]:.3f}, {camera[2]:.3f}) m")
    print(f"Frame: {frame_path.name} ({width} x {height})")
    for zone in config["zones"]:
        polygon = zone["polygon"]
        center = polygon.mean(axis=0)
        ground_m = float(np.hypot(center[0] - camera[0], center[1] - camera[1]))
        distance_m = float(np.linalg.norm(np.array([center[0], center[1], 0.0]) - camera))
        uv = world_to_image(H, polygon)
        depth = (np.column_stack([polygon, np.zeros(len(polygon))]) @ R.T + t.ravel())[:, 2]
        inside = _corners_inside(uv, depth, width, height)
        print(
            f"{zone['id']}: ground distance {ground_m:.2f} m "
            f"({distance_m:.2f} m from the camera center to the zone center), "
            f"all corners inside image: {inside}"
        )
        if not inside:
            print(f"  projected corners:\n{np.array2string(uv, precision=1)}")

    annotations = load_annotations(dataset, cam_id)
    visible = annotations.loc[annotations["visible"], ["X_m", "Y_m"]].to_numpy()
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    birdseye_path = output_dir / f"zones_birdseye_{cam_id}.png"
    image_path = output_dir / f"zones_image_{cam_id}.png"
    save_birdseye(birdseye_path, visible, config["zones"], camera[:2], cam_id)
    save_image_overlay(image_path, image, config["zones"], H, cam_id)
    print(f"Saved {birdseye_path.name}")
    print(f"Saved {image_path.name}")


if __name__ == "__main__":
    main()
