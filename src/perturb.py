"""Calibration noise experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from src.data import (
    CELL_SIZE_M,
    DATA_ROOT,
    GRID_HEIGHT,
    GRID_WIDTH,
    ORIGIN_X_M,
    ORIGIN_Y_M,
    load_annotations,
    normalize_cam_id,
    project_root,
    resolve_dataset_root,
)
from src.evaluation import score_events
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    signed_distance_to_polygon,
)
from src.geometry import (
    ground_homography,
    image_to_world,
    load_camera_calibration,
    normalized_dlt,
    world_to_image,
)

# Wide layout: search axis-aligned rectangles on this ground step, keep the
# one that covers the most visible positions while every corner projects
# inside the image, then pull the edges in by WIDE_INSET_M.
WIDE_GRID_STEP_M = 0.5
WIDE_PIXEL_MARGIN = 40.0
WIDE_INSET_M = 0.5
_MIN_DEPTH_M = 0.05

WorldLayout = Callable[[str, np.ndarray, np.ndarray, np.ndarray, int, int], np.ndarray]


def _ground_edges(origin: float, cell_count: int, step_m: float) -> np.ndarray:
    """Grid lines from the ground origin through the far edge, spaced by step_m."""
    step_cells = int(round(step_m / CELL_SIZE_M))
    if step_cells < 1:
        raise ValueError(f"Grid step {step_m} m is smaller than one ground cell.")
    indices = np.arange(0, cell_count + 1, step_cells, dtype=np.int64)
    return origin + CELL_SIZE_M * indices


def _camera_depth(R: np.ndarray, t: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Camera-frame Z for ground points. Positive is in front of the camera."""
    world = np.column_stack([xy, np.zeros(len(xy))])
    return (world @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64).ravel())[:, 2]


def _corners(xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
    """Four corners of an axis-aligned ground rectangle, shape (4, 2)."""
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        dtype=np.float64,
    )


def _corners_inside(
    H: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: float,
) -> bool:
    """True when all four corners are in front of the camera and inside the image."""
    xy = _corners(*bounds)
    uv = world_to_image(H, xy)
    depth = _camera_depth(R, t, xy)
    return bool(
        np.all(depth > _MIN_DEPTH_M)
        and np.all(uv[:, 0] >= margin)
        and np.all(uv[:, 0] <= width - 1 - margin)
        and np.all(uv[:, 1] >= margin)
        and np.all(uv[:, 1] <= height - 1 - margin)
    )


def _count_prefix(points: np.ndarray) -> np.ndarray:
    """Inclusive prefix counts on the annotation grid.

    prefix[i + 1, j + 1] is the number of points with grid_x <= i and grid_y <= j.
    """
    grid_x = np.rint((points[:, 0] - ORIGIN_X_M) / CELL_SIZE_M).astype(np.int64)
    grid_y = np.rint((points[:, 1] - ORIGIN_Y_M) / CELL_SIZE_M).astype(np.int64)
    if (
        np.any(grid_x < 0)
        or np.any(grid_y < 0)
        or np.any(grid_x >= GRID_WIDTH)
        or np.any(grid_y >= GRID_HEIGHT)
    ):
        raise ValueError("A visible ground point lies outside the annotation grid.")
    histogram = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.int64)
    np.add.at(histogram, (grid_x, grid_y), 1)
    prefix = np.zeros((GRID_WIDTH + 1, GRID_HEIGHT + 1), dtype=np.int64)
    prefix[1:, 1:] = histogram
    return prefix.cumsum(axis=0).cumsum(axis=1)


def _count_rectangle(
    prefix: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> int:
    """Count visible positions inside an axis-aligned rectangle, edges included."""
    i0 = int(np.rint((xmin - ORIGIN_X_M) / CELL_SIZE_M))
    i1 = int(np.rint((xmax - ORIGIN_X_M) / CELL_SIZE_M))
    j0 = int(np.rint((ymin - ORIGIN_Y_M) / CELL_SIZE_M))
    j1 = int(np.rint((ymax - ORIGIN_Y_M) / CELL_SIZE_M))
    i0 = min(max(i0, 0), GRID_WIDTH)
    i1 = min(max(i1, 0), GRID_WIDTH - 1)
    j0 = min(max(j0, 0), GRID_HEIGHT)
    j1 = min(max(j1, 0), GRID_HEIGHT - 1)
    if i1 < i0 or j1 < j0:
        return 0
    return int(
        prefix[i1 + 1, j1 + 1] - prefix[i0, j1 + 1] - prefix[i1 + 1, j0] + prefix[i0, j0]
    )


def _first_frame(cam_id: str) -> tuple[Path, np.ndarray]:
    """Load the image that matches the earliest annotation file."""
    dataset = resolve_dataset_root(DATA_ROOT)
    annotations = sorted((dataset / "annotations_positions").glob("*.json"))
    if not annotations:
        raise FileNotFoundError(f"No annotations in {dataset / 'annotations_positions'}")
    folder = dataset / "Image_subsets" / cam_id
    matches = [
        path
        for path in sorted(folder.glob(f"{annotations[0].stem}.*"))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not matches:
        raise FileNotFoundError(f"No image for frame {annotations[0].stem} in {folder}")
    image = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {matches[0]}")
    return matches[0], image


def _wide_world_corners(
    cam_id: str,
    H: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Four corners of a large ground rectangle over the walking area.

    The rectangle is axis-aligned on the ground. Among candidates on a 0.5 m
    grid, it is the one that contains the most positions visible in this
    camera, with every corner in front of the camera and at least
    WIDE_PIXEL_MARGIN pixels inside the image. The edges are then moved
    inward by WIDE_INSET_M so the points are not on the extreme of the
    walking area. Corner order is (xmin, ymin), (xmax, ymin), (xmax, ymax),
    (xmin, ymax).
    """
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    visible = annotations.loc[annotations["visible"], ["X_m", "Y_m"]].to_numpy(dtype=np.float64)
    if len(visible) == 0:
        raise ValueError(f"No positions are visible in {cam_id}.")
    prefix = _count_prefix(visible)
    xs = _ground_edges(ORIGIN_X_M, GRID_WIDTH, WIDE_GRID_STEP_M)
    ys = _ground_edges(ORIGIN_Y_M, GRID_HEIGHT, WIDE_GRID_STEP_M)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_uv = world_to_image(H, grid)
    grid_depth = _camera_depth(R, t, grid)
    valid = (
        (grid_depth > _MIN_DEPTH_M)
        & (grid_uv[:, 0] >= WIDE_PIXEL_MARGIN)
        & (grid_uv[:, 0] <= width - 1 - WIDE_PIXEL_MARGIN)
        & (grid_uv[:, 1] >= WIDE_PIXEL_MARGIN)
        & (grid_uv[:, 1] <= height - 1 - WIDE_PIXEL_MARGIN)
    ).reshape(len(xs), len(ys))

    best: tuple[int, float, float, float, float, float] | None = None
    for i0 in range(len(xs) - 1):
        for i1 in range(i0 + 1, len(xs)):
            for j0 in range(len(ys) - 1):
                for j1 in range(j0 + 1, len(ys)):
                    if not (
                        valid[i0, j0] and valid[i1, j0] and valid[i1, j1] and valid[i0, j1]
                    ):
                        continue
                    bounds = (float(xs[i0]), float(xs[i1]), float(ys[j0]), float(ys[j1]))
                    count = _count_rectangle(prefix, *bounds)
                    area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
                    if best is None or count > best[0] or (count == best[0] and area > best[1]):
                        best = (count, area, *bounds)
    if best is None:
        raise RuntimeError(f"No ground rectangle for {cam_id} has all four corners inside the image.")

    chosen = best[2:]
    inset = WIDE_INSET_M
    while inset >= 0.0:
        trial = (
            chosen[0] + inset,
            chosen[1] - inset,
            chosen[2] + inset,
            chosen[3] - inset,
        )
        if (
            trial[1] > trial[0]
            and trial[3] > trial[2]
            and _corners_inside(H, R, t, trial, width, height, margin=1.0)
        ):
            chosen = trial
            break
        inset = round(inset - 0.1, 10)
    else:
        raise RuntimeError(f"Inset ground rectangle for {cam_id} does not stay inside the image.")
    return _corners(*chosen)


_LAYOUTS: dict[str, WorldLayout] = {
    "wide": _wide_world_corners,
}
_CALIBRATION_POINT_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_H_STAR_CACHE: dict[str, np.ndarray] = {}


def choose_calibration_points(
    cam_id: str | int,
    layout: str = "wide",
) -> tuple[np.ndarray, np.ndarray]:
    """Return 4 ground points and their exact pixels under the ground-truth H*.

    layout selects the point placement. "wide" covers the walking area.
    Further layouts (for example "narrow" or "near-only") can be added to
    _LAYOUTS. World points are (4, 2) meters. Pixel points are (4, 2) under
    H_world_to_img, in the same order.
    """
    name = normalize_cam_id(cam_id)
    if layout not in _LAYOUTS:
        known = ", ".join(sorted(_LAYOUTS))
        raise ValueError(f"Unknown calibration layout {layout!r}. Expected one of: {known}")
    cached = _CALIBRATION_POINT_CACHE.get((name, layout))
    if cached is not None:
        world, pixels = cached
        return world.copy(), pixels.copy()
    K, R, t = load_camera_calibration(DATA_ROOT, name)
    H_world_to_img = ground_homography(K, R, t)
    _image_path, image = _first_frame(name)
    height, width = image.shape[:2]
    world = _LAYOUTS[layout](name, H_world_to_img, R, t, width, height)
    pixels = world_to_image(H_world_to_img, world)
    _CALIBRATION_POINT_CACHE[(name, layout)] = (world, pixels)
    return world.copy(), pixels.copy()


def perturb_pixels(pixels: np.ndarray, eps_px: float, rng: np.random.Generator) -> np.ndarray:
    """Move each point by exactly eps_px pixels in a uniformly random direction.

    The direction is uniform on the circle, so the displacement length is
    eps_px rather than a sample from a disk. rng is a numpy Generator; two
    calls with generators created from the same seed match exactly.
    """
    pts = np.asarray(pixels, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pts.shape}")
    epsilon = float(eps_px)
    if epsilon < 0.0:
        raise ValueError(f"eps_px must be non-negative, got {eps_px}")
    angles = rng.uniform(0.0, 2.0 * np.pi, size=len(pts))
    offset = epsilon * np.column_stack((np.cos(angles), np.sin(angles)))
    return pts + offset


def estimate_H_img_to_world(pixels: np.ndarray, world_pts: np.ndarray) -> np.ndarray:
    """Estimate H_img_to_world from pixel to ground correspondences.

    pixels are the source and world_pts are the destination in meters.
    The fit is the normalized DLT, scaled so H[2, 2] = 1.
    """
    return normalized_dlt(pixels, world_pts)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the convex hull of (N, 2) points in counter-clockwise order."""
    ordered = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(ordered) < 3:
        raise ValueError("A convex hull needs at least 3 distinct points.")

    def turn(origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def _points_inside_convex(points: np.ndarray, hull: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """True for points inside a counter-clockwise convex polygon, boundary included."""
    inside = np.ones(len(points), dtype=bool)
    for index in range(len(hull)):
        start = hull[index]
        end = hull[(index + 1) % len(hull)]
        edge = end - start
        relative = points - start
        cross = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        inside &= cross >= -tol
    return inside


def mark_zones_inside_calibration_hull(zones: list[dict[str, Any]], calib_world: np.ndarray) -> None:
    """Record whether each zone lies inside the calibration-point hull.

    Sets inside_calibration_hull on every zone. The zone is inside only when
    all of its corners are inside the hull. Corners on the boundary count as
    inside, so a zone that shares an edge with the hull is inside.
    """
    hull = _convex_hull(np.asarray(calib_world, dtype=np.float64))
    for zone in zones:
        corners = np.asarray(zone["polygon"], dtype=np.float64)
        zone["inside_calibration_hull"] = bool(np.all(_points_inside_convex(corners, hull)))


def _h_star(cam_id: str) -> np.ndarray:
    """Return the ground-truth world-to-image homography for one camera."""
    cached = _H_STAR_CACHE.get(cam_id)
    if cached is not None:
        return cached
    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    homography = ground_homography(K, R, t)
    _H_STAR_CACHE[cam_id] = homography
    return homography


def _visible_tracks(gt_tracks: pd.DataFrame) -> pd.DataFrame:
    """Keep positions this camera can see, without modifying the caller's table."""
    tracks = gt_tracks
    if "visible" in tracks.columns:
        tracks = tracks.loc[tracks["visible"]]
    return tracks.loc[:, ["time_sec", "person_id", "X_m", "Y_m"]].copy()


def _detect_all(tracks: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Run zone, dwell, and proximity detection and stack the events."""
    zone_events = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell_events = detect_dwell_events(zone_events, cfg)
    proximity_events = detect_proximity_events(tracks, cfg)
    return pd.concat([zone_events, dwell_events, proximity_events], ignore_index=True)


def _error_stats(errors: np.ndarray) -> dict[str, float]:
    """Mean, median, 95th percentile, and max of distances in meters."""
    if len(errors) == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def _score_suffix(event_type: str, zone_id: str) -> str:
    """Flat key suffix: the event type, plus the zone when it is not the total."""
    if zone_id == "all":
        return event_type
    return f"{event_type}_{zone_id}"


def run_trial(
    cam_id: str | int,
    eps_px: float,
    seed: int,
    gt_tracks: pd.DataFrame,
    gt_events: pd.DataFrame,
    cfg: dict[str, Any],
    layout: str = "wide",
) -> dict[str, Any]:
    """Reproject ground-truth feet through a perturbed homography and score events.

    Each visible foot is projected with H* and lifted with the homography
    estimated from the perturbed calibration pixels. Detection therefore sees
    calibration error only. The returned dict holds eps_px, seed, precision,
    recall, and F1 for each event type and for each zone, position-error
    summaries overall and per zone, and the mean start delay of each type.
    """
    name = normalize_cam_id(cam_id)
    world_pts, exact_pixels = choose_calibration_points(name, layout=layout)
    rng = np.random.default_rng(int(seed))
    perturbed_pixels = perturb_pixels(exact_pixels, eps_px, rng)
    H_estimated = estimate_H_img_to_world(perturbed_pixels, world_pts)
    H_star = _h_star(name)

    tracks = _visible_tracks(gt_tracks)
    true_xy = tracks[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
    foot_pixels = world_to_image(H_star, true_xy)
    reprojected_xy = image_to_world(H_estimated, foot_pixels)
    # Position error is the unrounded geometric distance. Detection uses the
    # same 6-decimal meter resolution as position_id_to_world. A round trip
    # of a few ulps is below that resolution and must not flip a point that
    # sits on the hysteresis threshold.
    position_error = np.linalg.norm(reprojected_xy - true_xy, axis=1)
    rounded_xy = np.round(reprojected_xy, 6)
    tracks["X_m"] = rounded_xy[:, 0]
    tracks["Y_m"] = rounded_xy[:, 1]

    predicted = _detect_all(tracks, cfg)
    scores, matches = score_events(gt_events, predicted, cfg)

    result: dict[str, Any] = {"eps_px": float(eps_px), "seed": int(seed)}
    for row in scores.itertuples(index=False):
        suffix = _score_suffix(str(row.event_type), str(row.zone_id))
        result[f"precision_{suffix}"] = float(row.precision)
        result[f"recall_{suffix}"] = float(row.recall)
        result[f"f1_{suffix}"] = float(row.f1)
    for event_type in ("zone", "dwell", "proximity"):
        if matches.empty or "event_type" not in matches.columns:
            delay = math.nan
        else:
            delays = matches.loc[matches["event_type"] == event_type, "start_delay"]
            delay = float(delays.mean()) if len(delays) else math.nan
        result[f"start_delay_mean_{event_type}"] = delay

    overall = _error_stats(position_error)
    result["pos_mean_m"] = overall["mean"]
    result["pos_median_m"] = overall["median"]
    result["pos_p95_m"] = overall["p95"]
    result["pos_max_m"] = overall["max"]
    for zone in cfg["zones"]:
        inside = signed_distance_to_polygon(true_xy, zone["polygon"]) >= 0.0
        stats = _error_stats(position_error[inside])
        zone_id = str(zone["id"])
        result[f"pos_mean_m_{zone_id}"] = stats["mean"]
        result[f"pos_median_m_{zone_id}"] = stats["median"]
        result[f"pos_p95_m_{zone_id}"] = stats["p95"]
        result[f"pos_max_m_{zone_id}"] = stats["max"]
    return result


def _print_calibration_points(cam_id: str, layout: str, world: np.ndarray, pixels: np.ndarray) -> None:
    """Print the four world points and their exact pixels."""
    print(f"{layout} calibration points for {cam_id}")
    for index, (xy, uv) in enumerate(zip(world, pixels)):
        print(
            f"  {index}: world ({xy[0]:.3f}, {xy[1]:.3f}) m"
            f"    pixel ({uv[0]:.2f}, {uv[1]:.2f})"
        )


def save_calibration_overlay(
    cam_id: str | int,
    world: np.ndarray,
    pixels: np.ndarray,
    path: str | Path,
    layout: str = "wide",
) -> None:
    """Draw the calibration rectangle and its four corners on the first frame."""
    name = normalize_cam_id(cam_id)
    _image_path, image = _first_frame(name)
    canvas = image.copy()
    polygon = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [polygon], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    for index, (xy, uv) in enumerate(zip(world, pixels)):
        center = (int(round(float(uv[0]))), int(round(float(uv[1]))))
        cv2.circle(canvas, center, 10, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, 7, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        label = f"{index}  ({xy[0]:.1f}, {xy[1]:.1f}) m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _baseline = cv2.getTextSize(label, font, 0.6, 1)
        text_x = center[0] + 12
        text_y = center[1] - 12
        if text_x + text_size[0] > canvas.shape[1] - 8:
            text_x = center[0] - 12 - text_size[0]
        if text_y - text_size[1] < 8:
            text_y = center[1] + 12 + text_size[1]
        origin = (text_x, text_y)
        cv2.putText(canvas, label, origin, font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, origin, font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{name} {layout} calibration points",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{name} {layout} calibration points",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Could not write {output}")


def main() -> None:
    """Print the wide C1 calibration points and save them on the first frame."""
    cam_id = "C1"
    layout = "wide"
    world, pixels = choose_calibration_points(cam_id, layout=layout)
    _print_calibration_points(cam_id, layout, world, pixels)
    path = project_root() / "outputs" / f"calib_points_{cam_id}.png"
    save_calibration_overlay(cam_id, world, pixels, path, layout=layout)
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
