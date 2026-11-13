#!/usr/bin/env python3
"""Camera-bump sweep on C1 ground-truth tracks.

After calibration, the camera is rotated by a small angle about one axis of
its own frame (pitch / yaw / roll). True feet are projected with the bumped
pose; the system still lifts them with the original H*_img_to_world. Existing
sweep CSV files are not modified.
"""

from __future__ import annotations

import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.evaluation import score_events
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    load_event_config,
    signed_distance_to_polygon,
)
from src.geometry import (
    ground_homography,
    image_to_world,
    invert_homography,
    load_camera_calibration,
    world_to_image,
)

AXES = ("pitch", "yaw", "roll")
# Camera-frame axes (OpenCV): X right, Y down, Z forward.
AXIS_VECTORS = {
    "pitch": np.array([1.0, 0.0, 0.0], dtype=np.float64),  # tilt up/down
    "yaw": np.array([0.0, 1.0, 0.0], dtype=np.float64),  # pan left/right
    "roll": np.array([0.0, 0.0, 1.0], dtype=np.float64),  # rotate about view
}
ANGLES_DEG = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
N_TRIALS = 100
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
ZONE_IDS = ("near", "medium", "far")
PARALLEL_IF_SLOWER_THAN_S = 0.2

AXIS_COLORS = {
    "pitch": "#e07a2f",
    "yaw": "#2e9b4f",
    "roll": "#3d6fd8",
}
ZONE_COLORS = {
    "near": "#f2731a",
    "medium": "#26994d",
    "far": "#3366d9",
    "overall": "0.25",
}

_K: np.ndarray | None = None
_R: np.ndarray | None = None
_T: np.ndarray | None = None
_H_STAR_I2W: np.ndarray | None = None
_TRACKS: pd.DataFrame | None = None
_EVENTS: pd.DataFrame | None = None
_CFG: dict | None = None


def _init_worker(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H_star_i2w: np.ndarray,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
) -> None:
    global _K, _R, _T, _H_STAR_I2W, _TRACKS, _EVENTS, _CFG
    _K = K
    _R = R
    _T = t
    _H_STAR_I2W = H_star_i2w
    _TRACKS = tracks
    _EVENTS = events
    _CFG = cfg


def _detect_all(tracks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    zone_events = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell_events = detect_dwell_events(zone_events, cfg)
    proximity_events = detect_proximity_events(tracks, cfg)
    return pd.concat([zone_events, dwell_events, proximity_events], ignore_index=True)


def _error_stats(errors: np.ndarray) -> dict[str, float]:
    if len(errors) == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


def rotation_delta(axis: str, angle_deg: float, sign: float) -> np.ndarray:
    """Small rotation matrix about one camera-frame axis."""
    if axis not in AXIS_VECTORS:
        raise ValueError(f"Unknown axis {axis!r}. Expected one of {AXES}")
    angle_rad = float(sign) * math.radians(float(angle_deg))
    rvec = AXIS_VECTORS[axis] * angle_rad
    R_delta, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return np.asarray(R_delta, dtype=np.float64)


def bump_pose(
    R: np.ndarray,
    t: np.ndarray,
    axis: str,
    angle_deg: float,
    sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the camera about its optical centre: R' = RΔ R, t' = RΔ t."""
    R_delta = rotation_delta(axis, angle_deg, sign)
    R_bumped = R_delta @ np.asarray(R, dtype=np.float64)
    t_bumped = R_delta @ np.asarray(t, dtype=np.float64).reshape(3, 1)
    return R_bumped, t_bumped


def equivalent_pixel_shift(focal_length_px: float, angle_deg: float) -> float:
    """Approximate image-centre shift for a pure rotation: f * tan(θ)."""
    return float(focal_length_px) * math.tan(math.radians(float(angle_deg)))


def run_bump_trial(
    axis: str,
    angle_deg: float,
    seed: int,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H_star_i2w: np.ndarray,
    tracks_xy: pd.DataFrame,
    gt_events: pd.DataFrame,
    cfg: dict,
) -> dict[str, Any]:
    """One (axis, angle, seed): bump, project, lift with original H*, score."""
    rng = np.random.default_rng(int(seed))
    sign = 1.0 if float(angle_deg) == 0.0 else float(rng.choice([-1.0, 1.0]))
    R_bumped, t_bumped = bump_pose(R, t, axis, angle_deg, sign)
    H_bumped_w2i = ground_homography(K, R_bumped, t_bumped)

    tracks = tracks_xy.copy()
    true_xy = tracks[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
    foot_pixels = world_to_image(H_bumped_w2i, true_xy)
    reprojected = image_to_world(H_star_i2w, foot_pixels)
    position_error = np.linalg.norm(reprojected - true_xy, axis=1)
    rounded = np.round(reprojected, 6)
    tracks["X_m"] = rounded[:, 0]
    tracks["Y_m"] = rounded[:, 1]

    predicted = _detect_all(tracks, cfg)
    scores, matches = score_events(gt_events, predicted, cfg)

    result: dict[str, Any] = {
        "axis": axis,
        "angle_deg": float(angle_deg),
        "seed": int(seed),
        "sign": float(sign),
    }
    for row in scores.itertuples(index=False):
        if str(row.zone_id) == "all":
            result[f"f1_{row.event_type}"] = float(row.f1)
            result[f"precision_{row.event_type}"] = float(row.precision)
            result[f"recall_{row.event_type}"] = float(row.recall)
        elif str(row.event_type) in ("zone", "dwell"):
            result[f"f1_{row.event_type}_{row.zone_id}"] = float(row.f1)

    overall = _error_stats(position_error)
    result["pos_mean_m"] = overall["mean"]
    result["pos_p95_m"] = overall["p95"]
    for zone in cfg["zones"]:
        inside = signed_distance_to_polygon(true_xy, zone["polygon"]) >= 0.0
        stats = _error_stats(position_error[inside])
        zone_id = str(zone["id"])
        result[f"pos_mean_m_{zone_id}"] = stats["mean"]
        result[f"pos_p95_m_{zone_id}"] = stats["p95"]
    return result


def _jobs() -> list[tuple[str, float, int]]:
    return [(axis, float(angle), seed) for axis in AXES for angle in ANGLES_DEG for seed in SEEDS]


def _worker(job: tuple[str, float, int]) -> dict[str, Any]:
    axis, angle, seed = job
    assert _K is not None and _R is not None and _T is not None
    assert _H_STAR_I2W is not None and _TRACKS is not None and _EVENTS is not None and _CFG is not None
    return run_bump_trial(axis, angle, seed, _K, _R, _T, _H_STAR_I2W, _TRACKS, _EVENTS, _CFG)


def _group(frame: pd.DataFrame, axis: str, angle: float, column: str) -> np.ndarray:
    mask = (frame["axis"] == axis) & np.isclose(frame["angle_deg"], angle)
    return frame.loc[mask, column].to_numpy(dtype=float)


def _sanity_check(frame: pd.DataFrame) -> bool:
    print("\nSanity check: angle 0 → F1 = 1.0 for every axis")
    ok = True
    for axis in AXES:
        for event_type in EVENT_TYPES:
            values = _group(frame, axis, 0.0, f"f1_{event_type}")
            if len(values) == 0 or not np.allclose(values, 1.0, atol=1e-9):
                bad = float(np.min(values)) if len(values) else float("nan")
                print(f"  FAIL  {axis} {event_type}: min F1={bad:.6f}")
                ok = False
            else:
                print(f"  PASS  {axis} {event_type}: F1=1.000")
    print("Sanity check:", "PASS" if ok else "FAIL")
    return ok


def _print_equivalent_shifts(focal_px: float) -> None:
    print(f"\nEquivalent pixel shift at image centre ≈ f·tan(θ), f = {focal_px:.1f} px")
    print(f"  {'angle_deg':>10}  {'equiv_px':>10}")
    for angle in ANGLES_DEG:
        print(f"  {angle:10g}  {equivalent_pixel_shift(focal_px, angle):10.3f}")


def _print_summary(frame: pd.DataFrame) -> None:
    print(f"\n{N_TRIALS} trials per (axis, angle), seeds {SEEDS[0]}..{SEEDS[-1]}")
    for axis in AXES:
        print(f"\n{axis}")
        print(
            f"  {'ang':>6}  "
            f"{'F1z':>7} {'F1d':>7} {'F1p':>7}  "
            f"{'pos':>7}  "
            + " ".join(f"{'pos_'+z:>7}" for z in ZONE_IDS)
        )
        for angle in ANGLES_DEG:
            f1 = {
                et: float(np.nanmean(_group(frame, axis, float(angle), f"f1_{et}")))
                for et in EVENT_TYPES
            }
            pos = float(np.nanmean(_group(frame, axis, float(angle), "pos_mean_m")))
            posz = [
                float(np.nanmean(_group(frame, axis, float(angle), f"pos_mean_m_{z}")))
                for z in ZONE_IDS
            ]
            print(
                f"  {angle:6g}  "
                f"{f1['zone']:7.3f} {f1['dwell']:7.3f} {f1['proximity']:7.3f}  "
                f"{pos:7.4f}  "
                + " ".join(f"{v:7.4f}" for v in posz)
            )


def _plot_f1(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for axis in AXES:
            ys = [
                float(np.nanmean(_group(frame, axis, float(angle), f"f1_{event_type}")))
                for angle in ANGLES_DEG
            ]
            ax.plot(
                ANGLES_DEG,
                ys,
                marker="o",
                color=AXIS_COLORS[axis],
                label=axis,
                linewidth=1.8,
            )
        ax.set_title(event_type)
        ax.set_xlabel("bump angle (deg)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(ANGLES_DEG))
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("C1 event F1 vs camera bump", y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _plot_position(frame: pd.DataFrame, path: Path) -> None:
    scopes = [("overall", "pos_mean_m")] + [(z, f"pos_mean_m_{z}") for z in ZONE_IDS]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, axis in zip(axes, AXES):
        for label, column in scopes:
            ys = [
                float(np.nanmean(_group(frame, axis, float(angle), column)))
                for angle in ANGLES_DEG
            ]
            color = ZONE_COLORS.get(label, AXIS_COLORS[axis])
            ax.plot(
                ANGLES_DEG,
                ys,
                marker="o",
                color=color,
                label=label,
                linewidth=1.6,
            )
        ax.set_title(axis)
        ax.set_xlabel("bump angle (deg)")
        ax.set_ylabel("mean position error (m)")
        ax.set_xticks(list(ANGLES_DEG))
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("C1 position error vs camera bump")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _column_order(columns: list[str]) -> list[str]:
    ordered = ["axis", "angle_deg", "seed", "sign"]
    for prefix in ("f1_", "precision_", "recall_", "pos_"):
        ordered.extend(sorted(c for c in columns if c.startswith(prefix)))
    ordered.extend(c for c in columns if c not in ordered)
    return ordered


def main() -> None:
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    tracks = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]].copy()
    events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")

    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    H_star_w2i = ground_homography(K, R, t)
    H_star_i2w = invert_homography(H_star_w2i)
    focal_px = float(0.5 * (K[0, 0] + K[1, 1]))
    print(f"C1 focal length (mean fx, fy): {focal_px:.2f} px")
    _print_equivalent_shifts(focal_px)

    started = time.perf_counter()
    timed = run_bump_trial("pitch", 0.5, 0, K, R, t, H_star_i2w, tracks, events, cfg)
    trial_seconds = time.perf_counter() - started
    timed_key = ("pitch", 0.5, 0)
    pending = [job for job in _jobs() if job != timed_key]
    results = [timed]

    initargs = (K, R, t, H_star_i2w, tracks, events, cfg)
    if trial_seconds > PARALLEL_IF_SLOWER_THAN_S:
        workers = max(1, (os.cpu_count() or 2) - 1)
        print(f"One trial takes {trial_seconds:.2f} s. Running {workers} worker processes.")
        with Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, pending, chunksize=4),
                total=len(_jobs()),
                initial=1,
                desc="trials",
            ):
                results.append(result)
    else:
        print(f"One trial takes {trial_seconds:.2f} s. Running sequentially.")
        _init_worker(*initargs)
        for job in tqdm(pending, total=len(_jobs()), initial=1, desc="trials"):
            results.append(_worker(job))

    frame = pd.DataFrame(results)
    frame = frame.sort_values(["axis", "angle_deg", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]

    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"sweep_{cam_id}_bump.csv"
    frame.to_csv(table_path, index=False, float_format="%.6f")
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    _print_summary(frame)
    _sanity_check(frame)
    _plot_f1(frame, output_dir / f"f1_vs_bump_{cam_id}.png")
    _plot_position(frame, output_dir / f"position_error_vs_bump_{cam_id}.png")


if __name__ == "__main__":
    main()
