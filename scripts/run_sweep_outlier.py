#!/usr/bin/env python3
"""Outlier calibration-click sweep on C1 ground-truth tracks (n12 layout).

Every point gets a 1 px click error. One randomly chosen point also gets an
outlier of magnitude in {0, 10, 20, 30, 50} px. Three estimators are compared:

  dlt         normalized DLT least squares on all 12 points
  ransac      exhaustive RANSAC over all C(12,4) subsets (3 px transfer threshold)
  cv2_ransac  OpenCV RANSAC; 3 px converted to meters via the median local
              scale of H* at the calibration pixels (destination residuals)

Existing sweep CSV files are not modified.
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
)
from src.geometry import (
    image_to_world,
    invert_homography,
    load_camera_calibration,
    median_meters_per_pixel,
    normalized_dlt,
    ransac_homography_exhaustive,
    world_to_image,
)
from src.geometry import ground_homography
from src.perturb import choose_calibration_points, perturb_pixels

LAYOUT = "n12"
CLICK_EPS_PX = 1.0
OUTLIER_MAG_PX = (0, 10, 20, 30, 50)
N_TRIALS = 100
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
ESTIMATORS = ("dlt", "ransac", "cv2_ransac")
TRANSFER_THRESH_PX = 3.0
PARALLEL_IF_SLOWER_THAN_S = 0.2

ESTIMATOR_COLORS = {
    "dlt": "#e07a2f",
    "ransac": "#2e9b4f",
    "cv2_ransac": "#3d6fd8",
}
ESTIMATOR_LABELS = {
    "dlt": "DLT (all 12)",
    "ransac": "exhaustive RANSAC",
    "cv2_ransac": "OpenCV RANSAC",
}

_CAM_ID: str | None = None
_TRACKS: pd.DataFrame | None = None
_EVENTS: pd.DataFrame | None = None
_CFG: dict | None = None
_WORLD_PTS: np.ndarray | None = None
_EXACT_PIXELS: np.ndarray | None = None
_H_STAR: np.ndarray | None = None
_H_STAR_I2W: np.ndarray | None = None
_CV2_THRESH_M: float | None = None


def _init_worker(
    cam_id: str,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    H_star: np.ndarray,
    H_star_i2w: np.ndarray,
    cv2_thresh_m: float,
) -> None:
    global _CAM_ID, _TRACKS, _EVENTS, _CFG
    global _WORLD_PTS, _EXACT_PIXELS, _H_STAR, _H_STAR_I2W, _CV2_THRESH_M
    _CAM_ID = cam_id
    _TRACKS = tracks
    _EVENTS = events
    _CFG = cfg
    _WORLD_PTS = world_pts
    _EXACT_PIXELS = exact_pixels
    _H_STAR = H_star
    _H_STAR_I2W = H_star_i2w
    _CV2_THRESH_M = float(cv2_thresh_m)


def _detect_all(tracks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    zone_events = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell_events = detect_dwell_events(zone_events, cfg)
    proximity_events = detect_proximity_events(tracks, cfg)
    return pd.concat([zone_events, dwell_events, proximity_events], ignore_index=True)


def _score_with_H(
    H_estimated: np.ndarray,
    H_star: np.ndarray,
    tracks_xy: pd.DataFrame,
    gt_events: pd.DataFrame,
    cfg: dict,
) -> dict[str, float]:
    """Reproject GT feet through H_estimated and score events + position error."""
    tracks = tracks_xy.copy()
    true_xy = tracks[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
    foot_pixels = world_to_image(H_star, true_xy)
    reprojected = image_to_world(H_estimated, foot_pixels)
    position_error = np.linalg.norm(reprojected - true_xy, axis=1)
    rounded = np.round(reprojected, 6)
    tracks["X_m"] = rounded[:, 0]
    tracks["Y_m"] = rounded[:, 1]
    predicted = _detect_all(tracks, cfg)
    scores, _matches = score_events(gt_events, predicted, cfg)
    out: dict[str, float] = {"pos_mean_m": float(np.mean(position_error))}
    for event_type in EVENT_TYPES:
        row = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == "all")]
        out[f"f1_{event_type}"] = float(row["f1"].iloc[0]) if len(row) else float("nan")
    return out


def _apply_click_and_outlier(
    exact_pixels: np.ndarray,
    click_eps_px: float,
    outlier_mag_px: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Return clicked pixels and the index of the designated outlier point."""
    rng = np.random.default_rng(int(seed))
    clicked = perturb_pixels(exact_pixels, float(click_eps_px), rng)
    outlier_index = int(rng.integers(0, len(exact_pixels)))
    if float(outlier_mag_px) > 0.0:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        clicked[outlier_index] = clicked[outlier_index] + float(outlier_mag_px) * np.array(
            [math.cos(angle), math.sin(angle)], dtype=np.float64
        )
    return clicked, outlier_index


def estimate_homographies(
    pixels: np.ndarray,
    world_pts: np.ndarray,
    cv2_thresh_m: float,
) -> dict[str, dict[str, Any]]:
    """Run DLT, exhaustive RANSAC, and OpenCV RANSAC on the same clicks."""
    results: dict[str, dict[str, Any]] = {}

    H_dlt = normalized_dlt(pixels, world_pts)
    results["dlt"] = {
        "H": H_dlt,
        "inlier_mask": np.ones(len(pixels), dtype=bool),
        "n_inliers": len(pixels),
        "outlier_flagged": False,
    }

    ransac = ransac_homography_exhaustive(pixels, world_pts, threshold_px=TRANSFER_THRESH_PX)
    results["ransac"] = {
        "H": ransac["H"],
        "inlier_mask": ransac["inlier_mask"],
        "n_inliers": int(ransac["n_inliers"]),
    }

    found = cv2.findHomography(
        np.asarray(pixels, dtype=np.float64),
        np.asarray(world_pts, dtype=np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(cv2_thresh_m),
    )
    H_cv = found[0] if isinstance(found, tuple) else found
    mask_cv = found[1] if isinstance(found, tuple) else None
    if H_cv is None:
        # OpenCV failed; fall back to DLT so the trial still has a score.
        H_cv = H_dlt
        inlier_mask = np.ones(len(pixels), dtype=bool)
    else:
        H_cv = H_cv / H_cv[2, 2]
        if mask_cv is None:
            inlier_mask = np.ones(len(pixels), dtype=bool)
        else:
            inlier_mask = np.asarray(mask_cv).reshape(-1).astype(bool)
    results["cv2_ransac"] = {
        "H": H_cv,
        "inlier_mask": inlier_mask,
        "n_inliers": int(inlier_mask.sum()),
    }
    return results


def run_outlier_trial(
    outlier_mag_px: float,
    seed: int,
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    H_star: np.ndarray,
    cv2_thresh_m: float,
    click_eps_px: float = CLICK_EPS_PX,
) -> list[dict[str, Any]]:
    """One (magnitude, seed): build clicked pixels, estimate three Hs, score each."""
    clicked, outlier_index = _apply_click_and_outlier(
        exact_pixels, click_eps_px, outlier_mag_px, seed
    )
    estimates = estimate_homographies(clicked, world_pts, cv2_thresh_m)
    rows: list[dict[str, Any]] = []
    for name in ESTIMATORS:
        est = estimates[name]
        scores = _score_with_H(est["H"], H_star, tracks, events, cfg)
        flagged = not bool(est["inlier_mask"][outlier_index])
        # Only meaningful when there is a real outlier displacement.
        if float(outlier_mag_px) <= 0.0:
            flagged_value = float("nan")
        else:
            flagged_value = float(flagged)
        rows.append(
            {
                "estimator": name,
                "outlier_mag_px": float(outlier_mag_px),
                "seed": int(seed),
                "outlier_index": int(outlier_index),
                "n_inliers": int(est["n_inliers"]),
                "outlier_flagged": flagged_value,
                **scores,
            }
        )
    return rows


def _worker(job: tuple[float, int]) -> list[dict[str, Any]]:
    mag, seed = job
    assert _WORLD_PTS is not None and _EXACT_PIXELS is not None
    assert _TRACKS is not None and _EVENTS is not None and _CFG is not None
    assert _H_STAR is not None and _CV2_THRESH_M is not None
    return run_outlier_trial(
        mag,
        seed,
        _WORLD_PTS,
        _EXACT_PIXELS,
        _TRACKS,
        _EVENTS,
        _CFG,
        _H_STAR,
        _CV2_THRESH_M,
    )


def _jobs() -> list[tuple[float, int]]:
    return [(float(mag), seed) for mag in OUTLIER_MAG_PX for seed in SEEDS]


def _sanity_check(
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    H_star: np.ndarray,
    H_star_i2w: np.ndarray,
    cv2_thresh_m: float,
) -> bool:
    print("\nSanity check: 0 px click, 0 px outlier → F1 = 1.0 for all estimators")
    ok = True
    estimates = estimate_homographies(exact_pixels, world_pts, cv2_thresh_m)
    for name in ESTIMATORS:
        H = estimates[name]["H"]
        # Homography should match H*_img_to_world.
        diff = float(np.max(np.abs(H / H[2, 2] - H_star_i2w / H_star_i2w[2, 2])))
        scores = _score_with_H(H, H_star, tracks, events, cfg)
        for event_type in EVENT_TYPES:
            f1 = scores[f"f1_{event_type}"]
            if not np.isclose(f1, 1.0, atol=1e-9):
                print(f"  FAIL  {name} {event_type}: F1={f1:.6f} (H diff={diff:.2e})")
                ok = False
            else:
                print(f"  PASS  {name} {event_type}: F1=1.000")
        if estimates[name]["n_inliers"] != len(exact_pixels) and name != "dlt":
            # With perfect data every point should be an inlier.
            print(f"  FAIL  {name}: n_inliers={estimates[name]['n_inliers']} expected {len(exact_pixels)}")
            ok = False
    print("Sanity check:", "PASS" if ok else "FAIL")
    return ok


def _group(frame: pd.DataFrame, estimator: str, mag: float, column: str) -> np.ndarray:
    mask = (frame["estimator"] == estimator) & np.isclose(frame["outlier_mag_px"], mag)
    return frame.loc[mask, column].to_numpy(dtype=float)


def _print_summary(frame: pd.DataFrame) -> None:
    print(f"\n{N_TRIALS} trials per outlier magnitude, seeds {SEEDS[0]}..{SEEDS[-1]}")
    print(f"Click noise = {CLICK_EPS_PX:g} px on every point; n12 layout; threshold = {TRANSFER_THRESH_PX:g} px")
    for estimator in ESTIMATORS:
        print(f"\n{estimator}")
        print(
            f"  {'mag':>5}  {'F1z':>7} {'F1d':>7} {'F1p':>7}  "
            f"{'pos_m':>8}  {'detect%':>8}  {'nin':>5}"
        )
        for mag in OUTLIER_MAG_PX:
            f1 = {
                et: float(np.mean(_group(frame, estimator, float(mag), f"f1_{et}")))
                for et in EVENT_TYPES
            }
            pos = float(np.mean(_group(frame, estimator, float(mag), "pos_mean_m")))
            nin = float(np.mean(_group(frame, estimator, float(mag), "n_inliers")))
            if float(mag) <= 0.0:
                detect = float("nan")
            else:
                detect = 100.0 * float(np.nanmean(_group(frame, estimator, float(mag), "outlier_flagged")))
            detect_s = f"{detect:7.1f}%" if np.isfinite(detect) else "     n/a"
            print(
                f"  {mag:5g}  {f1['zone']:7.3f} {f1['dwell']:7.3f} {f1['proximity']:7.3f}  "
                f"{pos:8.4f}  {detect_s}  {nin:5.1f}"
            )


def _plot_f1(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for estimator in ESTIMATORS:
            ys = [
                float(np.mean(_group(frame, estimator, float(mag), f"f1_{event_type}")))
                for mag in OUTLIER_MAG_PX
            ]
            ax.plot(
                OUTLIER_MAG_PX,
                ys,
                marker="o",
                color=ESTIMATOR_COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
                linewidth=1.8,
            )
        ax.set_title(event_type)
        ax.set_xlabel("outlier magnitude (px)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(OUTLIER_MAG_PX))
        ax.set_xlim(-2, 52)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("C1 event F1 vs one outlier calibration click (n12)", y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def main() -> None:
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    tracks = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]].copy()
    events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")
    world_pts, exact_pixels = choose_calibration_points(cam_id, layout=LAYOUT)
    print(f"{LAYOUT}: {len(world_pts)} calibration points")

    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    H_star = ground_homography(K, R, t)
    H_star_i2w = invert_homography(H_star)
    scale = median_meters_per_pixel(H_star_i2w, exact_pixels)
    cv2_thresh_m = TRANSFER_THRESH_PX * scale
    print(
        f"OpenCV RANSAC threshold: {TRANSFER_THRESH_PX:g} px × "
        f"{scale:.4f} m/px (median local scale under H*) = {cv2_thresh_m:.4f} m"
    )

    _sanity_check(world_pts, exact_pixels, tracks, events, cfg, H_star, H_star_i2w, cv2_thresh_m)

    started = time.perf_counter()
    timed_rows = run_outlier_trial(
        30.0, 0, world_pts, exact_pixels, tracks, events, cfg, H_star, cv2_thresh_m
    )
    trial_seconds = time.perf_counter() - started
    timed_key = (30.0, 0)
    pending = [job for job in _jobs() if job != timed_key]
    results = list(timed_rows)

    initargs = (cam_id, tracks, events, cfg, world_pts, exact_pixels, H_star, H_star_i2w, cv2_thresh_m)
    if trial_seconds > PARALLEL_IF_SLOWER_THAN_S:
        workers = max(1, (os.cpu_count() or 2) - 1)
        print(f"One trial takes {trial_seconds:.2f} s. Running {workers} worker processes.")
        with Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
            for rows in tqdm(
                pool.imap_unordered(_worker, pending, chunksize=2),
                total=len(_jobs()),
                initial=1,
                desc="trials",
            ):
                results.extend(rows)
    else:
        print(f"One trial takes {trial_seconds:.2f} s. Running sequentially.")
        _init_worker(*initargs)
        for job in tqdm(pending, total=len(_jobs()), initial=1, desc="trials"):
            results.extend(_worker(job))

    frame = pd.DataFrame(results)
    frame = frame.sort_values(
        ["estimator", "outlier_mag_px", "seed"], kind="mergesort"
    ).reset_index(drop=True)

    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"sweep_{cam_id}_outlier.csv"
    frame.to_csv(table_path, index=False, float_format="%.6f")
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    _print_summary(frame)
    _plot_f1(frame, output_dir / f"f1_vs_outlier_{cam_id}.png")


if __name__ == "__main__":
    main()
