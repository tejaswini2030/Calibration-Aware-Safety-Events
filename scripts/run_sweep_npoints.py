#!/usr/bin/env python3
"""Sweep calibration noise vs number of calibration points on C1 GT tracks.

Layouts n4 / n6 / n8 / n12 all sit inside the same wide ground rectangle.
Each trial perturbs every calibration pixel by exactly eps px, re-estimates H
with normalized DLT (least squares for N > 4), and reuses run_trial for
projection, event detection, and scoring. Existing sweep CSV files are not
modified.
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.events import load_event_config
from src.perturb import _first_frame, choose_calibration_points, run_trial

LAYOUTS = ("n4", "n6", "n8", "n12")
EPS_PX = (0, 1, 2, 3, 5, 10)
N_TRIALS = 100
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
# GT-track trials are ~0.5 s; still worth parallelizing 2400 jobs.
PARALLEL_IF_SLOWER_THAN_S = 0.2

LAYOUT_COLORS = {
    "n4": "#e07a2f",
    "n6": "#2e9b4f",
    "n8": "#3d6fd8",
    "n12": "#c43c6a",
}
LAYOUT_MARKERS = {
    "n4": "o",
    "n6": "s",
    "n8": "^",
    "n12": "D",
}

_CAM_ID: str | None = None
_TRACKS: pd.DataFrame | None = None
_EVENTS: pd.DataFrame | None = None
_CFG: dict | None = None


def _init_worker(cam_id: str, tracks: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> None:
    global _CAM_ID, _TRACKS, _EVENTS, _CFG
    _CAM_ID = cam_id
    _TRACKS = tracks
    _EVENTS = events
    _CFG = cfg


def _worker(job: tuple[str, float, int]) -> dict[str, Any]:
    layout, eps_px, seed = job
    result = run_trial(_CAM_ID, eps_px, seed, _TRACKS, _EVENTS, _CFG, layout=layout)
    result["layout"] = layout
    return result


def _jobs() -> list[tuple[str, float, int]]:
    return [
        (layout, float(eps_px), seed)
        for layout in LAYOUTS
        for eps_px in EPS_PX
        for seed in SEEDS
    ]


def _run_sweep(
    cam_id: str,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    trial_seconds: float,
    timed_result: dict[str, Any],
) -> list[dict[str, Any]]:
    timed_key = (str(timed_result["layout"]), float(timed_result["eps_px"]), int(timed_result["seed"]))
    pending = [job for job in _jobs() if job != timed_key]
    results = [timed_result]
    initargs = (cam_id, tracks, events, cfg)
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
    return results


def _group_values(frame: pd.DataFrame, layout: str, eps_px: float, column: str) -> np.ndarray:
    mask = (frame["layout"] == layout) & np.isclose(frame["eps_px"], eps_px)
    return frame.loc[mask, column].to_numpy(dtype=float)


def _sanity_check_eps0(frame: pd.DataFrame) -> bool:
    print("\nSanity check: eps=0 F1 == 1.0 for every layout")
    ok = True
    for layout in LAYOUTS:
        for event_type in EVENT_TYPES:
            values = _group_values(frame, layout, 0.0, f"f1_{event_type}")
            if len(values) == 0:
                print(f"  FAIL  {layout} {event_type}: no rows")
                ok = False
                continue
            if not np.allclose(values, 1.0, atol=1e-9, equal_nan=False):
                bad = float(np.min(values))
                print(f"  FAIL  {layout} {event_type}: min F1={bad:.6f}")
                ok = False
            else:
                print(f"  PASS  {layout} {event_type}: F1=1.000")
    print("Sanity check:", "PASS" if ok else "FAIL")
    return ok


def _print_summary(frame: pd.DataFrame) -> None:
    print(f"\n{N_TRIALS} trials at each eps, seeds {SEEDS[0]}..{SEEDS[-1]}")
    for layout in LAYOUTS:
        print(f"\n{layout}")
        print(
            f"  {'eps':>5}  "
            f"{'F1z':>7} {'F1d':>7} {'F1p':>7}  "
            f"{'pos_mean':>9}"
        )
        for eps_px in EPS_PX:
            f1 = {et: float(np.mean(_group_values(frame, layout, float(eps_px), f"f1_{et}"))) for et in EVENT_TYPES}
            pos = float(np.mean(_group_values(frame, layout, float(eps_px), "pos_mean_m")))
            print(
                f"  {eps_px:5g}  "
                f"{f1['zone']:7.3f} {f1['dwell']:7.3f} {f1['proximity']:7.3f}  "
                f"{pos:9.4f}"
            )


def _curve(frame: pd.DataFrame, layout: str, column: str) -> np.ndarray:
    return np.asarray(
        [float(np.mean(_group_values(frame, layout, float(eps), column))) for eps in EPS_PX],
        dtype=float,
    )


def _plot_f1(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for layout in LAYOUTS:
            ys = _curve(frame, layout, f"f1_{event_type}")
            ax.plot(
                EPS_PX,
                ys,
                marker=LAYOUT_MARKERS[layout],
                color=LAYOUT_COLORS[layout],
                label=layout,
                linewidth=1.8,
            )
        ax.set_title(event_type)
        ax.set_xlabel("calibration noise (px)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(EPS_PX))
        ax.set_xlim(-0.4, 10.4)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("C1 event F1 vs calibration noise (number of points)", y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _plot_position_error(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for layout in LAYOUTS:
        ys = _curve(frame, layout, "pos_mean_m")
        ax.plot(
            EPS_PX,
            ys,
            marker=LAYOUT_MARKERS[layout],
            color=LAYOUT_COLORS[layout],
            label=layout,
            linewidth=1.8,
        )
    ax.set_xlabel("calibration noise (px)")
    ax.set_ylabel("mean position error (m)")
    ax.set_xticks(list(EPS_PX))
    ax.set_xlim(-0.4, 10.4)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("C1 position error vs calibration noise (number of points)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def _save_layout_overlay(cam_id: str, path: Path) -> None:
    """Four panels: each n-point layout drawn on the first frame."""
    import cv2

    _image_path, image = _first_frame(cam_id)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, corner_pixels = choose_calibration_points(cam_id, layout="n4")
    closed = np.vstack([corner_pixels, corner_pixels[:1]])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, layout in zip(axes.ravel(), LAYOUTS):
        world, pixels = choose_calibration_points(cam_id, layout=layout)
        ax.imshow(rgb)
        ax.plot(closed[:, 0], closed[:, 1], color="0.85", linewidth=1.2, linestyle="--")
        ax.scatter(
            pixels[:, 0],
            pixels[:, 1],
            s=55,
            c=LAYOUT_COLORS[layout],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        for index, uv in enumerate(pixels):
            ax.text(uv[0] + 12, uv[1] - 8, f"{index}", color="white", fontsize=7)
        ax.set_title(f"{layout}  (N={len(world)})")
        ax.set_xlim(0, rgb.shape[1])
        ax.set_ylim(rgb.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{cam_id} calibration layouts inside the wide rectangle", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _column_order(columns: list[str]) -> list[str]:
    ordered = ["layout", "eps_px", "seed"]
    for prefix in ("f1_", "precision_", "recall_", "start_delay_mean_", "pos_"):
        ordered.extend(sorted(column for column in columns if column.startswith(prefix)))
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def main() -> None:
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    tracks = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]].copy()
    events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Warm layout caches and save the overlay before timing trials.
    for layout in LAYOUTS:
        world, pixels = choose_calibration_points(cam_id, layout=layout)
        print(f"{layout}: {len(world)} points")
        print(
            "  world X "
            f"[{world[:, 0].min():.1f}, {world[:, 0].max():.1f}]  Y "
            f"[{world[:, 1].min():.1f}, {world[:, 1].max():.1f}]"
        )
    _save_layout_overlay(cam_id, output_dir / f"calib_points_npoints_{cam_id}.png")

    started = time.perf_counter()
    timed = run_trial(cam_id, 5.0, 0, tracks, events, cfg, layout="n8")
    timed["layout"] = "n8"
    trial_seconds = time.perf_counter() - started

    results = _run_sweep(cam_id, tracks, events, cfg, trial_seconds, timed)
    frame = pd.DataFrame(results)
    frame = frame.sort_values(["layout", "eps_px", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]

    table_path = output_dir / f"sweep_{cam_id}_npoints.csv"
    frame.to_csv(table_path, index=False, float_format="%.6f")
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    _print_summary(frame)
    _sanity_check_eps0(frame)
    _plot_f1(frame, output_dir / f"f1_vs_eps_npoints_{cam_id}.png")
    _plot_position_error(frame, output_dir / f"position_error_vs_eps_npoints_{cam_id}.png")


if __name__ == "__main__":
    main()
