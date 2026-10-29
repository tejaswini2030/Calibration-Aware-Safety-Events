#!/usr/bin/env python3
"""Sweep calibration-pixel noise and summarize event scores for camera C1."""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.events import load_event_config
from src.perturb import choose_calibration_points, run_trial

EPS_PX = (0, 1, 3, 5, 10)
N_TRIALS = 200
# Same seeds at every noise level so the trials are paired.
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
# A trial slower than this is worth spreading across processes.
PARALLEL_IF_SLOWER_THAN_S = 1.0

EVENT_COLORS = {
    "zone": "#e07a2f",
    "dwell": "#2e9b4f",
    "proximity": "#3d6fd8",
}
ZONE_COLORS = {
    "near": "#f2731a",
    "medium": "#26994d",
    "far": "#3366d9",
}

_CAM_ID: str | None = None
_TRACKS: pd.DataFrame | None = None
_EVENTS: pd.DataFrame | None = None
_CFG: dict | None = None


def _init_worker(cam_id: str, tracks: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> None:
    """Keep the shared inputs in each worker so every job is not pickled."""
    global _CAM_ID, _TRACKS, _EVENTS, _CFG
    _CAM_ID = cam_id
    _TRACKS = tracks
    _EVENTS = events
    _CFG = cfg


def _worker(job: tuple[float, int]) -> dict:
    """Run one (eps, seed) trial inside a worker process."""
    eps_px, seed = job
    return run_trial(_CAM_ID, eps_px, seed, _TRACKS, _EVENTS, _CFG)


def _jobs() -> list[tuple[float, int]]:
    return [(float(eps_px), seed) for eps_px in EPS_PX for seed in SEEDS]


def _run_sweep(
    cam_id: str,
    tracks: pd.DataFrame,
    events: pd.DataFrame,
    cfg: dict,
    trial_seconds: float,
    timed_result: dict,
) -> list[dict]:
    """Run every eps/seed pair, reusing the trial that was timed."""
    pending = [job for job in _jobs() if not (job[0] == 5.0 and job[1] == 0)]
    results = [timed_result]
    if trial_seconds > PARALLEL_IF_SLOWER_THAN_S:
        workers = max(1, (os.cpu_count() or 2) - 1)
        print(f"One trial takes {trial_seconds:.2f} s. Running {workers} worker processes.")
        with Pool(workers, initializer=_init_worker, initargs=(cam_id, tracks, events, cfg)) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, pending, chunksize=4),
                total=len(_jobs()),
                initial=1,
                desc="trials",
            ):
                results.append(result)
    else:
        print(f"One trial takes {trial_seconds:.2f} s. Running sequentially.")
        for eps_px, seed in tqdm(pending, total=len(_jobs()), initial=1, desc="trials"):
            results.append(run_trial(cam_id, eps_px, seed, tracks, events, cfg))
    return results


def _column_order(columns: list[str]) -> list[str]:
    """Put identity columns first, then scores, delays, and position error."""
    ordered = ["eps_px", "seed"]
    for prefix in ("f1_", "precision_", "recall_", "start_delay_mean_", "pos_"):
        ordered.extend(sorted(column for column in columns if column.startswith(prefix)))
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def _group_values(frame: pd.DataFrame, eps_px: float, column: str) -> np.ndarray:
    return frame.loc[np.isclose(frame["eps_px"], eps_px), column].to_numpy(dtype=float)


def _print_summary(frame: pd.DataFrame, zones: list[dict]) -> None:
    """Print F1 percentiles and position error at each noise level."""
    print(f"\n{N_TRIALS} trials at each eps, seeds {SEEDS[0]}..{SEEDS[-1]}")
    print("Position error is the mean, across trials, of each trial's mean and 95th percentile.")
    for eps_px in EPS_PX:
        print(f"\neps = {eps_px:g} px")
        print(f"  {'event':<12} {'F1 mean':>8} {'F1 p5':>8} {'F1 p95':>8}")
        for event_type in EVENT_TYPES:
            values = _group_values(frame, eps_px, f"f1_{event_type}")
            print(
                f"  {event_type:<12} {np.mean(values):8.3f} "
                f"{np.percentile(values, 5):8.3f} {np.percentile(values, 95):8.3f}"
            )
        print(f"  {'position':<12} {'mean (m)':>8} {'p95 (m)':>8}")
        scopes = [("overall", "")] + [(str(zone["id"]), f"_{zone['id']}") for zone in zones]
        for label, suffix in scopes:
            mean_values = _group_values(frame, eps_px, f"pos_mean_m{suffix}")
            p95_values = _group_values(frame, eps_px, f"pos_p95_m{suffix}")
            print(f"  {label:<12} {np.mean(mean_values):8.4f} {np.mean(p95_values):8.4f}")


def _curve(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and 5th/95th percentile of one column at each eps."""
    means, low, high = [], [], []
    for eps_px in EPS_PX:
        values = _group_values(frame, float(eps_px), column)
        means.append(float(np.mean(values)))
        low.append(float(np.percentile(values, 5)))
        high.append(float(np.percentile(values, 95)))
    return np.asarray(means), np.asarray(low), np.asarray(high)


def _zone_label(zone: dict) -> str:
    """Zone name plus whether it sits inside the calibration-point hull."""
    inside = zone.get("inside_calibration_hull")
    if inside is None:
        return str(zone["name"])
    place = "inside hull" if inside else "outside hull"
    return f"{zone['name']} ({place})"


def _band_ylim(bands: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Y limits that keep the 5th–95th band visible without a large empty region."""
    low = min(float(np.min(band[1])) for band in bands)
    high = max(float(np.max(band[2])) for band in bands)
    span = max(high - low, 0.05)
    return max(0.0, low - 0.12 * span), min(1.05, high + 0.08 * span)


def _style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("calibration noise (px)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(EPS_PX))
    ax.set_xlim(-0.4, 10.4)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def _plot_f1_by_type(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bands = [_curve(frame, f"f1_{event_type}") for event_type in EVENT_TYPES]
    for event_type, (mean, low, high) in zip(EVENT_TYPES, bands):
        color = EVENT_COLORS[event_type]
        ax.fill_between(EPS_PX, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(EPS_PX, mean, color=color, marker="o", label=event_type)
    ax.set_ylim(*_band_ylim(bands))
    ax.set_title("C1 event F1 vs calibration noise")
    _style_axis(ax, "F1")
    _save(fig, path)


def _plot_f1_by_zone(frame: pd.DataFrame, zones: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    collected: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for ax, event_type in zip(axes, ("zone", "dwell")):
        for zone in zones:
            column = f"f1_{event_type}_{zone['id']}"
            mean, low, high = _curve(frame, column)
            collected.append((mean, low, high))
            color = ZONE_COLORS.get(str(zone["id"]), "0.4")
            ax.fill_between(EPS_PX, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(EPS_PX, mean, color=color, marker="o", label=_zone_label(zone))
        ax.set_title(event_type)
    bottom, top = _band_ylim(collected)
    for ax in axes:
        ax.set_ylim(bottom, top)
        _style_axis(ax, "F1")
    fig.suptitle("C1 zone and dwell F1 by zone")
    _save(fig, path)


def _plot_position_error(frame: pd.DataFrame, zones: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scopes = [("overall", "", "0.25")] + [
        (_zone_label(zone), f"_{zone['id']}", ZONE_COLORS.get(str(zone["id"]), "0.4"))
        for zone in zones
    ]
    for label, suffix, color in scopes:
        mean, _low, _high = _curve(frame, f"pos_mean_m{suffix}")
        p95, _p95_low, _p95_high = _curve(frame, f"pos_p95_m{suffix}")
        ax.plot(EPS_PX, mean, color=color, marker="o", label=f"{label} mean")
        ax.plot(EPS_PX, p95, color=color, marker="s", linestyle="--", label=f"{label} 95th")
    ax.set_ylim(bottom=0.0)
    ax.set_title("C1 position error vs calibration noise")
    _style_axis(ax, "position error (m)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def _plot_precision_recall(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    collected: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for ax, metric in zip(axes, ("precision", "recall")):
        for event_type in EVENT_TYPES:
            mean, low, high = _curve(frame, f"{metric}_{event_type}")
            collected.append((mean, low, high))
            color = EVENT_COLORS[event_type]
            ax.fill_between(EPS_PX, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(EPS_PX, mean, color=color, marker="o", label=event_type)
        ax.set_title(metric)
    bottom, top = _band_ylim(collected)
    for ax, metric in zip(axes, ("precision", "recall")):
        ax.set_ylim(bottom, top)
        _style_axis(ax, metric)
    fig.suptitle("C1 precision and recall vs calibration noise")
    _save(fig, path)


def main() -> None:
    """Run the C1 noise sweep, save the trial table, and write the summary plots."""
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    tracks = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]].copy()
    events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")
    # The first layout search is one-time work, so time a trial after it is cached.
    choose_calibration_points(cam_id)

    started = time.perf_counter()
    timed = run_trial(cam_id, 5.0, 0, tracks, events, cfg)
    trial_seconds = time.perf_counter() - started

    results = _run_sweep(cam_id, tracks, events, cfg, trial_seconds, timed)
    frame = pd.DataFrame(results)
    frame = frame.sort_values(["eps_px", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]

    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"sweep_{cam_id}_gt_tracks.csv"
    frame.to_csv(table_path, index=False)
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    zones = list(cfg["zones"])
    _print_summary(frame, zones)
    _plot_f1_by_type(frame, output_dir / f"f1_vs_eps_{cam_id}.png")
    _plot_f1_by_zone(frame, zones, output_dir / f"f1_vs_eps_by_zone_{cam_id}.png")
    _plot_position_error(frame, zones, output_dir / f"position_error_vs_eps_{cam_id}.png")
    _plot_precision_recall(frame, output_dir / f"precision_recall_vs_eps_{cam_id}.png")


if __name__ == "__main__":
    main()
