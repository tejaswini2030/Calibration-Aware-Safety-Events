#!/usr/bin/env python3
"""Sweep calibration noise across WHERE the points are placed (C1, GT tracks).

Compares wide / narrow / near_only / far_only rectangles, each as 4 corners and
as a 3 x 4 grid. Existing sweep CSV files are not modified.
"""

from __future__ import annotations

import copy
import json
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
from src.perturb import (
    LAYOUT_RECTANGLES_M,
    _first_frame,
    choose_calibration_points,
    corners_project_inside_image,
    mark_zones_inside_calibration_hull,
    run_trial,
)

# (family, n_points layout name, point count)
LAYOUT_SPECS: tuple[tuple[str, str, int], ...] = (
    ("wide", "wide", 4),
    ("wide", "wide_n12", 12),
    ("narrow", "narrow", 4),
    ("narrow", "narrow_n12", 12),
    ("near_only", "near_only", 4),
    ("near_only", "near_only_n12", 12),
    ("far_only", "far_only", 4),
    ("far_only", "far_only_n12", 12),
)
LAYOUT_NAMES = tuple(spec[1] for spec in LAYOUT_SPECS)
FAMILIES = ("wide", "narrow", "near_only", "far_only")

EPS_PX = (0, 1, 3, 5, 10)
N_TRIALS = 100
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
ZONE_IDS = ("near", "medium", "far")
PARALLEL_IF_SLOWER_THAN_S = 0.2

FAMILY_COLORS = {
    "wide": "#e07a2f",
    "narrow": "#2e9b4f",
    "near_only": "#3d6fd8",
    "far_only": "#c43c6a",
}
ZONE_COLORS = {
    "near": "#f2731a",
    "medium": "#26994d",
    "far": "#3366d9",
}
ZONE_LABELS = {"near": "Near", "medium": "Medium", "far": "Far"}

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


def _family_of(layout: str) -> str:
    return layout.removesuffix("_n12")


def _jobs() -> list[tuple[str, float, int]]:
    return [
        (layout, float(eps), seed)
        for layout in LAYOUT_NAMES
        for eps in EPS_PX
        for seed in SEEDS
    ]


def _worker(job: tuple[str, float, int]) -> dict[str, Any]:
    layout, eps_px, seed = job
    result = run_trial(_CAM_ID, eps_px, seed, _TRACKS, _EVENTS, _CFG, layout=layout)
    result["layout"] = layout
    result["family"] = _family_of(layout)
    result["n_points"] = 12 if layout.endswith("_n12") else 4
    return result


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


def _check_layouts(cam_id: str) -> dict[str, dict[str, bool]]:
    """Verify corners project inside the image; return zone-in-hull flags per layout."""
    print("\nCorner projection check (must be inside the C1 image):")
    hull_flags: dict[str, dict[str, bool]] = {}
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    for family, layout, n_points in LAYOUT_SPECS:
        world, pixels = choose_calibration_points(cam_id, layout=layout)
        inside = corners_project_inside_image(cam_id, world, margin=1.0)
        bounds = LAYOUT_RECTANGLES_M[family]
        status = "PASS" if inside else "FAIL"
        print(
            f"  {status}  {layout:<16} N={n_points}  "
            f"X[{bounds[0]:g},{bounds[1]:g}] Y[{bounds[2]:g},{bounds[3]:g}]  "
            f"pixels u[{pixels[:, 0].min():.0f},{pixels[:, 0].max():.0f}] "
            f"v[{pixels[:, 1].min():.0f},{pixels[:, 1].max():.0f}]"
        )
        if not inside:
            raise RuntimeError(f"Layout {layout} has corners outside the image.")
        zones = copy.deepcopy(cfg["zones"])
        mark_zones_inside_calibration_hull(zones, world)
        hull_flags[layout] = {str(z["id"]): bool(z["inside_calibration_hull"]) for z in zones}
    print("\nZone inside layout convex hull:")
    print(f"  {'layout':<16} " + " ".join(f"{z:>8}" for z in ZONE_IDS))
    for layout in LAYOUT_NAMES:
        flags = hull_flags[layout]
        cells = " ".join(f"{'yes' if flags[z] else 'no':>8}" for z in ZONE_IDS)
        print(f"  {layout:<16} {cells}")
    return hull_flags


def _save_layout_overlay(cam_id: str, path: Path) -> None:
    import cv2

    _path, image = _first_frame(cam_id)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, (family, layout, n_points) in zip(axes.ravel(), LAYOUT_SPECS):
        world, pixels = choose_calibration_points(cam_id, layout=layout)
        ax.imshow(rgb)
        # Draw the rectangle outline from the 4 corners of this family.
        _, corner_px = choose_calibration_points(cam_id, layout=family)
        closed = np.vstack([corner_px, corner_px[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color="0.85", linewidth=1.2, linestyle="--")
        ax.scatter(
            pixels[:, 0],
            pixels[:, 1],
            s=40,
            c=FAMILY_COLORS[family],
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
        )
        ax.set_title(f"{layout}  (N={n_points})")
        ax.set_xlim(0, rgb.shape[1])
        ax.set_ylim(rgb.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{cam_id} calibration layouts (4-point and 12-point)", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _group(frame: pd.DataFrame, layout: str, eps: float, column: str) -> np.ndarray:
    mask = (frame["layout"] == layout) & np.isclose(frame["eps_px"], eps)
    return frame.loc[mask, column].to_numpy(dtype=float)


def _sanity_check(frame: pd.DataFrame) -> bool:
    print("\nSanity check: eps=0 F1 == 1.0 for every layout")
    ok = True
    for layout in LAYOUT_NAMES:
        for event_type in EVENT_TYPES:
            values = _group(frame, layout, 0.0, f"f1_{event_type}")
            if len(values) == 0 or not np.allclose(values, 1.0, atol=1e-9):
                bad = float(np.min(values)) if len(values) else float("nan")
                print(f"  FAIL  {layout} {event_type}: min F1={bad:.6f}")
                ok = False
            else:
                print(f"  PASS  {layout} {event_type}: F1=1.000")
    print("Sanity check:", "PASS" if ok else "FAIL")
    return ok


def _print_summary(frame: pd.DataFrame, hull_flags: dict[str, dict[str, bool]]) -> None:
    print(f"\n{N_TRIALS} trials at each eps, seeds {SEEDS[0]}..{SEEDS[-1]}")
    for layout in LAYOUT_NAMES:
        family = _family_of(layout)
        n_points = 12 if layout.endswith("_n12") else 4
        print(f"\n{layout} (family={family}, N={n_points})")
        print(
            f"  {'eps':>5}  "
            f"{'F1z':>7} {'F1d':>7} {'F1p':>7}  "
            f"{'pos':>7}  "
            + " ".join(f"{'F1z_'+z:>8}" for z in ZONE_IDS)
            + "  "
            + " ".join(f"{'pos_'+z:>7}" for z in ZONE_IDS)
        )
        for eps in EPS_PX:
            f1 = {et: float(np.mean(_group(frame, layout, float(eps), f"f1_{et}"))) for et in EVENT_TYPES}
            pos = float(np.mean(_group(frame, layout, float(eps), "pos_mean_m")))
            f1z = [float(np.mean(_group(frame, layout, float(eps), f"f1_zone_{z}"))) for z in ZONE_IDS]
            posz = [float(np.mean(_group(frame, layout, float(eps), f"pos_mean_m_{z}"))) for z in ZONE_IDS]
            print(
                f"  {eps:5g}  "
                f"{f1['zone']:7.3f} {f1['dwell']:7.3f} {f1['proximity']:7.3f}  "
                f"{pos:7.4f}  "
                + " ".join(f"{v:8.3f}" for v in f1z)
                + "  "
                + " ".join(f"{v:7.4f}" for v in posz)
            )
        flags = hull_flags[layout]
        hull = ", ".join(f"{z}={'in' if flags[z] else 'out'}" for z in ZONE_IDS)
        print(f"  hull: {hull}")


def _plot_f1(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for family in FAMILIES:
            for n_points, linestyle, marker in ((4, "-", "o"), (12, "--", "s")):
                layout = family if n_points == 4 else f"{family}_n12"
                ys = [float(np.mean(_group(frame, layout, float(eps), f"f1_{event_type}"))) for eps in EPS_PX]
                ax.plot(
                    EPS_PX,
                    ys,
                    color=FAMILY_COLORS[family],
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4,
                    linewidth=1.6,
                    label=f"{family} N={n_points}",
                )
        ax.set_title(event_type)
        ax.set_xlabel("calibration noise (px)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(EPS_PX))
        ax.set_xlim(-0.4, 10.4)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("C1 event F1 vs noise by calibration layout", y=1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _plot_position_by_zone(
    frame: pd.DataFrame,
    hull_flags: dict[str, dict[str, bool]],
    path: Path,
    eps_px: float = 5.0,
) -> None:
    """Grouped bars: position error at fixed eps, hatched when zone is outside the hull."""
    layouts = list(LAYOUT_NAMES)
    x = np.arange(len(layouts))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, zone_id in enumerate(ZONE_IDS):
        heights = []
        outside = []
        for layout in layouts:
            heights.append(float(np.mean(_group(frame, layout, eps_px, f"pos_mean_m_{zone_id}"))))
            outside.append(not hull_flags[layout][zone_id])
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            heights,
            width,
            label=ZONE_LABELS[zone_id],
            color=ZONE_COLORS[zone_id],
            edgecolor="black",
            linewidth=0.4,
        )
        for bar, is_out in zip(bars, outside):
            if is_out:
                bar.set_hatch("///")
                bar.set_alpha(0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=30, ha="right")
    ax.set_ylabel("mean position error (m)")
    ax.set_title(f"Position error by zone at eps = {eps_px:g} px (hatched = outside hull)")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def _column_order(columns: list[str]) -> list[str]:
    ordered = ["layout", "family", "n_points", "eps_px", "seed"]
    for prefix in ("f1_", "precision_", "recall_", "start_delay_mean_", "pos_"):
        ordered.extend(sorted(c for c in columns if c.startswith(prefix)))
    ordered.extend(c for c in columns if c not in ordered)
    return ordered


def _save_hull_metadata(cam_id: str, hull_flags: dict[str, dict[str, bool]], path: Path) -> None:
    payload = {
        "camera": cam_id,
        "layouts": {
            layout: {
                "family": _family_of(layout),
                "n_points": 12 if layout.endswith("_n12") else 4,
                "rectangle_m": list(LAYOUT_RECTANGLES_M[_family_of(layout)]),
                "zones": {
                    zone_id: {"inside_calibration_hull": bool(inside)}
                    for zone_id, inside in flags.items()
                },
            }
            for layout, flags in hull_flags.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path.name}")


def main() -> None:
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    tracks = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]].copy()
    events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    hull_flags = _check_layouts(cam_id)
    _save_hull_metadata(cam_id, hull_flags, output_dir / f"layout_hull_metadata_{cam_id}.json")
    _save_layout_overlay(cam_id, output_dir / f"calib_points_layouts_{cam_id}.png")

    # Warm caches before timing.
    for layout in LAYOUT_NAMES:
        choose_calibration_points(cam_id, layout=layout)

    started = time.perf_counter()
    timed = run_trial(cam_id, 5.0, 0, tracks, events, cfg, layout="narrow_n12")
    timed["layout"] = "narrow_n12"
    timed["family"] = "narrow"
    timed["n_points"] = 12
    trial_seconds = time.perf_counter() - started

    results = _run_sweep(cam_id, tracks, events, cfg, trial_seconds, timed)
    frame = pd.DataFrame(results)
    frame = frame.sort_values(["layout", "eps_px", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]

    table_path = output_dir / f"sweep_{cam_id}_layouts.csv"
    frame.to_csv(table_path, index=False, float_format="%.6f")
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    _print_summary(frame, hull_flags)
    _sanity_check(frame)
    _plot_f1(frame, output_dir / f"f1_vs_eps_layouts_{cam_id}.png")
    _plot_position_by_zone(frame, hull_flags, output_dir / f"position_error_by_zone_layouts_{cam_id}.png")


if __name__ == "__main__":
    main()
