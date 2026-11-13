#!/usr/bin/env python3
"""Compare C1 vs C6 calibration sweeps (n-points and pitch bump) and plot overlays."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import project_root

EVENT_TYPES = ("zone", "dwell", "proximity")
LAYOUTS = ("n4", "n12")
EPS_COMPARE = (5.0, 10.0)
EPS_ALL = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
ANGLES = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
CAM_COLORS = {"C1": "#e07a2f", "C6": "#3d6fd8"}
LAYOUT_LINESTYLES = {"n4": "-", "n12": "--"}


def _mean_at(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = frame.loc[mask, column].to_numpy(dtype=float)
    return float(np.nanmean(values)) if len(values) else float("nan")


def _tolerance(frame: pd.DataFrame, layout: str, event_type: str) -> float:
    """Largest eps with mean F1 >= 0.95; 0 if even eps=0 fails (should not)."""
    best = float("nan")
    for eps in EPS_ALL:
        mask = (frame["layout"] == layout) & np.isclose(frame["eps_px"], eps)
        mean_f1 = _mean_at(frame, mask, f"f1_{event_type}")
        if mean_f1 >= 0.95 - 1e-12:
            best = float(eps)
    return best


def build_compare_table(c1: pd.DataFrame, c6: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for layout in LAYOUTS:
        for eps in EPS_COMPARE:
            row: dict = {
                "row_type": "f1_at_eps",
                "layout": layout,
                "eps_px": float(eps),
                "event_type": "",
            }
            for cam, frame in (("C1", c1), ("C6", c6)):
                mask = (frame["layout"] == layout) & np.isclose(frame["eps_px"], eps)
                for event_type in EVENT_TYPES:
                    row[f"{cam}_f1_{event_type}"] = _mean_at(frame, mask, f"f1_{event_type}")
                row[f"{cam}_pos_mean_m"] = _mean_at(frame, mask, "pos_mean_m")
            rows.append(row)
        for event_type in EVENT_TYPES:
            rows.append(
                {
                    "row_type": "tolerance",
                    "layout": layout,
                    "eps_px": float("nan"),
                    "event_type": event_type,
                    "C1_tolerance_eps": _tolerance(c1, layout, event_type),
                    "C6_tolerance_eps": _tolerance(c6, layout, event_type),
                }
            )
    return pd.DataFrame(rows)


def _print_compare(table: pd.DataFrame, c1: pd.DataFrame, c6: pd.DataFrame) -> None:
    print("\nC1 vs C6 mean F1 and position error at eps 5 and 10")
    print(
        f"{'layout':>6} {'eps':>4}  "
        f"{'C1_z':>6} {'C6_z':>6}  {'C1_d':>6} {'C6_d':>6}  "
        f"{'C1_p':>6} {'C6_p':>6}  {'C1_pos':>7} {'C6_pos':>7}"
    )
    f1_rows = table.loc[table["row_type"] == "f1_at_eps"]
    for _, r in f1_rows.iterrows():
        print(
            f"{r['layout']:>6} {r['eps_px']:4g}  "
            f"{r['C1_f1_zone']:6.3f} {r['C6_f1_zone']:6.3f}  "
            f"{r['C1_f1_dwell']:6.3f} {r['C6_f1_dwell']:6.3f}  "
            f"{r['C1_f1_proximity']:6.3f} {r['C6_f1_proximity']:6.3f}  "
            f"{r['C1_pos_mean_m']:7.4f} {r['C6_pos_mean_m']:7.4f}"
        )

    print("\nTolerance: largest eps with mean F1 >= 0.95")
    print(f"{'layout':>6} {'event':>10}  {'C1':>5} {'C6':>5}")
    for layout in LAYOUTS:
        for event_type in EVENT_TYPES:
            print(
                f"{layout:>6} {event_type:>10}  "
                f"{_tolerance(c1, layout, event_type):5g} "
                f"{_tolerance(c6, layout, event_type):5g}"
            )


def plot_f1_vs_eps(c1: pd.DataFrame, c6: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for cam, frame in (("C1", c1), ("C6", c6)):
            for layout in LAYOUTS:
                ys = [
                    _mean_at(
                        frame,
                        (frame["layout"] == layout) & np.isclose(frame["eps_px"], eps),
                        f"f1_{event_type}",
                    )
                    for eps in EPS_ALL
                ]
                ax.plot(
                    EPS_ALL,
                    ys,
                    color=CAM_COLORS[cam],
                    linestyle=LAYOUT_LINESTYLES[layout],
                    marker="o",
                    linewidth=1.8,
                    label=f"{cam} {layout}",
                )
        ax.set_title(event_type)
        ax.set_xlabel("calibration noise (px)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(EPS_ALL))
        ax.set_xlim(-0.4, 10.4)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("C1 vs C6 event F1 vs calibration noise", y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def plot_f1_vs_bump(c1: pd.DataFrame, c6: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for cam, frame in (("C1", c1), ("C6", c6)):
            pitch = frame.loc[frame["axis"] == "pitch"]
            ys = [
                _mean_at(pitch, np.isclose(pitch["angle_deg"], angle), f"f1_{event_type}")
                for angle in ANGLES
            ]
            ax.plot(
                ANGLES,
                ys,
                color=CAM_COLORS[cam],
                marker="o",
                linewidth=1.8,
                label=cam,
            )
        ax.set_title(event_type)
        ax.set_xlabel("pitch bump (deg)")
        ax.set_ylabel("F1")
        ax.set_xticks(list(ANGLES))
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("C1 vs C6 event F1 vs pitch bump", y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def main() -> None:
    out = project_root() / "outputs"
    c1_n = pd.read_csv(out / "sweep_C1_npoints.csv")
    c6_n = pd.read_csv(out / "sweep_C6_npoints.csv")
    c1_b = pd.read_csv(out / "sweep_C1_bump.csv")
    c6_b = pd.read_csv(out / "sweep_C6_bump.csv")

    table = build_compare_table(c1_n, c6_n)
    path = out / "compare_C1_C6.csv"
    table.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved {path.name} ({len(table)} rows)")
    _print_compare(table, c1_n, c6_n)
    plot_f1_vs_eps(c1_n, c6_n, out / "f1_vs_eps_C1_vs_C6.png")
    plot_f1_vs_bump(c1_b, c6_b, out / "f1_vs_bump_C1_vs_C6.png")


if __name__ == "__main__":
    main()
