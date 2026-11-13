#!/usr/bin/env python3
"""Rebuild every final report figure and table from saved CSVs only.

Writes PDF + PNG figures and CSV + LaTeX tables under outputs/report/.
Missing inputs are reported and skipped; no new experiments are run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import project_root

# --- shared style -----------------------------------------------------------------

EVENT_TYPES = ("zone", "dwell", "proximity")
EVENT_COLORS = {
    "zone": "#e07a2f",
    "dwell": "#2e9b4f",
    "proximity": "#3d6fd8",
}
ZONE_IDS = ("near", "medium", "far")
ZONE_LABELS = {"near": "Near", "medium": "Medium", "far": "Far"}
ZONE_COLORS = {
    "near": "#f2731a",
    "medium": "#26994d",
    "far": "#3366d9",
}
LAYOUTS = ("n4", "n6", "n8", "n12")
LAYOUT_COLORS = {
    "n4": "#e07a2f",
    "n6": "#2e9b4f",
    "n8": "#3d6fd8",
    "n12": "#c43c6a",
}
LAYOUT_MARKERS = {"n4": "o", "n6": "s", "n8": "^", "n12": "D"}
LAYOUT_LABELS = {"n4": "4 pts", "n6": "6 pts", "n8": "8 pts", "n12": "12 pts"}

MODEL_ORDER = (
    "oracle_gt_boxes",
    "yolo11n_bytetrack",
    "yolo11n_deepsort",
    "yolo11x_bytetrack",
    "yolo11x_deepsort",
)
MODEL_LABELS = {
    "oracle_gt_boxes": "oracle",
    "yolo11n_bytetrack": "YOLOn+Byte",
    "yolo11n_deepsort": "YOLOn+DS",
    "yolo11x_bytetrack": "YOLOx+Byte",
    "yolo11x_deepsort": "YOLOx+DS",
}
MODEL_COLORS = {
    "oracle_gt_boxes": "#222222",
    "yolo11n_bytetrack": "#e07a2f",
    "yolo11n_deepsort": "#2e9b4f",
    "yolo11x_bytetrack": "#3d6fd8",
    "yolo11x_deepsort": "#c43c6a",
}
GT_REF_COLOR = "0.45"
BEST_TRACKER = "yolo11x_deepsort"

FIG_SINGLE = (3.5, 2.6)
FIG_DOUBLE = (7.0, 2.6)
FIG_DOUBLE_TALL = (7.0, 2.8)
FONT_SIZE = 8
TITLE_SIZE = 9
LABEL_SIZE = 8
LEGEND_SIZE = 7
TICK_SIZE = 7
DPI = 300


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _paths() -> dict[str, Path]:
    root = project_root()
    out = root / "outputs" / "report"
    return {
        "out": out,
        "gt_sweep": root / "outputs" / "sweep_C1_gt_tracks.csv",
        "npoints": root / "outputs" / "sweep_C1_npoints.csv",
        "pred": root / "outputs" / "sweep_C1_pred_tracks.csv",
        "tolerance": root / "outputs" / "tolerance_npoints_C1.csv",
        "events": root / "outputs" / "events_on_tracks_C1.csv",
        "tracking": root / "results" / "tracks" / "tracking_summary_C1.csv",
    }


def _load(path: Path, created: list[str]) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"MISSING  {path.relative_to(project_root())}  — skipping dependents")
        return None
    return pd.read_csv(path)


def _save_fig(fig: plt.Figure, stem: Path, created: list[str]) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        created.append(str(path.relative_to(project_root())))
        print(f"Saved {path.relative_to(project_root())}")
    plt.close(fig)


def _eps_levels(frame: pd.DataFrame) -> list[float]:
    return [float(x) for x in sorted(frame["eps_px"].unique())]


def _curve(
    frame: pd.DataFrame,
    column: str,
    *,
    mask: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return eps, mean, p5, p95 for one column."""
    data = frame if mask is None else frame.loc[mask]
    eps = _eps_levels(data)
    means, lows, highs = [], [], []
    for value in eps:
        vals = data.loc[np.isclose(data["eps_px"], value), column].to_numpy(dtype=float)
        means.append(float(np.mean(vals)))
        lows.append(float(np.percentile(vals, 5)))
        highs.append(float(np.percentile(vals, 95)))
    return (
        np.asarray(eps, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
    )


def _style_eps_axis(ax: plt.Axes, eps: np.ndarray, ylabel: str) -> None:
    ax.set_xlabel("calibration noise (px)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(eps))
    ax.set_xlim(float(eps.min()) - 0.4, float(eps.max()) + 0.4)


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _write_csv(frame: pd.DataFrame, path: Path, created: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    created.append(str(path.relative_to(project_root())))
    print(f"Saved {path.relative_to(project_root())}")


def _write_tex(lines: list[str], path: Path, created: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    created.append(str(path.relative_to(project_root())))
    print(f"Saved {path.relative_to(project_root())}")


# --- figures ----------------------------------------------------------------------


def fig_f1_vs_eps(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    eps = None
    for event_type in EVENT_TYPES:
        x, mean, low, high = _curve(gt, f"f1_{event_type}")
        eps = x
        color = EVENT_COLORS[event_type]
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=color, marker="o", markersize=3.5, label=event_type, linewidth=1.4)
    assert eps is not None
    _style_eps_axis(ax, eps, "F1")
    ax.set_ylim(0.55, 1.02)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Event F1 vs calibration noise")
    fig.tight_layout()
    _save_fig(fig, out / "fig_f1_vs_eps", created)


def fig_f1_by_zone(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharey=True)
    eps = None
    for ax, event_type in zip(axes, ("zone", "dwell")):
        for zone_id in ZONE_IDS:
            x, mean, low, high = _curve(gt, f"f1_{event_type}_{zone_id}")
            eps = x
            color = ZONE_COLORS[zone_id]
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                x,
                mean,
                color=color,
                marker="o",
                markersize=3.5,
                label=ZONE_LABELS[zone_id],
                linewidth=1.4,
            )
        assert eps is not None
        _style_eps_axis(ax, eps, "F1" if event_type == "zone" else "")
        ax.set_title(event_type)
        ax.set_ylim(0.4, 1.02)
        ax.legend(frameon=False, loc="lower left")
    fig.suptitle("Zone and dwell F1 by zone", y=1.02, fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_fig(fig, out / "fig_f1_by_zone", created)


def fig_position_error(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    scopes = [("overall", "pos_mean_m", "0.25")] + [
        (ZONE_LABELS[z], f"pos_mean_m_{z}", ZONE_COLORS[z]) for z in ZONE_IDS
    ]
    eps = None
    for label, column, color in scopes:
        x, mean, _low, _high = _curve(gt, column)
        eps = x
        ax.plot(x, mean, color=color, marker="o", markersize=3.5, label=label, linewidth=1.4)
    assert eps is not None
    _style_eps_axis(ax, eps, "mean position error (m)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE - 0.5)
    ax.set_title("Position error vs calibration noise")
    fig.tight_layout()
    _save_fig(fig, out / "fig_position_error", created)


def fig_npoints(npoints: pd.DataFrame, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE, sharey=True)
    for ax, event_type in zip(axes, EVENT_TYPES):
        for layout in LAYOUTS:
            x, mean, _low, _high = _curve(
                npoints, f"f1_{event_type}", mask=npoints["layout"] == layout
            )
            ax.plot(
                x,
                mean,
                color=LAYOUT_COLORS[layout],
                marker=LAYOUT_MARKERS[layout],
                markersize=3.5,
                label=LAYOUT_LABELS[layout],
                linewidth=1.4,
            )
        _style_eps_axis(ax, np.asarray(_eps_levels(npoints)), "F1" if event_type == "zone" else "")
        ax.set_title(event_type)
        ax.set_ylim(0.55, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("F1 vs noise by number of calibration points", y=1.14, fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_fig(fig, out / "fig_npoints", created)


def fig_pred_tracks(pred: pd.DataFrame, gt: pd.DataFrame | None, out: Path, created: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG_DOUBLE_TALL, sharey=True)
    models = [m for m in MODEL_ORDER if m in set(pred["model"])]
    for ax, event_type in zip(axes, EVENT_TYPES):
        for model in models:
            x, mean, _low, _high = _curve(
                pred, f"f1_{event_type}_anonymous", mask=pred["model"] == model
            )
            ax.plot(
                x,
                mean,
                color=MODEL_COLORS.get(model, "0.3"),
                marker="o",
                markersize=3.0,
                label=MODEL_LABELS.get(model, model),
                linewidth=1.3,
            )
        if gt is not None:
            x_gt, mean_gt, _l, _h = _curve(gt, f"f1_{event_type}")
            ax.plot(
                x_gt,
                mean_gt,
                color=GT_REF_COLOR,
                linestyle="--",
                linewidth=1.3,
                label="GT tracks",
            )
        _style_eps_axis(ax, np.asarray(_eps_levels(pred)), "anonymous F1" if event_type == "zone" else "")
        ax.set_title(event_type)
        ax.set_ylim(0.0, 1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Anonymous F1 on predicted tracks", y=1.18, fontsize=TITLE_SIZE)
    fig.tight_layout()
    _save_fig(fig, out / "fig_pred_tracks", created)


def fig_perception_vs_calibration(
    gt: pd.DataFrame,
    events: pd.DataFrame,
    out: Path,
    created: list[str],
) -> None:
    """Compare GT tracks at 10 px vs best tracker at perfect calibration (identity)."""
    calib = []
    for event_type in EVENT_TYPES:
        vals = gt.loc[np.isclose(gt["eps_px"], 10.0), f"f1_{event_type}"].to_numpy(dtype=float)
        calib.append(float(np.mean(vals)))
    track = []
    row = events.loc[
        (events["model"] == BEST_TRACKER) & (events["scoring"] == "identity")
    ]
    if row.empty:
        print(f"MISSING  identity row for {BEST_TRACKER} in events_on_tracks — skipping fig_perception_vs_calibration")
        return
    row = row.iloc[0]
    for event_type in EVENT_TYPES:
        track.append(float(row[f"f1_{event_type}"]))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    x = np.arange(len(EVENT_TYPES))
    width = 0.36
    bars1 = ax.bar(
        x - width / 2,
        calib,
        width,
        label="GT tracks @ 10 px",
        color="#7a7a7a",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        track,
        width,
        label=f"{MODEL_LABELS[BEST_TRACKER]} @ 0 px",
        color=[EVENT_COLORS[et] for et in EVENT_TYPES],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(EVENT_TYPES))
    ax.set_ylabel("identity F1")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Perception error vs calibration noise")
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    fig.tight_layout()
    _save_fig(fig, out / "fig_perception_vs_calibration", created)


# --- tables -----------------------------------------------------------------------


def tab_sweep_gt(gt: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for eps in _eps_levels(gt):
        row: dict[str, Any] = {"eps_px": eps}
        for event_type in EVENT_TYPES:
            vals = gt.loc[np.isclose(gt["eps_px"], eps), f"f1_{event_type}"].to_numpy(dtype=float)
            mean = float(np.mean(vals))
            p5 = float(np.percentile(vals, 5))
            p95 = float(np.percentile(vals, 95))
            row[f"f1_{event_type}"] = mean
            row[f"f1_{event_type}_p5"] = p5
            row[f"f1_{event_type}_p95"] = p95
            row[f"f1_{event_type}_cell"] = f"{mean:.3f} ({p5:.3f}–{p95:.3f})"
        pos = gt.loc[np.isclose(gt["eps_px"], eps), "pos_mean_m"].to_numpy(dtype=float)
        row["pos_mean_m"] = float(np.mean(pos))
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(
        table[
            ["eps_px"]
            + [f"f1_{et}" for et in EVENT_TYPES]
            + [f"f1_{et}_p5" for et in EVENT_TYPES]
            + [f"f1_{et}_p95" for et in EVENT_TYPES]
            + ["pos_mean_m"]
        ],
        out / "tab_sweep_gt.csv",
        created,
    )
    lines = [
        r"\begin{tabular}{rcccc}",
        r"\toprule",
        r"$\varepsilon$ (px) & Zone F1 & Dwell F1 & Proximity F1 & Pos. err. (m) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['eps_px']:g} & "
            f"{row['f1_zone_cell']} & "
            f"{row['f1_dwell_cell']} & "
            f"{row['f1_proximity_cell']} & "
            f"{row['pos_mean_m']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_sweep_gt.tex", created)


def tab_tolerance(tol: pd.DataFrame, out: Path, created: list[str]) -> None:
    sub = tol.loc[tol["table"] == "tolerance"].copy()
    if sub.empty:
        print("MISSING  tolerance rows in tolerance_npoints_C1.csv — skipping tab_tolerance")
        return
    keep = sub[["layout", "event_type", "max_eps_mean_f1_ge_0.95", "max_eps_p5_f1_ge_0.95"]].copy()
    keep = keep.rename(
        columns={
            "max_eps_mean_f1_ge_0.95": "eps_mean_ge_0.95",
            "max_eps_p5_f1_ge_0.95": "eps_p5_ge_0.95",
        }
    )
    # Stable order.
    keep["layout"] = pd.Categorical(keep["layout"], list(LAYOUTS), ordered=True)
    keep["event_type"] = pd.Categorical(keep["event_type"], list(EVENT_TYPES), ordered=True)
    keep = keep.sort_values(["layout", "event_type"]).reset_index(drop=True)
    _write_csv(keep, out / "tab_tolerance.csv", created)

    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Layout & Event & $\varepsilon_{\mathrm{mean}\ge 0.95}$ & $\varepsilon_{\mathrm{p5}\ge 0.95}$ \\",
        r"\midrule",
    ]
    for _, row in keep.iterrows():
        lines.append(
            f"{row['layout']} & {row['event_type']} & "
            f"{_fmt(float(row['eps_mean_ge_0.95']), 0)} & "
            f"{_fmt(float(row['eps_p5_ge_0.95']), 0)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_tolerance.tex", created)


def tab_tracking(tracking: pd.DataFrame, out: Path, created: list[str]) -> None:
    cols = {
        "model": "model",
        "ground_det_precision": "precision",
        "ground_det_recall": "recall",
        "ground_id_switches": "id_switches",
        "ground_mota": "mota",
    }
    table = tracking[list(cols.keys())].rename(columns=cols).copy()
    table["model"] = pd.Categorical(table["model"], list(MODEL_ORDER), ordered=True)
    table = table.sort_values("model").reset_index(drop=True)
    _write_csv(table, out / "tab_tracking.csv", created)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Precision & Recall & ID switches & MOTA \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{MODEL_LABELS.get(str(row['model']), row['model'])} & "
            f"{row['precision']:.3f} & {row['recall']:.3f} & "
            f"{int(row['id_switches'])} & {row['mota']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_tracking.tex", created)


def tab_events_tracks(events: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        row: dict[str, Any] = {"model": model}
        for scoring in ("identity", "anonymous"):
            match = events.loc[(events["model"] == model) & (events["scoring"] == scoring)]
            if match.empty:
                for et in EVENT_TYPES:
                    row[f"f1_{et}_{scoring}"] = np.nan
            else:
                r = match.iloc[0]
                for et in EVENT_TYPES:
                    row[f"f1_{et}_{scoring}"] = float(r[f"f1_{et}"])
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(table, out / "tab_events_tracks.csv", created)
    lines = [
        r"\begin{tabular}{lccc|ccc}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{Identity F1} & \multicolumn{3}{c}{Anonymous F1} \\",
        r"Model & Zone & Dwell & Prox. & Zone & Dwell & Prox. \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{MODEL_LABELS.get(str(row['model']), row['model'])} & "
            f"{_fmt(row['f1_zone_identity'])} & {_fmt(row['f1_dwell_identity'])} & {_fmt(row['f1_proximity_identity'])} & "
            f"{_fmt(row['f1_zone_anonymous'])} & {_fmt(row['f1_dwell_anonymous'])} & {_fmt(row['f1_proximity_anonymous'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_events_tracks.tex", created)


def tab_relative_f1(pred: pd.DataFrame, out: Path, created: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        if model not in set(pred["model"]):
            continue
        m = pred.loc[pred["model"] == model]
        row: dict[str, Any] = {"model": model}
        for event_type, scoring in (
            ("zone", "anonymous"),
            ("dwell", "anonymous"),
            ("proximity", "identity"),
        ):
            col = f"f1_{event_type}_{scoring}"
            base = float(m.loc[np.isclose(m["eps_px"], 0.0), col].mean())
            at10 = float(m.loc[np.isclose(m["eps_px"], 10.0), col].mean())
            rel = float("nan") if not np.isfinite(base) or base == 0.0 else at10 / base
            row[f"rel_f1_{event_type}"] = rel
            row[f"f1_{event_type}_eps0"] = base
            row[f"f1_{event_type}_eps10"] = at10
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_csv(table, out / "tab_relative_f1.csv", created)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Zone (anon.) & Dwell (anon.) & Prox. (identity) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{MODEL_LABELS.get(str(row['model']), row['model'])} & "
            f"{_fmt(row['rel_f1_zone'])} & {_fmt(row['rel_f1_dwell'])} & {_fmt(row['rel_f1_proximity'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    _write_tex(lines, out / "tab_relative_f1.tex", created)


# --- main -------------------------------------------------------------------------


def main() -> None:
    _apply_style()
    paths = _paths()
    out = paths["out"]
    out.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    gt = _load(paths["gt_sweep"], created)
    npoints = _load(paths["npoints"], created)
    pred = _load(paths["pred"], created)
    tol = _load(paths["tolerance"], created)
    events = _load(paths["events"], created)
    tracking = _load(paths["tracking"], created)

    if gt is not None:
        fig_f1_vs_eps(gt, out, created)
        fig_f1_by_zone(gt, out, created)
        fig_position_error(gt, out, created)
        tab_sweep_gt(gt, out, created)
    if npoints is not None:
        fig_npoints(npoints, out, created)
    if pred is not None:
        fig_pred_tracks(pred, gt, out, created)
        tab_relative_f1(pred, out, created)
    if gt is not None and events is not None:
        fig_perception_vs_calibration(gt, events, out, created)
    if tol is not None:
        tab_tolerance(tol, out, created)
    if tracking is not None:
        tab_tracking(tracking, out, created)
    if events is not None:
        tab_events_tracks(events, out, created)

    print("\nCreated files:")
    for path in created:
        print(f"  {path}")
    print(f"Total: {len(created)}")


if __name__ == "__main__":
    main()
