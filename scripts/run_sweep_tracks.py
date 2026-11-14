#!/usr/bin/env python3
"""Calibration-noise sweep on predicted C1 tracks (and an oracle of GT boxes).

Same wide 4-point layout and eps levels as the ground-truth-track sweep, but
each trial projects detector/tracker box bottoms through the *estimated*
homography. The H*-based idmap is held fixed so identity scores only change
through geometry, not through rematching. Zone and dwell use the full
on-ground feet; proximity uses the 1 m-inset inner area, matching
scripts/run_events_on_tracks.py. Each trial is scored with both identity and
anonymous matching.

Writes outputs/sweep_C1_pred_tracks.csv as a checkpoint so a killed run can
resume. Pass --force to delete that file and recompute every trial.
"""

from __future__ import annotations

import argparse
import importlib.util
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

from src.data import DATA_ROOT, load_annotations, normalize_cam_id, project_root, resolve_dataset_root
from src.events import load_event_config
from src.geometry import image_to_world
from src.perturb import (
    choose_calibration_points,
    estimate_H_img_to_world,
    perturb_pixels,
)
from src.tracking import (
    DEFAULT_BOUNDARY_MARGIN_M,
    drop_outside_ground_area,
    iter_track_files,
    relabel_tracks,
    validate_tracks,
)

_EOT_PATH = Path(__file__).resolve().parent / "run_events_on_tracks.py"
_EOT_SPEC = importlib.util.spec_from_file_location("run_events_on_tracks", _EOT_PATH)
assert _EOT_SPEC is not None and _EOT_SPEC.loader is not None
_eot = importlib.util.module_from_spec(_EOT_SPEC)
_EOT_SPEC.loader.exec_module(_eot)

EPS_PX = (0, 1, 3, 5, 10)
N_TRIALS = 100
SEEDS = tuple(range(N_TRIALS))
EVENT_TYPES = ("zone", "dwell", "proximity")
PARALLEL_IF_SLOWER_THAN_S = 1.0
ORACLE_MODEL = "oracle_gt_boxes"
SCORING_KINDS = ("identity", "anonymous")

MODEL_COLORS = {
    "oracle_gt_boxes": "#222222",
    "yolo11n_bytetrack": "#e07a2f",
    "yolo11n_deepsort": "#2e9b4f",
    "yolo11x_bytetrack": "#3d6fd8",
    "yolo11x_deepsort": "#c43c6a",
}
GT_REF_COLOR = "0.45"

_CAM_ID: str | None = None
_CFG: dict[str, Any] | None = None
_MARGIN: float = DEFAULT_BOUNDARY_MARGIN_M
_WORLD_PTS: np.ndarray | None = None
_EXACT_PIXELS: np.ndarray | None = None
_GT_EVENTS: pd.DataFrame | None = None
_GT_PROXIMITY: pd.DataFrame | None = None
_GT_PROXIMITY_TRACKS: pd.DataFrame | None = None
_MODEL_FEET: dict[str, pd.DataFrame] | None = None


def _init_worker(
    cam_id: str,
    cfg: dict[str, Any],
    margin: float,
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    gt_events: pd.DataFrame,
    gt_proximity: pd.DataFrame,
    gt_proximity_tracks: pd.DataFrame,
    model_feet: dict[str, pd.DataFrame],
) -> None:
    """Keep shared inputs in each worker so jobs stay small."""
    global _CAM_ID, _CFG, _MARGIN, _WORLD_PTS, _EXACT_PIXELS
    global _GT_EVENTS, _GT_PROXIMITY, _GT_PROXIMITY_TRACKS, _MODEL_FEET
    _CAM_ID = cam_id
    _CFG = cfg
    _MARGIN = float(margin)
    _WORLD_PTS = world_pts
    _EXACT_PIXELS = exact_pixels
    _GT_EVENTS = gt_events
    _GT_PROXIMITY = gt_proximity
    _GT_PROXIMITY_TRACKS = gt_proximity_tracks
    _MODEL_FEET = model_feet


def _jobs(models: list[str]) -> list[tuple[str, float, int]]:
    """All (model, eps, seed) jobs. eps=0 is deterministic, so only seed 0 is computed."""
    jobs: list[tuple[str, float, int]] = []
    for model in models:
        for eps_px in EPS_PX:
            seeds = (0,) if float(eps_px) == 0.0 else SEEDS
            for seed in seeds:
                jobs.append((model, float(eps_px), int(seed)))
    return jobs


def _expand_eps0_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replicate each eps=0 seed-0 row across seeds 0..N-1 (identical H)."""
    expanded: list[dict[str, Any]] = []
    for row in results:
        if float(row["eps_px"]) != 0.0:
            expanded.append(row)
            continue
        if int(row["seed"]) != 0:
            continue
        for seed in SEEDS:
            copy = dict(row)
            copy["seed"] = int(seed)
            expanded.append(copy)
    return expanded


def _load_completed(path: Path) -> dict[tuple[str, float, int], dict[str, Any]]:
    """Resume map of completed (model, eps, seed) → row. eps=0 keeps only seed 0."""
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    done: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        eps = float(row["eps_px"])
        seed = int(row["seed"])
        if eps == 0.0 and seed != 0:
            continue
        key = (str(row["model"]), eps, seed)
        done[key] = row
    return done


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore/delete the checkpoint CSV and recompute every trial",
    )
    return parser.parse_args()


def _checkpoint(path: Path, results: list[dict[str, Any]]) -> None:
    """Write the expanded trial table so a killed run can resume."""
    frame = pd.DataFrame(_expand_eps0_rows(results))
    frame = frame.sort_values(["model", "eps_px", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]
    frame.to_csv(path, index=False, float_format="%.6f")


def _feet_from_boxes(boxes: pd.DataFrame) -> pd.DataFrame:
    """Boxes with person_id → time, person_id, and bottom-center pixel."""
    needed = ["time_sec", "person_id", "xmin", "xmax", "ymax"]
    missing = [column for column in needed if column not in boxes.columns]
    if missing:
        raise ValueError(f"boxes missing columns: {missing}")
    u = (boxes["xmin"].to_numpy(dtype=float) + boxes["xmax"].to_numpy(dtype=float)) / 2.0
    v = boxes["ymax"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "time_sec": boxes["time_sec"].to_numpy(dtype=float),
            "person_id": boxes["person_id"].to_numpy(dtype=int),
            "u": u,
            "v": v,
        }
    )


def prepare_oracle_feet(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible GT boxes as an oracle track table (person_id already correct)."""
    visible = annotations.loc[annotations["visible"]].copy()
    if visible.empty:
        return pd.DataFrame(columns=["time_sec", "person_id", "u", "v"])
    boxes = pd.DataFrame(
        {
            "frame_idx": visible["frame_idx"].astype(int),
            "frame_number": visible["frame_number"].astype(int),
            "time_sec": visible["time_sec"].astype(float),
            "track_id": visible["person_id"].astype(int),
            "xmin": visible["xmin"].astype(float),
            "ymin": visible["ymin"].astype(float),
            "xmax": visible["xmax"].astype(float),
            "ymax": visible["ymax"].astype(float),
            "score": 1.0,
            "person_id": visible["person_id"].astype(int),
        }
    )
    return _feet_from_boxes(boxes)


def prepare_model_feet(tracks: pd.DataFrame, idmap: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Apply the fixed H*-based idmap and keep bottom-center pixels."""
    validate_tracks(tracks)
    relabeled, dropped = relabel_tracks(tracks, idmap)
    return _feet_from_boxes(relabeled), dropped


def run_trial_tracks(
    model: str,
    eps_px: float,
    seed: int,
    feet: pd.DataFrame,
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    gt_events: pd.DataFrame,
    gt_proximity: pd.DataFrame,
    gt_proximity_tracks: pd.DataFrame,
    cfg: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    """One (model, eps, seed): estimate H, project feet, detect, score both ways."""
    rng = np.random.default_rng(int(seed))
    perturbed = perturb_pixels(exact_pixels, float(eps_px), rng)
    H_estimated = estimate_H_img_to_world(perturbed, world_pts)
    uv = feet[["u", "v"]].to_numpy(dtype=np.float64)
    xy = image_to_world(H_estimated, uv)
    positions = pd.DataFrame(
        {
            "time_sec": feet["time_sec"].to_numpy(dtype=float),
            "person_id": feet["person_id"].to_numpy(dtype=int),
            "X_m": xy[:, 0],
            "Y_m": xy[:, 1],
        }
    )
    on_ground, _outside = drop_outside_ground_area(positions)
    inner = _eot.filter_inner_area(on_ground, margin)
    events = _eot.detect_events_on_positions(on_ground, cfg, proximity_tracks=inner)
    identity = _eot.score_model_events(events, gt_events, gt_proximity, cfg)
    anonymous = _eot.score_model_events_anonymous(
        events,
        gt_events,
        gt_proximity,
        pred_proximity_tracks=inner,
        gt_proximity_tracks=gt_proximity_tracks,
        cfg=cfg,
    )

    result: dict[str, Any] = {
        "model": model,
        "eps_px": float(eps_px),
        "seed": int(seed),
    }
    for scoring_name, scores in (("identity", identity), ("anonymous", anonymous)):
        for event_type in EVENT_TYPES:
            result[f"f1_{event_type}_{scoring_name}"] = float(scores[f"f1_{event_type}"])
            result[f"precision_{event_type}_{scoring_name}"] = float(scores[f"precision_{event_type}"])
            result[f"recall_{event_type}_{scoring_name}"] = float(scores[f"recall_{event_type}"])
    return result


def _worker(job: tuple[str, float, int]) -> dict[str, Any]:
    model, eps_px, seed = job
    assert _MODEL_FEET is not None and _CFG is not None
    assert _WORLD_PTS is not None and _EXACT_PIXELS is not None
    assert _GT_EVENTS is not None and _GT_PROXIMITY is not None and _GT_PROXIMITY_TRACKS is not None
    return run_trial_tracks(
        model,
        eps_px,
        seed,
        _MODEL_FEET[model],
        _WORLD_PTS,
        _EXACT_PIXELS,
        _GT_EVENTS,
        _GT_PROXIMITY,
        _GT_PROXIMITY_TRACKS,
        _CFG,
        _MARGIN,
    )


def _run_sweep(
    models: list[str],
    cam_id: str,
    cfg: dict[str, Any],
    margin: float,
    world_pts: np.ndarray,
    exact_pixels: np.ndarray,
    gt_events: pd.DataFrame,
    gt_proximity: pd.DataFrame,
    gt_proximity_tracks: pd.DataFrame,
    model_feet: dict[str, pd.DataFrame],
    trial_seconds: float,
    timed_result: dict[str, Any],
    checkpoint_path: Path,
    completed: dict[tuple[str, float, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run every (model, eps, seed), reusing the trial that was timed."""
    timed_key = (str(timed_result["model"]), float(timed_result["eps_px"]), int(timed_result["seed"]))
    completed = dict(completed)
    completed[timed_key] = timed_result
    all_jobs = _jobs(models)
    pending = [job for job in all_jobs if job not in completed]
    results = list(completed.values())
    initargs = (
        cam_id,
        cfg,
        margin,
        world_pts,
        exact_pixels,
        gt_events,
        gt_proximity,
        gt_proximity_tracks,
        model_feet,
    )
    checkpoint_every = 25

    def _store(result: dict[str, Any]) -> None:
        key = (str(result["model"]), float(result["eps_px"]), int(result["seed"]))
        completed[key] = result
        results.append(result)
        if len(completed) % checkpoint_every == 0 or len(completed) == len(all_jobs):
            _checkpoint(checkpoint_path, list(completed.values()))

    if not pending:
        print(f"All {len(all_jobs)} computed trials already finished.")
        _checkpoint(checkpoint_path, results)
        return list(completed.values())

    initial = len(all_jobs) - len(pending)
    if trial_seconds > PARALLEL_IF_SLOWER_THAN_S:
        workers = max(1, (os.cpu_count() or 2) - 1)
        print(
            f"One trial takes {trial_seconds:.2f} s. Running {workers} worker processes "
            f"({len(pending)} remaining of {len(all_jobs)} computed trials; "
            f"eps=0 uses seed 0 only)."
        )
        with Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, pending, chunksize=4),
                total=len(all_jobs),
                initial=initial,
                desc="trials",
            ):
                _store(result)
    else:
        print(
            f"One trial takes {trial_seconds:.2f} s. Running sequentially "
            f"({len(pending)} remaining of {len(all_jobs)})."
        )
        _init_worker(*initargs)
        for job in tqdm(pending, total=len(all_jobs), initial=initial, desc="trials"):
            _store(_worker(job))
    return list(completed.values())


def _group_values(frame: pd.DataFrame, model: str, eps_px: float, column: str) -> np.ndarray:
    mask = (frame["model"] == model) & np.isclose(frame["eps_px"], eps_px)
    return frame.loc[mask, column].to_numpy(dtype=float)


def _mean_f1_table(frame: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Mean F1 and relative F1 = mean(eps) / mean(eps=0) per model and noise."""
    rows: list[dict[str, Any]] = []
    for model in models:
        baselines: dict[tuple[str, str], float] = {}
        for scoring in SCORING_KINDS:
            for event_type in EVENT_TYPES:
                baselines[(scoring, event_type)] = float(
                    np.mean(_group_values(frame, model, 0.0, f"f1_{event_type}_{scoring}"))
                )
        for eps_px in EPS_PX:
            row: dict[str, Any] = {"model": model, "eps_px": float(eps_px)}
            for scoring in SCORING_KINDS:
                for event_type in EVENT_TYPES:
                    values = _group_values(frame, model, float(eps_px), f"f1_{event_type}_{scoring}")
                    mean = float(np.mean(values))
                    base = baselines[(scoring, event_type)]
                    row[f"f1_{event_type}_{scoring}"] = mean
                    row[f"rel_f1_{event_type}_{scoring}"] = (
                        float("nan") if not np.isfinite(base) or base == 0.0 else mean / base
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame, models: list[str]) -> None:
    print(f"\n{N_TRIALS} trials at each eps, seeds {SEEDS[0]}..{SEEDS[-1]}")
    for model in models:
        print(f"\n{model}")
        for scoring in SCORING_KINDS:
            print(
                f"  {scoring:<10}  {'eps':>5}  "
                f"{'F1z':>7} {'F1d':>7} {'F1p':>7}  "
                f"{'relz':>7} {'reld':>7} {'relp':>7}"
            )
            for eps_px in EPS_PX:
                row = summary.loc[
                    (summary["model"] == model) & np.isclose(summary["eps_px"], eps_px)
                ].iloc[0]
                print(
                    f"  {'':<10}  {eps_px:5g}  "
                    f"{row[f'f1_zone_{scoring}']:7.3f} "
                    f"{row[f'f1_dwell_{scoring}']:7.3f} "
                    f"{row[f'f1_proximity_{scoring}']:7.3f}  "
                    f"{row[f'rel_f1_zone_{scoring}']:7.3f} "
                    f"{row[f'rel_f1_dwell_{scoring}']:7.3f} "
                    f"{row[f'rel_f1_proximity_{scoring}']:7.3f}"
                )


def _sanity_check_eps0(frame: pd.DataFrame, reference: pd.DataFrame, models: list[str]) -> bool:
    """eps=0 F1 must match events_on_tracks_C1.csv (saved to 4 decimals)."""
    print("\nSanity check: eps=0 F1 vs events_on_tracks_C1.csv")
    ok = True
    for model in models:
        for scoring in SCORING_KINDS:
            ref = reference.loc[(reference["model"] == model) & (reference["scoring"] == scoring)]
            if ref.empty:
                print(f"  FAIL  {model} {scoring}: missing reference row")
                ok = False
                continue
            ref_row = ref.iloc[0]
            for event_type in EVENT_TYPES:
                values = _group_values(frame, model, 0.0, f"f1_{event_type}_{scoring}")
                # All seeds at eps=0 share the same H, so all trials must agree.
                if len(values) == 0 or not np.all(
                    (values == values[0]) | (np.isnan(values) & np.isnan(values[0]))
                ):
                    print(f"  FAIL  {model} {scoring} {event_type}: eps=0 trials disagree")
                    ok = False
                    continue
                # events_on_tracks_C1.csv was written with float_format="%.4f".
                got = float(f"{values[0]:.4f}")
                expected = float(f"{float(ref_row[f'f1_{event_type}']):.4f}")
                if got != expected and not (np.isnan(got) and np.isnan(expected)):
                    print(
                        f"  FAIL  {model} {scoring} {event_type}: "
                        f"got {got:.4f}, expected {expected:.4f}"
                    )
                    ok = False
                else:
                    print(f"  PASS  {model} {scoring} {event_type}: {got:.4f}")
    print("Sanity check:", "PASS" if ok else "FAIL")
    return ok


def _gt_reference_curve(gt_frame: pd.DataFrame, event_type: str) -> np.ndarray:
    means = []
    for eps_px in EPS_PX:
        values = gt_frame.loc[np.isclose(gt_frame["eps_px"], eps_px), f"f1_{event_type}"].to_numpy(
            dtype=float
        )
        means.append(float(np.mean(values)))
    return np.asarray(means, dtype=float)


def _plot_f1_vs_eps(
    summary: pd.DataFrame,
    models: list[str],
    gt_frame: pd.DataFrame,
    path: Path,
    relative: bool,
) -> None:
    """Anonymous mean F1 (or relative F1) vs eps; GT-track sweep as dashed ref."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True)
    prefix = "rel_f1" if relative else "f1"
    ylabel = "relative F1" if relative else "F1"
    title = (
        "C1 relative anonymous F1 vs calibration noise (predicted tracks)"
        if relative
        else "C1 anonymous F1 vs calibration noise (predicted tracks)"
    )
    for ax, event_type in zip(axes, EVENT_TYPES):
        for model in models:
            ys = []
            for eps_px in EPS_PX:
                row = summary.loc[
                    (summary["model"] == model) & np.isclose(summary["eps_px"], eps_px)
                ].iloc[0]
                ys.append(float(row[f"{prefix}_{event_type}_anonymous"]))
            color = MODEL_COLORS.get(model, "0.3")
            ax.plot(EPS_PX, ys, marker="o", color=color, label=model, linewidth=1.8)
        gt_means = _gt_reference_curve(gt_frame, event_type)
        if relative:
            base = gt_means[0]
            gt_ys = gt_means / base if base else np.full_like(gt_means, np.nan)
        else:
            gt_ys = gt_means
        ax.plot(
            EPS_PX,
            gt_ys,
            color=GT_REF_COLOR,
            linestyle="--",
            linewidth=1.6,
            label="gt_tracks (ref)",
        )
        ax.set_title(event_type)
        ax.set_xlabel("calibration noise (px)")
        ax.set_xticks(list(EPS_PX))
        ax.set_xlim(-0.4, 10.4)
        ax.grid(True, alpha=0.3)
        if relative:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, y=1.14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def _column_order(columns: list[str]) -> list[str]:
    ordered = ["model", "eps_px", "seed"]
    for scoring in SCORING_KINDS:
        for metric in ("f1", "precision", "recall"):
            ordered.extend(
                f"{metric}_{event_type}_{scoring}"
                for event_type in EVENT_TYPES
                if f"{metric}_{event_type}_{scoring}" in columns
            )
    ordered.extend(column for column in columns if column not in ordered)
    return ordered


def _load_models(tracks_dir: Path, cam_id: str, annotations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Oracle feet plus the four default track files, idmap already applied."""
    model_feet: dict[str, pd.DataFrame] = {ORACLE_MODEL: prepare_oracle_feet(annotations)}
    print(f"{ORACLE_MODEL}: {len(model_feet[ORACLE_MODEL])} foot pixels")
    files = list(iter_track_files(tracks_dir, cam_id))
    if not files:
        raise FileNotFoundError(f"No tracks_{cam_id}_*.csv in {tracks_dir}")
    for path in files:
        name = path.stem.removeprefix(f"tracks_{cam_id}_")
        idmap_path = tracks_dir / f"idmap_{cam_id}_{name}.csv"
        if not idmap_path.is_file():
            raise FileNotFoundError(f"Missing {idmap_path.name}. Run scripts/check_tracks.py first.")
        feet, dropped = prepare_model_feet(pd.read_csv(path), pd.read_csv(idmap_path))
        model_feet[name] = feet
        print(f"{name}: {len(feet)} foot pixels (dropped {dropped} duplicate person rows)")
    return model_feet


def main() -> None:
    args = _parse_args()
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = normalize_cam_id(str(cfg["camera"]))
    margin = float(DEFAULT_BOUNDARY_MARGIN_M)
    output_dir = project_root() / "outputs"
    tracks_dir = project_root() / "results" / "tracks"

    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    gt_events = pd.read_csv(output_dir / f"gt_events_{cam_id}.csv")
    prox_path = output_dir / f"gt_events_{cam_id}_proximity_inner.csv"
    if not prox_path.is_file():
        raise FileNotFoundError(
            f"Missing {prox_path.name}. Run scripts/run_events_on_tracks.py first."
        )
    gt_proximity = pd.read_csv(prox_path)
    gt_world = _eot.positions_from_gt_world(annotations)
    gt_proximity_tracks = _eot.filter_inner_area(gt_world, margin)

    # Warm the wide-layout cache before timing a trial.
    world_pts, exact_pixels = choose_calibration_points(cam_id, layout="wide")
    model_feet = _load_models(tracks_dir, cam_id, annotations)
    models = list(model_feet.keys())

    table_path = output_dir / f"sweep_{cam_id}_pred_tracks.csv"
    n_jobs = len(_jobs(models))
    if args.force:
        if table_path.is_file():
            table_path.unlink()
            print(
                f"FORCE: deleted checkpoint {table_path.name}; "
                f"recomputing all {n_jobs} trials from scratch."
            )
        else:
            print(f"FORCE: no checkpoint present; computing all {n_jobs} trials from scratch.")
        completed = {}
    else:
        completed = _load_completed(table_path)
        if completed:
            print(
                f"RESUME: loading {table_path.name} "
                f"({len(completed)} of {n_jobs} computed trials already on disk)."
            )
        else:
            print(f"COMPUTE: no checkpoint at {table_path.name}; running all {n_jobs} trials.")

    timed_model = models[1] if len(models) > 1 else models[0]
    timed_key = (timed_model, 5.0, 0)
    if timed_key in completed:
        timed = completed[timed_key]
        trial_seconds = PARALLEL_IF_SLOWER_THAN_S + 0.1
        print(f"Reusing timed trial {timed_key} from checkpoint.")
    else:
        started = time.perf_counter()
        timed = run_trial_tracks(
            timed_model,
            5.0,
            0,
            model_feet[timed_model],
            world_pts,
            exact_pixels,
            gt_events,
            gt_proximity,
            gt_proximity_tracks,
            cfg,
            margin,
        )
        trial_seconds = time.perf_counter() - started

    results = _run_sweep(
        models,
        cam_id,
        cfg,
        margin,
        world_pts,
        exact_pixels,
        gt_events,
        gt_proximity,
        gt_proximity_tracks,
        model_feet,
        trial_seconds,
        timed,
        table_path,
        completed,
    )
    frame = pd.DataFrame(_expand_eps0_rows(results))
    frame = frame.sort_values(["model", "eps_px", "seed"], kind="mergesort").reset_index(drop=True)
    frame = frame[_column_order(list(frame.columns))]
    frame.to_csv(table_path, index=False, float_format="%.6f")
    print(f"Saved {table_path.name} ({len(frame)} rows)")

    summary = _mean_f1_table(frame, models)
    _print_summary(summary, models)

    reference = pd.read_csv(output_dir / f"events_on_tracks_{cam_id}.csv")
    _sanity_check_eps0(frame, reference, models)

    gt_sweep_path = output_dir / f"sweep_{cam_id}_gt_tracks.csv"
    if not gt_sweep_path.is_file():
        raise FileNotFoundError(f"Missing {gt_sweep_path.name} for the dashed reference curves.")
    gt_frame = pd.read_csv(gt_sweep_path)
    _plot_f1_vs_eps(
        summary,
        models,
        gt_frame,
        output_dir / f"f1_vs_eps_pred_tracks_{cam_id}.png",
        relative=False,
    )
    _plot_f1_vs_eps(
        summary,
        models,
        gt_frame,
        output_dir / f"rel_f1_vs_eps_pred_tracks_{cam_id}.png",
        relative=True,
    )


if __name__ == "__main__":
    main()
