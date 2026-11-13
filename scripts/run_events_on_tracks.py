#!/usr/bin/env python3
"""Run zone / dwell / proximity detection on C1 tracks under perfect calibration.

Uses H* for foot projection. Zone and dwell are scored against
outputs/gt_events_C1.csv. Proximity uses only feet inside the 1 m-inset
inner area on both the predicted side and a separately saved ground-truth
file outputs/gt_events_C1_proximity_inner.csv, so edge people do not create
fake proximity pairs.

Also runs an oracle that treats ground-truth boxes as tracks, so the cost of
the box bottom-center foot convention is visible without detection error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, normalize_cam_id, project_root, resolve_dataset_root
from src.evaluation import EVENT_TYPES, score_events, score_events_anonymous
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    load_event_config,
)
from src.tracking import (
    DEFAULT_BOUNDARY_MARGIN_M,
    drop_outside_ground_area,
    foot_points_world,
    in_ground_area,
    iter_track_files,
    relabel_tracks,
    validate_tracks,
)

EVENT_TRACK_COLUMNS = ["time_sec", "person_id", "X_m", "Y_m"]
LATE_DELAY_S = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--tracks-dir", default=str(project_root() / "results" / "tracks"))
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN_M,
        help="meters to shrink the ground area before proximity detection (default 1.0)",
    )
    return parser.parse_args()


def filter_inner_area(tracks: pd.DataFrame, margin_m: float = DEFAULT_BOUNDARY_MARGIN_M) -> pd.DataFrame:
    """Keep rows whose (X_m, Y_m) lie in the annotated area shrunk by margin_m."""
    if tracks.empty:
        return tracks.loc[:, [column for column in EVENT_TRACK_COLUMNS if column in tracks.columns]].copy()
    keep = in_ground_area(tracks["X_m"].to_numpy(), tracks["Y_m"].to_numpy(), margin_m)
    return tracks.loc[keep].reset_index(drop=True)


def detect_events_on_positions(
    tracks: pd.DataFrame,
    cfg: dict[str, Any],
    proximity_tracks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run zone, dwell, and proximity on event-format tracks.

    Zone and dwell always use `tracks`. Proximity uses `proximity_tracks` when
    given (the inner-area subset), otherwise the same table.
    """
    needed = [column for column in EVENT_TRACK_COLUMNS if column not in tracks.columns]
    if needed:
        raise ValueError(f"tracks missing columns for events: {needed}")
    zone_events = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell_events = detect_dwell_events(zone_events, cfg)
    prox_source = tracks if proximity_tracks is None else proximity_tracks
    proximity_events = detect_proximity_events(prox_source, cfg)
    combined = pd.concat([zone_events, dwell_events, proximity_events], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(["start_time", "event_type"], kind="mergesort").reset_index(drop=True)


def positions_from_tracks(
    tracks: pd.DataFrame,
    idmap: pd.DataFrame,
    cam_id: str,
) -> tuple[pd.DataFrame, int]:
    """Relabel tracks and project box bottoms with H* into event coordinates."""
    validate_tracks(tracks)
    relabeled, dropped = relabel_tracks(tracks, idmap)
    world = foot_points_world(relabeled, cam_id)
    on_ground, _outside = drop_outside_ground_area(world)
    return on_ground.loc[:, EVENT_TRACK_COLUMNS].copy(), dropped


def positions_from_gt_boxes(annotations: pd.DataFrame, cam_id: str) -> pd.DataFrame:
    """Treat visible ground-truth boxes as tracks and project their bottoms with H*."""
    visible = annotations.loc[annotations["visible"]].copy()
    if visible.empty:
        return pd.DataFrame(columns=EVENT_TRACK_COLUMNS)
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
    world = foot_points_world(boxes, cam_id)
    on_ground, _outside = drop_outside_ground_area(world)
    return on_ground.loc[:, EVENT_TRACK_COLUMNS].copy()


def positions_from_gt_world(annotations: pd.DataFrame) -> pd.DataFrame:
    """Visible annotated positions, already in meters (no box projection)."""
    visible = annotations.loc[annotations["visible"], EVENT_TRACK_COLUMNS]
    return visible.copy().reset_index(drop=True)


def build_proximity_gt_inner(
    positions: pd.DataFrame,
    cfg: dict[str, Any],
    margin_m: float,
) -> pd.DataFrame:
    """Proximity events on ground-truth feet restricted to the inner area."""
    inner = filter_inner_area(positions, margin_m)
    return detect_proximity_events(inner, cfg)


def _f1_all(scores: pd.DataFrame, event_type: str) -> float:
    row = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == "all")]
    if row.empty:
        return float("nan")
    return float(row["f1"].iloc[0])


def _delay_stats(matches: pd.DataFrame, event_type: str) -> dict[str, float]:
    """Mean, median, and share of matched start delays above LATE_DELAY_S."""
    if matches.empty:
        delays = np.array([], dtype=float)
    else:
        delays = matches.loc[matches["event_type"] == event_type, "start_delay"].to_numpy(dtype=float)
    if len(delays) == 0:
        return {
            f"start_delay_mean_{event_type}": float("nan"),
            f"start_delay_median_{event_type}": float("nan"),
            f"start_delay_frac_gt_{LATE_DELAY_S:g}s_{event_type}": float("nan"),
        }
    return {
        f"start_delay_mean_{event_type}": float(np.mean(delays)),
        f"start_delay_median_{event_type}": float(np.median(delays)),
        f"start_delay_frac_gt_{LATE_DELAY_S:g}s_{event_type}": float(np.mean(delays > LATE_DELAY_S)),
    }


def _split_gt_pred(
    pred_events: pd.DataFrame,
    gt_zone_dwell: pd.DataFrame,
    gt_proximity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_zone_dwell_only = gt_zone_dwell.loc[gt_zone_dwell["event_type"].isin(["zone", "dwell"])]
    pred_zone_dwell = pred_events.loc[pred_events["event_type"].isin(["zone", "dwell"])]
    pred_proximity = pred_events.loc[pred_events["event_type"] == "proximity"]
    return gt_zone_dwell_only, pred_zone_dwell, gt_proximity, pred_proximity


def _combine_type_scores(
    scores_zd: pd.DataFrame,
    matches_zd: pd.DataFrame,
    scores_pr: pd.DataFrame,
    matches_pr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.concat(
        [
            scores_zd.loc[scores_zd["event_type"].isin(["zone", "dwell"])],
            scores_pr.loc[scores_pr["event_type"] == "proximity"],
        ],
        ignore_index=True,
    )
    matches = pd.concat(
        [
            matches_zd.loc[matches_zd["event_type"].isin(["zone", "dwell"])] if len(matches_zd) else matches_zd,
            matches_pr.loc[matches_pr["event_type"] == "proximity"] if len(matches_pr) else matches_pr,
        ],
        ignore_index=True,
    )
    return scores, matches


def score_model_events(
    pred_events: pd.DataFrame,
    gt_zone_dwell: pd.DataFrame,
    gt_proximity: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Identity-based scores: zone/dwell vs full GT, proximity vs inner GT."""
    gt_zd, pred_zd, gt_pr, pred_pr = _split_gt_pred(pred_events, gt_zone_dwell, gt_proximity)
    scores_zd, matches_zd = score_events(gt_zd, pred_zd, cfg)
    scores_pr, matches_pr = score_events(gt_pr, pred_pr, cfg)
    scores, matches = _combine_type_scores(scores_zd, matches_zd, scores_pr, matches_pr)
    out: dict[str, float] = {"scoring": "identity"}
    for event_type in EVENT_TYPES:
        out[f"f1_{event_type}"] = _f1_all(scores, event_type)
        out.update(_delay_stats(matches, event_type))
        type_rows = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == "all")]
        if type_rows.empty:
            out[f"precision_{event_type}"] = float("nan")
            out[f"recall_{event_type}"] = float("nan")
        else:
            out[f"precision_{event_type}"] = float(type_rows["precision"].iloc[0])
            out[f"recall_{event_type}"] = float(type_rows["recall"].iloc[0])
        if event_type in {"zone", "dwell"}:
            for zone in cfg["zones"]:
                zone_id = str(zone["id"])
                zone_row = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == zone_id)]
                out[f"f1_{event_type}_{zone_id}"] = (
                    float(zone_row["f1"].iloc[0]) if len(zone_row) else float("nan")
                )
    return out


def score_model_events_anonymous(
    pred_events: pd.DataFrame,
    gt_zone_dwell: pd.DataFrame,
    gt_proximity: pd.DataFrame,
    pred_proximity_tracks: pd.DataFrame,
    gt_proximity_tracks: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Identity-free scores. Proximity also needs nearby pair midpoints."""
    gt_zd, pred_zd, gt_pr, pred_pr = _split_gt_pred(pred_events, gt_zone_dwell, gt_proximity)
    scores_zd, matches_zd = score_events_anonymous(gt_zd, pred_zd, cfg)
    scores_pr, matches_pr = score_events_anonymous(
        gt_pr,
        pred_pr,
        cfg,
        gt_tracks=gt_proximity_tracks,
        pred_tracks=pred_proximity_tracks,
    )
    scores, _matches = _combine_type_scores(scores_zd, matches_zd, scores_pr, matches_pr)
    out: dict[str, Any] = {"scoring": "anonymous"}
    for event_type in EVENT_TYPES:
        out[f"f1_{event_type}"] = _f1_all(scores, event_type)
        type_rows = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == "all")]
        if type_rows.empty:
            out[f"precision_{event_type}"] = float("nan")
            out[f"recall_{event_type}"] = float("nan")
        else:
            out[f"precision_{event_type}"] = float(type_rows["precision"].iloc[0])
            out[f"recall_{event_type}"] = float(type_rows["recall"].iloc[0])
    return out


def _load_idmap(tracks_dir: Path, cam_id: str, model: str) -> pd.DataFrame:
    path = tracks_dir / f"idmap_{cam_id}_{model}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}. Run scripts/check_tracks.py first.")
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    margin = float(args.boundary_margin)
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    if str(cfg["camera"]) != cam_id:
        raise ValueError(f"config camera is {cfg['camera']!r}, expected {cam_id!r}")

    tracks_dir = Path(args.tracks_dir)
    files = list(iter_track_files(tracks_dir, cam_id))
    if not files:
        raise FileNotFoundError(f"No tracks_{cam_id}_*.csv in {tracks_dir}")

    annotations = load_annotations(resolve_dataset_root(args.data_root), cam_id)
    gt_world = positions_from_gt_world(annotations)
    gt_events_path = project_root() / "outputs" / f"gt_events_{cam_id}.csv"
    gt_events = pd.read_csv(gt_events_path)
    gt_proximity_inner = build_proximity_gt_inner(gt_world, cfg, margin)
    prox_gt_path = project_root() / "outputs" / f"gt_events_{cam_id}_proximity_inner.csv"
    gt_proximity_inner.to_csv(prox_gt_path, index=False)
    print(
        f"Saved {prox_gt_path.name}: {len(gt_proximity_inner)} proximity events "
        f"on ground truth inside the {margin:g} m-inset area "
        f"(full GT proximity count was "
        f"{int((gt_events['event_type'] == 'proximity').sum())})"
    )

    identity_rows: list[dict[str, Any]] = []
    anonymous_rows: list[dict[str, Any]] = []
    gt_proximity_tracks = filter_inner_area(gt_world, margin)

    def _run_one(name: str, positions: pd.DataFrame, dropped: int | None = None) -> None:
        inner = filter_inner_area(positions, margin)
        events = detect_events_on_positions(positions, cfg, proximity_tracks=inner)
        identity = score_model_events(events, gt_events, gt_proximity_inner, cfg)
        anonymous = score_model_events_anonymous(
            events,
            gt_events,
            gt_proximity_inner,
            pred_proximity_tracks=inner,
            gt_proximity_tracks=gt_proximity_tracks,
            cfg=cfg,
        )
        identity_rows.append({"model": name, **identity})
        anonymous_rows.append({"model": name, **anonymous})
        extra = ""
        if dropped is not None:
            extra = f"dropped {dropped} duplicate person rows; "
        print(
            f"{name}: {extra}{len(positions)} feet, {len(inner)} inside inner area; "
            f"identity F1 zone/dwell/prox "
            f"{identity['f1_zone']:.3f}/{identity['f1_dwell']:.3f}/{identity['f1_proximity']:.3f}; "
            f"anonymous F1 "
            f"{anonymous['f1_zone']:.3f}/{anonymous['f1_dwell']:.3f}/{anonymous['f1_proximity']:.3f}"
        )

    # Oracle: ground-truth boxes projected by bottom-center.
    _run_one("oracle_gt_boxes", positions_from_gt_boxes(annotations, cam_id))

    for path in files:
        name = path.stem.removeprefix(f"tracks_{cam_id}_")
        tracks = pd.read_csv(path)
        idmap = _load_idmap(tracks_dir, cam_id, name)
        positions, dropped = positions_from_tracks(tracks, idmap, cam_id)
        _run_one(name, positions, dropped)

    identity = pd.DataFrame(identity_rows)
    anonymous = pd.DataFrame(anonymous_rows)
    identity_shown = [
        "model",
        "f1_zone",
        "f1_dwell",
        "f1_proximity",
        "start_delay_mean_zone",
        "start_delay_median_zone",
        "start_delay_frac_gt_2s_zone",
        "start_delay_mean_dwell",
        "start_delay_median_dwell",
        "start_delay_frac_gt_2s_dwell",
        "start_delay_mean_proximity",
        "start_delay_median_proximity",
        "start_delay_frac_gt_2s_proximity",
    ]
    anonymous_shown = [
        "model",
        "precision_zone",
        "recall_zone",
        "f1_zone",
        "precision_dwell",
        "recall_dwell",
        "f1_dwell",
        "precision_proximity",
        "recall_proximity",
        "f1_proximity",
    ]
    side_by_side = identity[["model", "f1_zone", "f1_dwell", "f1_proximity"]].rename(
        columns={
            "f1_zone": "F1z_id",
            "f1_dwell": "F1d_id",
            "f1_proximity": "F1p_id",
        }
    ).merge(
        anonymous[["model", "f1_zone", "f1_dwell", "f1_proximity"]].rename(
            columns={
                "f1_zone": "F1z_anon",
                "f1_dwell": "F1d_anon",
                "f1_proximity": "F1p_anon",
            }
        ),
        on="model",
    )

    print(f"\n{cam_id} identity-based scores (perfect H*, proximity inner margin {margin:g} m)")
    print(identity[identity_shown].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\n{cam_id} anonymous scores (zone/dwell by zone+time; proximity by time+midpoint ≤ 1 m)")
    print(anonymous[anonymous_shown].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\n{cam_id} F1 side by side (identity | anonymous)")
    print(side_by_side.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    saved = pd.concat([identity, anonymous], ignore_index=True)
    out = project_root() / "outputs" / f"events_on_tracks_{cam_id}.csv"
    saved.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out.name} ({len(identity)} identity + {len(anonymous)} anonymous rows)")


if __name__ == "__main__":
    main()
