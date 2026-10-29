"""Event-level matching and scores."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

EVENT_TYPES = ("zone", "dwell", "proximity")
SCORE_COLUMNS = ["event_type", "zone_id", "tp", "fp", "fn", "precision", "recall", "f1"]
MATCH_COLUMNS = [
    "event_type",
    "zone_id",
    "person_id",
    "person_a",
    "person_b",
    "gt_index",
    "pred_index",
    "gt_start",
    "gt_end",
    "pred_start",
    "pred_end",
    "start_delay",
    "temporal_iou",
]


def frame_dt_from_config(cfg: dict[str, Any]) -> float:
    """Return the frame step 1/fps from an event config."""
    fps = float(cfg["fps"])
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return 1.0 / fps


def covered_interval(start_time: float, end_time: float, frame_dt: float) -> tuple[float, float]:
    """Return the half-open interval of frames an event occupies.

    Stored events record the first and last annotated frame, so a one-frame
    event has start_time == end_time and a stored duration of 0. That frame
    still occupies time: the sample at t stands for [t, t + frame_dt).
    Measuring overlap on [start, end] would give one-frame events length 0,
    and an identical one-frame prediction would not count as a true positive.
    """
    if frame_dt <= 0:
        raise ValueError(f"frame_dt must be positive, got {frame_dt}")
    return float(start_time), float(end_time) + float(frame_dt)


def temporal_iou(
    gt_start: float,
    gt_end: float,
    pred_start: float,
    pred_end: float,
    frame_dt: float,
) -> float:
    """Return the temporal IoU of two events on frame-inclusive intervals."""
    gt_lo, gt_hi = covered_interval(gt_start, gt_end, frame_dt)
    pred_lo, pred_hi = covered_interval(pred_start, pred_end, frame_dt)
    intersection = min(gt_hi, pred_hi) - max(gt_lo, pred_lo)
    if intersection <= 0.0:
        return 0.0
    union = (gt_hi - gt_lo) + (pred_hi - pred_lo) - intersection
    return intersection / union


def _events_of_type(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Keep rows of one event type without changing their index."""
    if events is None or events.empty or "event_type" not in events.columns:
        columns = list(events.columns) if events is not None else []
        return pd.DataFrame(columns=columns)
    return events.loc[events["event_type"] == event_type]


def _identity_columns(event_type: str) -> tuple[str, ...]:
    """Columns that must agree before two events can match."""
    if event_type == "proximity":
        return ("person_a", "person_b")
    if event_type in ("zone", "dwell"):
        return ("zone_id", "person_id")
    raise ValueError(f"Unknown event type: {event_type}")


def _key_value(value: Any) -> Any:
    """Normalize an identity field so 1 and 1.0 are the same person."""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer():
        return int(number)
    return number


def _identity(row: pd.Series, columns: tuple[str, ...]) -> tuple[Any, ...] | None:
    """Build an identity key, or None when a required field is missing."""
    values = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            return None
        values.append(_key_value(value))
    return tuple(values)


def _require_columns(events: pd.DataFrame, columns: tuple[str, ...], event_type: str) -> None:
    """Raise if a non-empty event table is missing columns the matcher needs."""
    if events.empty:
        return
    missing = [column for column in columns if column not in events.columns]
    if missing:
        raise ValueError(f"{event_type} events are missing columns: {missing}")


def _empty_matches() -> pd.DataFrame:
    """Return a match table with the delay columns and no rows."""
    return pd.DataFrame(columns=MATCH_COLUMNS)


def match_events(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    event_type: str,
    frame_dt: float,
    min_iou: float = 0.0,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Match predictions to ground truth for one event type.

    Only events of this type are compared, and only when they share an
    identity: zone_id and person_id for zone and dwell, or (person_a, person_b)
    for proximity. The pair order is used as stored. A candidate must have
    temporal IoU strictly above min_iou, so min_iou=0 keeps any overlap.
    Matching is one-to-one and greedy by highest IoU. Equal IoU prefers the
    earlier ground-truth row, then the earlier prediction, which pairs each
    event with itself when the two tables are identical.
    """
    identity_columns = _identity_columns(event_type)
    required = ("start_time", "end_time", *identity_columns)
    gt = _events_of_type(gt_df, event_type)
    pred = _events_of_type(pred_df, event_type)
    _require_columns(gt, required, event_type)
    _require_columns(pred, required, event_type)
    if gt.empty or pred.empty:
        return [], gt, pred

    gt_keys = [_identity(row, identity_columns) for _, row in gt.iterrows()]
    pred_keys = [_identity(row, identity_columns) for _, row in pred.iterrows()]
    pred_positions: dict[tuple[Any, ...], list[int]] = {}
    for position, key in enumerate(pred_keys):
        if key is None:
            continue
        pred_positions.setdefault(key, []).append(position)

    gt_labels = list(gt.index)
    pred_labels = list(pred.index)
    gt_starts = gt["start_time"].to_numpy(dtype=float)
    gt_ends = gt["end_time"].to_numpy(dtype=float)
    pred_starts = pred["start_time"].to_numpy(dtype=float)
    pred_ends = pred["end_time"].to_numpy(dtype=float)

    candidates: list[tuple[float, int, int]] = []
    for gt_pos, key in enumerate(gt_keys):
        if key is None:
            continue
        for pred_pos in pred_positions.get(key, []):
            iou = temporal_iou(
                gt_starts[gt_pos],
                gt_ends[gt_pos],
                pred_starts[pred_pos],
                pred_ends[pred_pos],
                frame_dt,
            )
            if iou > min_iou:
                candidates.append((iou, gt_pos, pred_pos))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for iou, gt_pos, pred_pos in candidates:
        if gt_pos in matched_gt or pred_pos in matched_pred:
            continue
        matched_gt.add(gt_pos)
        matched_pred.add(pred_pos)
        gt_row = gt.iloc[gt_pos]
        pred_row = pred.iloc[pred_pos]
        pairs.append(
            {
                "event_type": event_type,
                "zone_id": gt_row["zone_id"] if "zone_id" in gt.columns else None,
                "person_id": gt_row["person_id"] if "person_id" in gt.columns else None,
                "person_a": gt_row["person_a"] if "person_a" in gt.columns else None,
                "person_b": gt_row["person_b"] if "person_b" in gt.columns else None,
                "gt_index": gt_labels[gt_pos],
                "pred_index": pred_labels[pred_pos],
                "gt_start": float(gt_starts[gt_pos]),
                "gt_end": float(gt_ends[gt_pos]),
                "pred_start": float(pred_starts[pred_pos]),
                "pred_end": float(pred_ends[pred_pos]),
                "start_delay": float(pred_starts[pred_pos] - gt_starts[gt_pos]),
                "temporal_iou": float(iou),
            }
        )

    unmatched_gt = gt.iloc[[pos for pos in range(len(gt)) if pos not in matched_gt]]
    unmatched_pred = pred.iloc[[pos for pos in range(len(pred)) if pos not in matched_pred]]
    return pairs, unmatched_gt, unmatched_pred


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, and F1. Undefined rates are NaN, not 0 or 1."""
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    elif precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _score_row(event_type: str, zone_id: str, tp: int, fp: int, fn: int) -> dict[str, Any]:
    """One score row for an event type or a single zone."""
    precision, recall, f1 = _metrics(tp, fp, fn)
    return {
        "event_type": event_type,
        "zone_id": zone_id,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _zone_ids(cfg: dict[str, Any], gt: pd.DataFrame, pred: pd.DataFrame) -> list[Any]:
    """Config zone order, then any extra zone ids present in the events."""
    ordered = [zone["id"] for zone in cfg.get("zones", [])]
    seen = set(ordered)
    extra: list[Any] = []
    for frame in (gt, pred):
        if frame.empty or "zone_id" not in frame.columns:
            continue
        for zone_id in frame["zone_id"].dropna().unique():
            if zone_id not in seen:
                seen.add(zone_id)
                extra.append(zone_id)
    return ordered + extra


def _counts_for_zone(
    pairs: list[dict[str, Any]],
    unmatched_gt: pd.DataFrame,
    unmatched_pred: pd.DataFrame,
    zone_id: str,
) -> tuple[int, int, int]:
    """TP, FP, and FN for one zone, or for the whole type when zone_id is 'all'."""
    if zone_id == "all":
        return len(pairs), len(unmatched_pred), len(unmatched_gt)
    tp = sum(1 for pair in pairs if pair["zone_id"] == zone_id)
    fn = 0 if unmatched_gt.empty else int((unmatched_gt["zone_id"] == zone_id).sum())
    fp = 0 if unmatched_pred.empty else int((unmatched_pred["zone_id"] == zone_id).sum())
    return tp, fp, fn


def score_events(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cfg: dict[str, Any],
    min_iou: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score predictions against ground truth.

    Returns a score table and the accepted matches. The table has one row per
    event type (zone_id 'all') and, for zone and dwell, one row per zone.
    Matches include start_delay = pred_start - gt_start for later alarm-delay
    measurement. A type or zone with no ground truth and no predictions has
    F1 NaN.
    """
    frame_dt = frame_dt_from_config(cfg)
    rows: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for event_type in EVENT_TYPES:
        pairs, unmatched_gt, unmatched_pred = match_events(
            gt_df,
            pred_df,
            event_type,
            frame_dt,
            min_iou=min_iou,
        )
        all_pairs.extend(pairs)
        tp, fp, fn = _counts_for_zone(pairs, unmatched_gt, unmatched_pred, "all")
        rows.append(_score_row(event_type, "all", tp, fp, fn))
        if event_type == "proximity":
            continue
        gt_type = _events_of_type(gt_df, event_type)
        pred_type = _events_of_type(pred_df, event_type)
        for zone_id in _zone_ids(cfg, gt_type, pred_type):
            tp, fp, fn = _counts_for_zone(pairs, unmatched_gt, unmatched_pred, zone_id)
            rows.append(_score_row(event_type, zone_id, tp, fp, fn))

    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS)
    if not all_pairs:
        return scores, _empty_matches()
    matches = pd.DataFrame(all_pairs)
    matches = matches.sort_values(
        ["event_type", "gt_start", "pred_start"],
        kind="mergesort",
    ).reset_index(drop=True)
    return scores, matches[MATCH_COLUMNS]
