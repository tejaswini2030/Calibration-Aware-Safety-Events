"""Tests for temporal matching and event scores."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.data import project_root
from src.evaluation import (
    covered_interval,
    match_events,
    match_events_anonymous,
    mean_pair_midpoint,
    score_events,
    score_events_anonymous,
)
from src.events import load_event_config

_EVENT_CONFIG = load_event_config(project_root() / "configs" / "events.yaml")
FRAME_DT = 1.0 / float(_EVENT_CONFIG["fps"])


def _event(event_type: str, **overrides) -> dict:
    """One event row. Duration is end_time - start_time, which is 0 for one frame."""
    row = {
        "event_type": event_type,
        "zone_id": None if event_type == "proximity" else "near",
        "person_id": None if event_type == "proximity" else 1,
        "person_a": 1 if event_type == "proximity" else None,
        "person_b": 2 if event_type == "proximity" else None,
        "start_time": 0.0,
        "end_time": 2.0,
    }
    row.update(overrides)
    row["duration"] = row["end_time"] - row["start_time"]
    return row


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build an event table."""
    return pd.DataFrame(rows)


def _total(scores: pd.DataFrame, event_type: str) -> pd.Series:
    """The type-level score row."""
    rows = scores.loc[(scores["event_type"] == event_type) & (scores["zone_id"] == "all")]
    assert len(rows) == 1
    return rows.iloc[0]


def test_identical_events_score_perfectly() -> None:
    """The same zone, dwell, and proximity events score precision, recall, and F1 of 1."""
    events = _frame(
        [
            _event("zone"),
            _event("dwell", start_time=2.0, end_time=5.0),
            _event("proximity", start_time=1.0, end_time=3.0),
        ]
    )
    scores, matches = score_events(events, events, _EVENT_CONFIG)
    totals = scores.loc[scores["zone_id"] == "all"]
    assert set(totals["event_type"]) == {"zone", "dwell", "proximity"}
    for _, row in totals.iterrows():
        assert row["tp"] == 1
        assert row["fp"] == 0
        assert row["fn"] == 0
        assert row["precision"] == 1.0
        assert row["recall"] == 1.0
        assert row["f1"] == 1.0
    assert len(matches) == 3
    assert np.allclose(matches["start_delay"], 0.0)
    assert np.allclose(matches["temporal_iou"], 1.0)


def test_one_frame_event_matches_itself() -> None:
    """A one-frame event has stored duration 0 but still matches an identical prediction."""
    events = _frame([_event("zone", start_time=1.0, end_time=1.0)])
    assert events.iloc[0]["end_time"] - events.iloc[0]["start_time"] == 0.0
    lo, hi = covered_interval(1.0, 1.0, FRAME_DT)
    assert hi - lo == FRAME_DT

    pairs, unmatched_gt, unmatched_pred = match_events(
        events, events, "zone", frame_dt=FRAME_DT
    )
    assert len(pairs) == 1
    assert pairs[0]["temporal_iou"] == 1.0
    assert pairs[0]["start_delay"] == 0.0
    assert unmatched_gt.empty
    assert unmatched_pred.empty

    scores, matches = score_events(events, events, _EVENT_CONFIG)
    row = _total(scores, "zone")
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert len(matches) == 1


def test_wrong_person_is_false_positive_and_false_negative() -> None:
    """A prediction for a different person does not match."""
    gt = _frame([_event("zone", person_id=1)])
    pred = _frame([_event("zone", person_id=2)])
    scores, matches = score_events(gt, pred, _EVENT_CONFIG)
    row = _total(scores, "zone")
    assert row["tp"] == 0
    assert row["fp"] == 1
    assert row["fn"] == 1
    assert matches.empty
    near = scores.loc[(scores["event_type"] == "zone") & (scores["zone_id"] == "near")].iloc[0]
    assert near["tp"] == 0
    assert near["fp"] == 1
    assert near["fn"] == 1


def test_two_predictions_for_one_gt_leave_one_false_positive() -> None:
    """The higher-IoU prediction takes the only match; the other is a false positive."""
    gt = _frame([_event("zone", start_time=0.0, end_time=2.0)])
    pred = _frame(
        [
            _event("zone", start_time=2.0, end_time=2.0),
            _event("zone", start_time=0.5, end_time=2.0),
        ]
    )
    scores, matches = score_events(gt, pred, _EVENT_CONFIG)
    row = _total(scores, "zone")
    assert row["tp"] == 1
    assert row["fp"] == 1
    assert row["fn"] == 0
    assert len(matches) == 1
    assert matches.iloc[0]["pred_index"] == 1
    assert np.isclose(matches.iloc[0]["start_delay"], 0.5)


def test_empty_predictions_have_nan_precision() -> None:
    """No predictions leaves recall at 0 and precision undefined."""
    gt = _frame([_event("zone")])
    pred = pd.DataFrame(columns=gt.columns)
    scores, matches = score_events(gt, pred, _EVENT_CONFIG)
    row = _total(scores, "zone")
    assert row["tp"] == 0
    assert row["fp"] == 0
    assert row["fn"] == 1
    assert math.isnan(row["precision"])
    assert row["recall"] == 0.0
    assert math.isnan(row["f1"])
    assert matches.empty


def test_no_events_reports_nan_f1() -> None:
    """A type with no ground truth and no predictions has F1 NaN, not 0 or 1."""
    empty = pd.DataFrame(columns=["event_type", "zone_id", "person_id", "start_time", "end_time"])
    scores, matches = score_events(empty, empty, _EVENT_CONFIG)
    assert matches.empty
    assert scores["f1"].isna().all()
    assert not scores["f1"].isin([0.0, 1.0]).any()


def test_anonymous_zone_matches_across_person_ids() -> None:
    """Anonymous zone matching ignores person_id and pairs by zone + temporal IoU."""
    gt = _frame([_event("zone", person_id=1, start_time=0.0, end_time=2.0)])
    pred = _frame([_event("zone", person_id=99, start_time=0.0, end_time=2.0)])
    scores_id, matches_id = score_events(gt, pred, _EVENT_CONFIG)
    assert _total(scores_id, "zone")["tp"] == 0
    assert matches_id.empty

    scores, matches = score_events_anonymous(gt, pred, _EVENT_CONFIG)
    row = _total(scores, "zone")
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0
    assert row["f1"] == 1.0
    assert len(matches) == 1


def test_anonymous_zone_greedy_highest_iou() -> None:
    """When two preds overlap one GT zone event, the higher-IoU pred wins."""
    gt = _frame([_event("zone", person_id=1, start_time=0.0, end_time=2.0)])
    pred = _frame(
        [
            _event("zone", person_id=2, start_time=1.5, end_time=2.0),
            _event("zone", person_id=3, start_time=0.0, end_time=2.0),
        ]
    )
    pairs, unmatched_gt, unmatched_pred = match_events_anonymous(
        gt, pred, "zone", frame_dt=FRAME_DT
    )
    assert len(pairs) == 1
    assert pairs[0]["pred_index"] == 1
    assert unmatched_gt.empty
    assert len(unmatched_pred) == 1


def test_anonymous_proximity_requires_nearby_midpoints() -> None:
    """Proximity anonymous match needs overlapping time and midpoints within 1 m."""
    times = np.array([0.0, FRAME_DT, 2 * FRAME_DT], dtype=float)
    gt_tracks = pd.DataFrame(
        {
            "time_sec": np.tile(times, 2),
            "person_id": np.repeat([1, 2], len(times)),
            "X_m": np.concatenate([np.zeros(len(times)), np.full(len(times), 0.5)]),
            "Y_m": np.zeros(2 * len(times)),
        }
    )
    # Predicted pair shifted by 0.4 m (OK) vs 2.0 m (too far).
    pred_near = gt_tracks.copy()
    pred_near["person_id"] = pred_near["person_id"].map({1: 10, 2: 20})
    pred_near["X_m"] = pred_near["X_m"] + 0.4
    pred_far = pred_near.copy()
    pred_far["X_m"] = pred_far["X_m"] + 1.6

    gt = _frame([_event("proximity", start_time=0.0, end_time=2 * FRAME_DT)])
    pred = _frame(
        [
            _event("proximity", person_a=10, person_b=20, start_time=0.0, end_time=2 * FRAME_DT),
        ]
    )
    mid_gt = mean_pair_midpoint(gt_tracks, 1, 2, 0.0, 3 * FRAME_DT)
    mid_near = mean_pair_midpoint(pred_near, 10, 20, 0.0, 3 * FRAME_DT)
    assert mid_gt is not None and mid_near is not None
    assert float(np.linalg.norm(mid_gt - mid_near)) <= 1.0

    scores_near, matches_near = score_events_anonymous(
        gt, pred, _EVENT_CONFIG, gt_tracks=gt_tracks, pred_tracks=pred_near
    )
    assert _total(scores_near, "proximity")["tp"] == 1
    assert len(matches_near) == 1

    scores_far, matches_far = score_events_anonymous(
        gt, pred, _EVENT_CONFIG, gt_tracks=gt_tracks, pred_tracks=pred_far
    )
    assert _total(scores_far, "proximity")["tp"] == 0
    assert matches_far.empty
