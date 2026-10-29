"""Tests for polygon distance and the shared event state machine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import project_root
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    hysteresis_intervals,
    load_event_config,
    signed_distance_to_polygon,
)

# Same clock as the annotations: one frame is 1/fps seconds.
_EVENT_CONFIG = load_event_config(project_root() / "configs" / "events.yaml")
FRAME_DT = 1.0 / float(_EVENT_CONFIG["fps"])

# 2 m x 2 m square. The center is 1 m from every edge.
SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def test_center_of_square_is_positive_one() -> None:
    """A point at the center of a 2x2 square is 1 m inside."""
    distance = signed_distance_to_polygon(np.array([[1.0, 1.0]]), SQUARE)
    assert distance.shape == (1,)
    assert np.isclose(distance[0], 1.0)


def test_point_one_meter_outside_edge_is_negative_one() -> None:
    """A point 1 m outside an edge is 1 m outside."""
    distance = signed_distance_to_polygon(np.array([[1.0, -1.0]]), SQUARE)
    assert np.isclose(distance[0], -1.0)


def test_point_on_edge_is_zero() -> None:
    """A point on an edge has signed distance 0."""
    distance = signed_distance_to_polygon(np.array([[1.0, 0.0]]), SQUARE)
    assert np.isclose(distance[0], 0.0)


def _intervals(
    times: np.ndarray,
    margins: np.ndarray,
    h: float,
    debounce_frames: int,
    max_gap_frames: int,
    use_hysteresis: bool = True,
) -> list[tuple[float, float]]:
    """Call the state machine with the config frame step, 1/fps."""
    return hysteresis_intervals(
        times,
        margins,
        h=h,
        debounce_frames=debounce_frames,
        max_gap_frames=max_gap_frames,
        use_hysteresis=use_hysteresis,
        frame_dt=FRAME_DT,
    )


def _seconds(frame_numbers: np.ndarray) -> np.ndarray:
    """Convert frame indices to time_sec using the config frame step."""
    return np.asarray(frame_numbers, dtype=float) * FRAME_DT


def test_steady_margin_opens_one_event() -> None:
    """A margin that climbs from -1 to +1 and stays there opens one event."""
    ramp = np.linspace(-1.0, 1.0, 11)
    margins = np.concatenate([ramp, np.ones(5)])
    times = _seconds(np.arange(len(margins)))
    events = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    assert len(events) == 1
    assert events[0][1] == times[-1]
    assert events[0][0] < events[0][1]


def test_hysteresis_ignores_small_jitter() -> None:
    """Jitter inside the band opens nothing; naive thresholding opens many events."""
    margins = np.tile([-0.05, 0.05], 20)
    times = _seconds(np.arange(len(margins)))
    with_band = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    naive = _intervals(
        times,
        margins,
        h=0.1,
        debounce_frames=2,
        max_gap_frames=1,
        use_hysteresis=False,
    )
    assert with_band == []
    assert len(naive) == 20


def test_single_positive_frame_does_not_open_when_debounce_is_two() -> None:
    """One positive frame is not enough when two consecutive frames are required."""
    times = _seconds(np.arange(4))
    margins = np.array([-1.0, 1.0, -1.0, -1.0])
    events = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    assert events == []


def test_allowed_one_frame_gap_keeps_one_event() -> None:
    """A one-frame hole is allowed and does not split an event that is already open."""
    # Frames 0, 1, 2, then 4, 5. Frame 3 is missing.
    times = _seconds(np.array([0, 1, 2, 4, 5]))
    margins = np.ones(5)
    events = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    assert events == [(0.0, 5.0 * FRAME_DT)]


def test_long_gap_splits_the_event() -> None:
    """Three missing frames are beyond the allowed gap, so the track forms two events."""
    # Frames 0, 1, 2, then 6, 7. Frames 3, 4, 5 are missing.
    times = _seconds(np.array([0, 1, 2, 6, 7]))
    margins = np.ones(5)
    events = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    assert events == [(0.0, 2.0 * FRAME_DT), (6.0 * FRAME_DT, 7.0 * FRAME_DT)]


def test_allowed_gap_interrupts_an_unfinished_debounce() -> None:
    """+, missing, + does not open an event when two consecutive frames are required.

    The first positive sample starts a debounce run of length 1. The missing
    frame is within max_gap_frames, so it would not close an event that was
    already open, but it is not itself a positive margin. The run resets.
    The next positive sample starts again at length 1, still short of
    debounce_frames=2, so no event opens.
    """
    # Frame 0 positive, frame 1 missing, frame 2 positive.
    times = _seconds(np.array([0, 2]))
    margins = np.array([1.0, 1.0])
    events = _intervals(times, margins, h=0.1, debounce_frames=2, max_gap_frames=1)
    assert events == []


def _track(times: np.ndarray, xy: np.ndarray, person_id: int = 1) -> pd.DataFrame:
    """Build a one-person track table."""
    return pd.DataFrame(
        {
            "time_sec": times,
            "person_id": person_id,
            "X_m": xy[:, 0],
            "Y_m": xy[:, 1],
        }
    )


def test_walk_through_zone_is_one_event_and_not_a_dwell() -> None:
    """A straight crossing shorter than dwell_seconds is one zone event and no dwell."""
    xs = np.arange(-1.0, 3.0 + 1e-9, 0.5)
    times = np.arange(len(xs)) * FRAME_DT
    xy = np.column_stack([xs, np.ones(len(xs))])
    zones = [{"id": "box", "polygon": SQUARE}]
    zone_events = detect_zone_events(_track(times, xy), zones, _EVENT_CONFIG)
    dwell_events = detect_dwell_events(zone_events, _EVENT_CONFIG)
    assert len(zone_events) == 1
    assert zone_events.iloc[0]["event_type"] == "zone"
    assert zone_events.iloc[0]["zone_id"] == "box"
    assert float(zone_events.iloc[0]["duration"]) < float(_EVENT_CONFIG["dwell_seconds"])
    assert dwell_events.empty


def test_five_second_stay_dwells_after_the_threshold() -> None:
    """Staying 5 s opens one zone event and a dwell alarm dwell_seconds after entry."""
    dwell_seconds = float(_EVENT_CONFIG["dwell_seconds"])
    times = np.arange(0.0, 5.0 + 1e-9, FRAME_DT)
    xy = np.tile(np.array([[1.0, 1.0]]), (len(times), 1))
    zones = [{"id": "box", "polygon": SQUARE}]
    zone_events = detect_zone_events(_track(times, xy), zones, _EVENT_CONFIG)
    dwell_events = detect_dwell_events(zone_events, _EVENT_CONFIG)
    assert len(zone_events) == 1
    assert np.isclose(zone_events.iloc[0]["duration"], 5.0)
    assert len(dwell_events) == 1
    assert dwell_events.iloc[0]["event_type"] == "dwell"
    assert np.isclose(
        dwell_events.iloc[0]["start_time"],
        zone_events.iloc[0]["start_time"] + dwell_seconds,
    )
    assert np.isclose(dwell_events.iloc[0]["end_time"], zone_events.iloc[0]["end_time"])
    assert np.isclose(dwell_events.iloc[0]["duration"], 5.0 - dwell_seconds)


def test_boundary_jitter_does_not_chatter() -> None:
    """Small motion around the boundary opens at most one zone event."""
    rng = np.random.default_rng(0)
    times = np.arange(40) * FRAME_DT
    jitter = rng.uniform(-0.05, 0.05, size=len(times))
    xy = np.column_stack([np.ones(len(times)), jitter])
    zones = [{"id": "box", "polygon": SQUARE}]
    zone_events = detect_zone_events(_track(times, xy), zones, _EVENT_CONFIG)
    assert len(zone_events) <= 1


def _pair_track(times: np.ndarray, xy_a: np.ndarray, xy_b: np.ndarray) -> pd.DataFrame:
    """Build tracks for person 2 and person 1, with the higher id listed first."""
    first = pd.DataFrame(
        {"time_sec": times, "person_id": 2, "X_m": xy_b[:, 0], "Y_m": xy_b[:, 1]}
    )
    second = pd.DataFrame(
        {"time_sec": times, "person_id": 1, "X_m": xy_a[:, 0], "Y_m": xy_a[:, 1]}
    )
    return pd.concat([first, second], ignore_index=True)


def test_approach_and_separate_is_one_proximity_event() -> None:
    """Two people who meet and then separate produce one proximity event."""
    positions = np.arange(4.0, -4.0 - 1e-9, -0.25)
    times = np.arange(len(positions)) * FRAME_DT
    origin = np.zeros((len(positions), 2))
    walker = np.column_stack([positions, np.zeros(len(positions))])
    events = detect_proximity_events(_pair_track(times, origin, walker), _EVENT_CONFIG)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["event_type"] == "proximity"
    assert row["person_a"] < row["person_b"]
    assert float(row["min_distance"]) < float(_EVENT_CONFIG["proximity_threshold_m"])


def test_threshold_jitter_does_not_flicker() -> None:
    """Separation that jitters around the threshold opens at most one event."""
    rng = np.random.default_rng(0)
    times = np.arange(40) * FRAME_DT
    separation = float(_EVENT_CONFIG["proximity_threshold_m"]) + rng.uniform(-0.05, 0.05, len(times))
    origin = np.zeros((len(times), 2))
    other = np.column_stack([separation, np.zeros(len(times))])
    events = detect_proximity_events(_pair_track(times, origin, other), _EVENT_CONFIG)
    assert len(events) <= 1


def test_people_three_meters_apart_have_no_proximity_event() -> None:
    """A pair that stays 3 m apart never enters the proximity threshold."""
    times = np.arange(10) * FRAME_DT
    origin = np.zeros((len(times), 2))
    other = np.column_stack([np.full(len(times), 3.0), np.zeros(len(times))])
    events = detect_proximity_events(_pair_track(times, origin, other), _EVENT_CONFIG)
    assert len(events) == 0


def test_prefilter_uses_each_tracks_df_separately() -> None:
    """A far pair is not an event, and the same people moved close are, on a second call.

    The pre-filter must be computed from the positions passed in. A pair that
    is 3 m apart in one table must not suppress the event when a later table
    places those people 1.2 m apart.
    """
    times = np.arange(10) * FRAME_DT
    origin = np.zeros((len(times), 2))
    far = np.column_stack([np.full(len(times), 3.0), np.zeros(len(times))])
    close = np.column_stack([np.full(len(times), 1.2), np.zeros(len(times))])
    far_events = detect_proximity_events(_pair_track(times, origin, far), _EVENT_CONFIG)
    close_events = detect_proximity_events(_pair_track(times, origin, close), _EVENT_CONFIG)
    assert len(far_events) == 0
    assert len(close_events) == 1
    assert close_events.iloc[0]["person_a"] < close_events.iloc[0]["person_b"]
    assert np.isclose(close_events.iloc[0]["min_distance"], 1.2)
