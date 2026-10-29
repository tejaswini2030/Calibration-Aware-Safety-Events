"""Safety event logic."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as PolygonPath

_REQUIRED_KEYS = (
    "camera",
    "fps",
    "zones",
    "zone_hysteresis_m",
    "proximity_threshold_m",
    "proximity_hysteresis_m",
    "dwell_seconds",
    "debounce_frames",
    "max_gap_frames",
)


def load_event_config(path: str | Path) -> dict[str, Any]:
    """Load an event YAML file.

    Each zone polygon is returned as an (M, 2) array of world coordinates
    in meters.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Event config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Event config must be a mapping: {config_path}")
    missing = [key for key in _REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Event config is missing keys: {missing}")
    zones = []
    for zone in config["zones"]:
        if not {"id", "name", "polygon"} <= set(zone):
            raise ValueError(f"Zone is missing id, name, or polygon: {zone}")
        polygon = np.asarray(zone["polygon"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(f"Zone {zone['id']} polygon must have shape (M, 2), M >= 3")
        zones.append({**zone, "polygon": polygon})
    config["zones"] = zones
    return config


def _boundary_distances(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return the distance from each point to the nearest polygon edge."""
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    length2 = np.sum(segments * segments, axis=1)
    length2 = np.maximum(length2, 1e-24)
    relative = points[:, None, :] - starts[None, :, :]
    scale = np.sum(relative * segments[None, :, :], axis=2) / length2
    scale = np.clip(scale, 0.0, 1.0)
    closest = starts[None, :, :] + scale[:, :, None] * segments[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def signed_distance_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Signed distance in meters from each point to a polygon boundary.

    points is (N, 2) and polygon is (M, 2), both in meters. The result is
    (N,): positive inside, negative outside, and zero on the boundary.
    """
    pts = np.asarray(points, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"polygon must have shape (M, 2), M >= 3, got {poly.shape}")
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    distance = _boundary_distances(pts, poly)
    closed = np.vstack([poly, poly[0]])
    inside = PolygonPath(closed).contains_points(pts)
    signed = np.where(inside, distance, -distance)
    signed[np.isclose(distance, 0.0)] = 0.0
    return signed


def _infer_frame_dt(times: np.ndarray) -> float:
    """Infer a frame step from the smallest positive gap. Fallback only."""
    if len(times) < 2:
        return 1.0
    gaps = np.diff(times)
    positive = gaps[gaps > 0]
    if len(positive) == 0:
        raise ValueError("times must increase")
    return float(np.min(positive))


def _frame_step(times: np.ndarray, frame_dt: float | None) -> float:
    """Return the time step of one frame.

    Callers should pass 1/fps from the event config. Inference from the
    time stamps is only a fallback, and it warns, because a track whose
    shortest gap is already a hole would hide that hole.
    """
    if frame_dt is not None:
        if frame_dt <= 0:
            raise ValueError(f"frame_dt must be positive, got {frame_dt}")
        return float(frame_dt)
    inferred = _infer_frame_dt(times)
    warnings.warn(
        f"frame_dt was not passed; inferred {inferred:g} from the time stamps. "
        "Pass 1/fps from the event config instead.",
        stacklevel=3,
    )
    return inferred


def _missing_frames(previous: float, current: float, frame_dt: float) -> int:
    """Return how many frames are absent between two observations."""
    steps = int(round((current - previous) / frame_dt))
    return max(steps - 1, 0)


def hysteresis_intervals(
    times: np.ndarray,
    margins: np.ndarray,
    h: float,
    debounce_frames: int,
    max_gap_frames: int,
    use_hysteresis: bool = True,
    frame_dt: float | None = None,
) -> list[tuple[float, float]]:
    """Return (start_time, end_time) intervals for one entity.

    times and margins are sorted observations. Frames where the entity is
    missing are left out. A margin >= +h for debounce_frames consecutive
    frames turns the event on, starting at the first of those frames. A
    margin <= -h for that many consecutive frames turns it off; end_time is
    the last frame before that run. Margins strictly between -h and +h do
    not change the state. A gap longer than max_gap_frames closes an open
    event at the last seen frame. An event still open at the end closes at
    the last frame.

    use_hysteresis=False forces h=0 and debounce_frames=1. frame_dt should
    be 1/fps from the event config (0.5 s at 2 fps). If it is omitted, the
    smallest gap in times is used and a warning is issued.

    An allowed gap does not count as a positive or negative sample. It
    leaves an already open event running, but it resets a debounce run that
    has not yet reached debounce_frames, so +, missing, + does not open an
    event when debounce_frames is 2.
    """
    observed_times = np.asarray(times, dtype=np.float64).reshape(-1)
    observed_margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    if observed_times.shape != observed_margins.shape:
        raise ValueError("times and margins must have the same length")
    if len(observed_times) == 0:
        return []
    if np.any(np.diff(observed_times) <= 0):
        raise ValueError("times must be strictly increasing")
    if not use_hysteresis:
        h = 0.0
        debounce_frames = 1
    if h < 0:
        raise ValueError(f"h must be non-negative, got {h}")
    if debounce_frames < 1:
        raise ValueError(f"debounce_frames must be at least 1, got {debounce_frames}")
    if max_gap_frames < 0:
        raise ValueError(f"max_gap_frames must be non-negative, got {max_gap_frames}")

    step = _frame_step(observed_times, frame_dt)
    intervals: list[tuple[float, float]] = []
    active = False
    start_time: float | None = None
    on_count = 0
    off_count = 0
    on_first: float | None = None
    off_first_index: int | None = None

    for index, (time, margin) in enumerate(zip(observed_times, observed_margins)):
        if index > 0:
            missing = _missing_frames(float(observed_times[index - 1]), float(time), step)
            if missing > max_gap_frames:
                if active and start_time is not None:
                    intervals.append((start_time, float(observed_times[index - 1])))
                active = False
                start_time = None
                on_count = 0
                off_count = 0
                on_first = None
                off_first_index = None
            elif missing > 0:
                # The hole is short enough to keep an open event, but it is not
                # one of the consecutive samples a debounce run needs. +, missing,
                # + with debounce_frames=2 therefore never turns on.
                on_count = 0
                off_count = 0
                on_first = None
                off_first_index = None

        if margin >= h:
            off_count = 0
            off_first_index = None
            if on_count == 0:
                on_first = float(time)
            on_count += 1
            if not active and on_count >= debounce_frames and on_first is not None:
                active = True
                start_time = on_first
        elif margin <= -h:
            on_count = 0
            on_first = None
            if off_count == 0:
                off_first_index = index
            off_count += 1
            if active and off_count >= debounce_frames and off_first_index is not None:
                if off_first_index == 0 or start_time is None:
                    raise RuntimeError("an event turned off with no preceding frame")
                intervals.append((start_time, float(observed_times[off_first_index - 1])))
                active = False
                start_time = None
        else:
            on_count = 0
            off_count = 0
            on_first = None
            off_first_index = None

    if active and start_time is not None:
        intervals.append((start_time, float(observed_times[-1])))
    return intervals


EVENT_COLUMNS = ["event_type", "zone_id", "person_id", "start_time", "end_time", "duration"]


def _empty_events() -> pd.DataFrame:
    """Return an event table with the standard columns and no rows."""
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _frame_dt_from_config(cfg: dict[str, Any]) -> float:
    """Return the frame step 1/fps from an event config."""
    fps = float(cfg["fps"])
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return 1.0 / fps


def detect_zone_events(
    tracks_df: pd.DataFrame,
    zones: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Detect zone entry and exit for each person.

    tracks_df has time_sec, person_id, X_m, and Y_m. The margin passed to
    the state machine is the signed distance to that zone, in meters.
    """
    if tracks_df.empty or not zones:
        return _empty_events()
    frame_dt = _frame_dt_from_config(cfg)
    hysteresis_m = float(cfg["zone_hysteresis_m"])
    debounce_frames = int(cfg["debounce_frames"])
    max_gap_frames = int(cfg["max_gap_frames"])
    use_hysteresis = bool(cfg.get("use_hysteresis", True))

    rows: list[dict[str, Any]] = []
    for zone in zones:
        polygon = np.asarray(zone["polygon"], dtype=np.float64)
        for person_id, track in tracks_df.groupby("person_id", sort=False):
            ordered = track.sort_values("time_sec")
            times = ordered["time_sec"].to_numpy(dtype=np.float64)
            points = ordered[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
            margins = signed_distance_to_polygon(points, polygon)
            intervals = hysteresis_intervals(
                times,
                margins,
                h=hysteresis_m,
                debounce_frames=debounce_frames,
                max_gap_frames=max_gap_frames,
                use_hysteresis=use_hysteresis,
                frame_dt=frame_dt,
            )
            for start_time, end_time in intervals:
                rows.append(
                    {
                        "event_type": "zone",
                        "zone_id": zone["id"],
                        "person_id": person_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                    }
                )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def detect_dwell_events(zone_events: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Turn long zone visits into dwell alarms.

    A zone event that lasts at least dwell_seconds produces one dwell event.
    The alarm starts dwell_seconds after the zone entry and ends when the
    zone event ends.
    """
    if zone_events.empty:
        return _empty_events()
    dwell_seconds = float(cfg["dwell_seconds"])
    long_enough = zone_events.loc[zone_events["duration"] >= dwell_seconds].copy()
    if long_enough.empty:
        return _empty_events()
    long_enough["event_type"] = "dwell"
    long_enough["start_time"] = long_enough["start_time"] + dwell_seconds
    long_enough["duration"] = long_enough["end_time"] - long_enough["start_time"]
    return long_enough[EVENT_COLUMNS].reset_index(drop=True)


PROXIMITY_COLUMNS = [
    "event_type",
    "person_a",
    "person_b",
    "start_time",
    "end_time",
    "duration",
    "min_distance",
]


def _empty_proximity_events() -> pd.DataFrame:
    """Return a proximity table with the standard columns and no rows."""
    return pd.DataFrame(columns=PROXIMITY_COLUMNS)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return the (N, N) Euclidean distance matrix for points of shape (N, 2)."""
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def detect_proximity_events(tracks_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Detect when an unordered pair of people stays within the proximity threshold.

    For each frame, distances come from a numpy pairwise matrix of the people
    present in tracks_df. The state-machine margin is proximity_threshold_m
    minus that distance. Pairs that never come within threshold + hysteresis
    + 0.5 m on these positions are skipped. That pre-filter is recomputed
    from tracks_df on every call and is never cached from ground truth: under
    a perturbed homography, pairs that were far apart in ground truth can
    appear close, and those false positives must be detected. min_distance
    is the closest separation during the event.
    """
    if tracks_df.empty:
        return _empty_proximity_events()
    frame_dt = _frame_dt_from_config(cfg)
    threshold_m = float(cfg["proximity_threshold_m"])
    hysteresis_m = float(cfg["proximity_hysteresis_m"])
    debounce_frames = int(cfg["debounce_frames"])
    max_gap_frames = int(cfg["max_gap_frames"])
    use_hysteresis = bool(cfg.get("use_hysteresis", True))
    keep_within_m = threshold_m + hysteresis_m + 0.5

    samples: dict[tuple[Any, Any], list[tuple[float, float]]] = {}
    for time_sec, frame in tracks_df.groupby("time_sec", sort=True):
        present = frame.drop_duplicates("person_id", keep="first")
        if len(present) < 2:
            continue
        person_ids = present["person_id"].to_numpy()
        points = present[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
        distances = _pairwise_distances(points)
        time = float(time_sec)
        for i, j in zip(*np.triu_indices(len(person_ids), k=1)):
            person_a = person_ids[i]
            person_b = person_ids[j]
            if person_a == person_b:
                continue
            if person_b < person_a:
                person_a, person_b = person_b, person_a
            samples.setdefault((person_a, person_b), []).append((time, float(distances[i, j])))

    rows: list[dict[str, Any]] = []
    for (person_a, person_b), observations in samples.items():
        ordered = sorted(observations)
        times = np.array([item[0] for item in ordered], dtype=np.float64)
        pair_distances = np.array([item[1] for item in ordered], dtype=np.float64)
        # Pre-filter from the positions in this tracks_df only. Do not cache
        # the candidate pairs from ground truth: a perturbed homography can
        # move people who were far apart in ground truth close together, and
        # those false proximity events have to be detected.
        closest_m = float(np.min(pair_distances))
        if closest_m > keep_within_m:
            continue
        intervals = hysteresis_intervals(
            times,
            threshold_m - pair_distances,
            h=hysteresis_m,
            debounce_frames=debounce_frames,
            max_gap_frames=max_gap_frames,
            use_hysteresis=use_hysteresis,
            frame_dt=frame_dt,
        )
        for start_time, end_time in intervals:
            during = (times >= start_time) & (times <= end_time)
            rows.append(
                {
                    "event_type": "proximity",
                    "person_a": person_a,
                    "person_b": person_b,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "min_distance": float(np.min(pair_distances[during])),
                }
            )
    return pd.DataFrame(rows, columns=PROXIMITY_COLUMNS)
