#!/usr/bin/env python3
"""Build ground-truth zone, dwell, and proximity events for one camera."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    load_event_config,
)

EVENT_TYPE_ORDER = ("zone", "dwell", "proximity")
EVENT_COLORS = {
    "zone": "#e07a2f",
    "dwell": "#2e9b4f",
    "proximity": "#3d6fd8",
}
# One row per event stops being readable once proximity fills the figure.
TOO_MANY_PROXIMITY = 40


def visible_tracks(cam_id: str) -> pd.DataFrame:
    """Load annotations and keep positions this camera can see."""
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    visible = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]]
    return visible.copy()


def collect_events(tracks: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run zone, dwell, and proximity detection and stack the results.

    The detectors pass frame_dt = 1/fps from cfg into hysteresis_intervals.
    """
    zone_events = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell_events = detect_dwell_events(zone_events, cfg)
    proximity_events = detect_proximity_events(tracks, cfg)
    combined = pd.concat([zone_events, dwell_events, proximity_events], ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(["start_time", "event_type"], kind="mergesort").reset_index(drop=True)


def _people_involved(events: pd.DataFrame) -> int:
    """Count distinct people in one event type."""
    if events.empty:
        return 0
    if "person_id" in events.columns and events["person_id"].notna().any():
        return int(events["person_id"].nunique(dropna=True))
    ids = pd.concat([events["person_a"], events["person_b"]], ignore_index=True)
    return int(ids.nunique(dropna=True))


def print_config(cfg: dict, frame_dt: float) -> None:
    """Print the thresholds and timing values used for detection."""
    print(f"camera: {cfg['camera']}")
    print(f"fps: {cfg['fps']}")
    print(f"frame_dt: {frame_dt:.3f} s  (1/fps)")
    print(f"zone_hysteresis_m: {cfg['zone_hysteresis_m']}")
    print(f"proximity_threshold_m: {cfg['proximity_threshold_m']}")
    print(f"proximity_hysteresis_m: {cfg['proximity_hysteresis_m']}")
    print(f"debounce_frames: {cfg['debounce_frames']}")
    print(f"max_gap_frames: {cfg['max_gap_frames']}")
    print(f"dwell_seconds: {cfg['dwell_seconds']}")


def print_event_summary(events: pd.DataFrame, cfg: dict) -> None:
    """Print counts, durations, and the people involved in each event type."""
    print("\nEvents with hysteresis")
    for event_type in EVENT_TYPE_ORDER:
        subset = events.loc[events["event_type"] == event_type] if not events.empty else events
        count = 0 if events.empty else int((events["event_type"] == event_type).sum())
        print(f"  {event_type}: {count}")
        if event_type in {"zone", "dwell"}:
            for zone in cfg["zones"]:
                zone_id = zone["id"]
                if events.empty or "zone_id" not in events.columns:
                    zone_count = 0
                else:
                    zone_count = int(
                        ((events["event_type"] == event_type) & (events["zone_id"] == zone_id)).sum()
                    )
                print(f"    {zone_id}: {zone_count}")
        if count == 0:
            print("    duration mean/median: n/a")
            print("    distinct people: 0")
            continue
        duration = subset["duration"]
        print(f"    duration mean/median: {duration.mean():.2f} s / {duration.median():.2f} s")
        print(f"    distinct people: {_people_involved(subset)}")


def print_comparison(with_hysteresis: pd.DataFrame, without_hysteresis: pd.DataFrame) -> None:
    """Print event counts with and without hysteresis, and their ratio."""
    print("\nevent type        with hysteresis  without hysteresis  ratio (without/with)")
    for event_type in EVENT_TYPE_ORDER:
        with_count = 0 if with_hysteresis.empty else int((with_hysteresis["event_type"] == event_type).sum())
        without_count = (
            0 if without_hysteresis.empty else int((without_hysteresis["event_type"] == event_type).sum())
        )
        if with_count == 0:
            ratio = "inf" if without_count else "—"
        else:
            ratio = f"{without_count / with_count:.2f}"
        print(f"{event_type:<16}  {with_count:>15}  {without_count:>18}  {ratio:>20}")


def save_timeline(events: pd.DataFrame, path: Path, title: str) -> None:
    """Draw one horizontal bar per event, colored by event type."""
    if events.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f"{title} (no events)")
        ax.set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    plotted = events.copy()
    plotted["_order"] = plotted["event_type"].map({name: index for index, name in enumerate(EVENT_TYPE_ORDER)})
    plotted = plotted.sort_values(["_order", "start_time"], kind="mergesort").reset_index(drop=True)
    height = min(18.0, max(4.0, 0.16 * len(plotted)))
    fig, ax = plt.subplots(figsize=(12, height))
    for event_type in EVENT_TYPE_ORDER:
        group = plotted.loc[plotted["event_type"] == event_type]
        if group.empty:
            continue
        ax.barh(
            group.index,
            group["duration"],
            left=group["start_time"],
            color=EVENT_COLORS[event_type],
            height=0.8,
            label=event_type,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("event")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Detect C1 ground-truth events with and without hysteresis."""
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    frame_dt = 1.0 / float(cfg["fps"])
    tracks = visible_tracks(cam_id)
    print(f"Visible rows in {cam_id}: {len(tracks)}")
    print_config(cfg, frame_dt)

    with_hysteresis = collect_events(tracks, cfg)
    output_dir = project_root() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / f"gt_events_{cam_id}.csv"
    with_hysteresis.to_csv(events_path, index=False)
    print_event_summary(with_hysteresis, cfg)
    print(f"Saved {events_path.name}")

    cfg_without = dict(cfg)
    cfg_without["use_hysteresis"] = False
    without_hysteresis = collect_events(tracks, cfg_without)
    nohyst_path = output_dir / f"gt_events_{cam_id}_nohyst.csv"
    without_hysteresis.to_csv(nohyst_path, index=False)
    print(f"\nSaved {nohyst_path.name}")
    print_comparison(with_hysteresis, without_hysteresis)

    timeline_path = output_dir / f"gt_events_timeline_{cam_id}.png"
    save_timeline(with_hysteresis, timeline_path, f"{cam_id} ground-truth events")
    print(f"Saved {timeline_path.name}")
    proximity_count = (
        0 if with_hysteresis.empty else int((with_hysteresis["event_type"] == "proximity").sum())
    )
    if proximity_count > TOO_MANY_PROXIMITY:
        zone_dwell = with_hysteresis.loc[with_hysteresis["event_type"].isin(["zone", "dwell"])]
        zone_dwell_path = output_dir / f"gt_events_timeline_{cam_id}_zone_dwell.png"
        save_timeline(zone_dwell, zone_dwell_path, f"{cam_id} zone and dwell events")
        print(
            f"Saved {zone_dwell_path.name} "
            f"({proximity_count} proximity events are too many to read on one row each)"
        )


if __name__ == "__main__":
    main()
