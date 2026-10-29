#!/usr/bin/env python3
"""Print summary statistics for WILDTRACK annotations on camera C1."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations


def main() -> None:
    """Load C1 annotations and print frame, identity, and ground-plane ranges."""
    frame = load_annotations(DATA_ROOT, "C1")
    timeline = (
        frame[["frame_idx", "frame_number", "time_sec"]]
        .drop_duplicates()
        .sort_values("frame_idx")
    )
    filename_steps = timeline["frame_number"].diff().dropna().unique()
    time_steps = timeline["time_sec"].diff().dropna().unique()
    frames_per_person = frame.groupby("person_id")["frame_idx"].nunique()
    n_frames = int(timeline["frame_idx"].nunique())
    visible_per_frame = (
        frame.loc[frame["visible"]]
        .groupby("frame_idx")
        .size()
        .reindex(range(n_frames), fill_value=0)
    )

    print(f"Frames: {n_frames}")
    print(f"Detected filename spacing: {[int(step) for step in filename_steps]}")
    print(f"Detected time spacing (s): {[float(step) for step in time_steps]}")
    print(f"Unique people: {frame['person_id'].nunique()}")
    print(f"Average frames per person: {frames_per_person.mean():.4f}")
    print(f"X_m min/max: {frame['X_m'].min():.4f} / {frame['X_m'].max():.4f}")
    print(f"Y_m min/max: {frame['Y_m'].min():.4f} / {frame['Y_m'].max():.4f}")
    print(
        "People visible in C1 per frame: "
        f"mean {visible_per_frame.mean():.4f}, "
        f"min {int(visible_per_frame.min())}, "
        f"max {int(visible_per_frame.max())}"
    )


if __name__ == "__main__":
    main()
