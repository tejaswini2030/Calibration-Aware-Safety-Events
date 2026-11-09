#!/usr/bin/env python3
"""Run every tracker on every detection CSV of one camera.

Reads results/detections/det_<cam>_<model>.csv and writes
results/tracks/tracks_<cam>_<model>_<tracker>.csv, so 2 detectors and
2 trackers give 4 track files.

Example:
    python scripts/run_tracking.py --cam C1 --trackers bytetrack deepsort
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ANNOTATED_FPS, DATA_ROOT, normalize_cam_id, project_root
from src.tracking import TRACK_COLUMNS, TRACKERS, annotated_frames, make_tracker, run_tracker, save_csv, validate_tracks

# Keep a lost track alive for 2 s (4 frames at 2 fps), then start a new ID.
MAX_LOST_SECONDS = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--trackers", nargs="+", default=list(TRACKERS))
    parser.add_argument("--det-dir", default=str(project_root() / "results" / "detections"))
    parser.add_argument("--out-dir", default=str(project_root() / "results" / "tracks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    frames = annotated_frames(args.data_root, cam_id)
    det_files = sorted(Path(args.det_dir).glob(f"det_{cam_id}_*.csv"))
    det_files = [path for path in det_files if not path.stem.endswith("_summary")]
    if not det_files:
        raise FileNotFoundError(f"No detection CSVs for {cam_id} in {args.det_dir}. Run scripts/run_detection.py first.")

    for det_path in det_files:
        detections = pd.read_csv(det_path)
        # Frames past the last detected frame (from a --limit test run) are left out.
        used_frames = frames.loc[frames["frame_idx"] <= detections["frame_idx"].max()]
        model = det_path.stem.removeprefix(f"det_{cam_id}_")
        for tracker_name in args.trackers:
            tracker = make_tracker(tracker_name, fps=ANNOTATED_FPS, max_lost_seconds=MAX_LOST_SECONDS)
            start = time.perf_counter()
            tracks = run_tracker(detections, used_frames, tracker)
            seconds = time.perf_counter() - start
            validate_tracks(tracks)
            out = save_csv(tracks, TRACK_COLUMNS, Path(args.out_dir) / f"tracks_{cam_id}_{model}_{tracker_name}.csv")
            print(
                f"{model} + {tracker_name}: {len(tracks)} rows, {tracks['track_id'].nunique()} track IDs, "
                f"{seconds:.1f} s -> {out.name}"
            )


if __name__ == "__main__":
    main()
