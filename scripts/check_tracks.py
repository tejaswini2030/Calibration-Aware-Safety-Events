#!/usr/bin/env python3
"""Check every track CSV of one camera against the ground-truth boxes.

For each tracks_<cam>_<detector>_<tracker>.csv this script
  1. validates the format,
  2. matches boxes to ground-truth boxes per frame (IoU >= 0.5),
  3. prints detection precision and recall, ID switches, and MOTA,
  4. saves the track-to-person link the event scoring needs, and
  5. draws one frame with predicted and ground-truth boxes.

Outputs go to results/tracks/: tracking_summary_<cam>.csv,
idmap_<cam>_<detector>_<tracker>.csv, and overlay_<cam>_<detector>_<tracker>.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, normalize_cam_id, project_root, resolve_dataset_root
from src.tracking import (
    annotated_frames,
    gt_boxes,
    iter_track_files,
    match_tracks_to_gt,
    track_to_person_map,
    tracking_summary,
    validate_tracks,
)

IOU_THRESHOLD = 0.5
OVERLAY_FRAME_IDX = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--tracks-dir", default=str(project_root() / "results" / "tracks"))
    parser.add_argument("--frame", type=int, default=OVERLAY_FRAME_IDX, help="frame_idx for the overlay image")
    return parser.parse_args()


def _color(track_id: int) -> tuple[int, int, int]:
    """A stable, bright BGR color per track ID."""
    hue = (int(track_id) * 37) % 180
    pixel = cv2.cvtColor(np.array([[[hue, 220, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return int(pixel[0]), int(pixel[1]), int(pixel[2])


def draw_overlay(image_path: str, tracks: pd.DataFrame, gt: pd.DataFrame, title: str, path: Path) -> None:
    """Ground truth in thin white, predictions in color with their track ID."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    for row in gt.itertuples(index=False):
        cv2.rectangle(image, (int(row.xmin), int(row.ymin)), (int(row.xmax), int(row.ymax)), (255, 255, 255), 1)
    for row in tracks.itertuples(index=False):
        color = _color(row.track_id)
        top_left = (int(row.xmin), int(row.ymin))
        cv2.rectangle(image, top_left, (int(row.xmax), int(row.ymax)), color, 2)
        label = f"{int(row.track_id)} {row.score:.2f}"
        cv2.putText(image, label, (top_left[0], max(14, top_left[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, label, (top_left[0], max(14, top_left[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for thickness, color in ((3, (0, 0, 0)), (1, (255, 255, 255))):
        cv2.putText(image, title, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    tracks_dir = Path(args.tracks_dir)
    files = list(iter_track_files(tracks_dir, cam_id))
    if not files:
        raise FileNotFoundError(f"No tracks_{cam_id}_*.csv in {tracks_dir}. Run scripts/run_tracking.py first.")

    gt = gt_boxes(load_annotations(resolve_dataset_root(args.data_root), cam_id))
    frames = annotated_frames(args.data_root, cam_id)
    image_path = frames.loc[frames["frame_idx"] == args.frame, "image_path"].iloc[0]

    rows = []
    for path in files:
        name = path.stem.removeprefix(f"tracks_{cam_id}_")
        tracks = pd.read_csv(path)
        validate_tracks(tracks)
        # Score only the frames this file covers (a --limit run covers fewer).
        covered = gt.loc[gt["frame_idx"] <= tracks["frame_idx"].max()]
        summary = tracking_summary(tracks, covered, IOU_THRESHOLD)
        rows.append({"model": name, **summary})

        matches = match_tracks_to_gt(tracks, covered, IOU_THRESHOLD)
        track_to_person_map(matches, tracks).to_csv(tracks_dir / f"idmap_{cam_id}_{name}.csv", index=False, float_format="%.3f")
        draw_overlay(
            image_path,
            tracks.loc[tracks["frame_idx"] == args.frame],
            gt.loc[gt["frame_idx"] == args.frame],
            f"{cam_id} {name} frame {args.frame}",
            tracks_dir / f"overlay_{cam_id}_{name}.png",
        )

    table = pd.DataFrame(rows)
    print(f"\n{cam_id} tracking vs ground-truth boxes (IoU >= {IOU_THRESHOLD})")
    shown = ["model", "det_precision", "det_recall", "mean_iou", "id_switches", "mota", "pred_tracks", "gt_people"]
    print(table[shown].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    out = tracks_dir / f"tracking_summary_{cam_id}.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out.name}, idmap_*.csv and overlay_*.png in {tracks_dir}")


if __name__ == "__main__":
    main()
