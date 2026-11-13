#!/usr/bin/env python3
"""Check every track CSV of one camera against the ground truth.

For each tracks_<cam>_<detector>_<tracker>.csv this script
  1. validates the format,
  2. projects each box's bottom-center to the ground plane with H*,
  3. drops predicted feet outside the annotated ground area,
  4. matches the remaining feet to ground-truth positions (within 0.5 m),
  5. scores with a boundary don't-care margin (default 1 m) and also margin 0,
  6. also matches boxes by IoU, kept as a secondary score,
  7. saves the track-to-person link from the ground-plane matches, and
  8. draws one frame with predicted and ground-truth boxes.

Scores and id maps stay in results/tracks/. Overlay PNGs go to
outputs/tracks/ so the large images are not committed. Foot-point
diagnostics live in scripts/diagnose_foot_points.py.
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
    DEFAULT_BOUNDARY_MARGIN_M,
    DISTANT_FOOT_M,
    GROUND_MATCH_DISTANCE_M,
    annotated_frames,
    classify_distant_feet,
    drop_outside_ground_area,
    foot_points_world,
    gt_boxes,
    gt_positions,
    in_ground_area,
    iter_track_files,
    match_tracks_ground,
    nearest_gt_distance,
    relabel_tracks,
    track_to_person_map,
    tracking_summary,
    tracking_summary_ground,
    validate_tracks,
)

IOU_THRESHOLD = 0.5
OVERLAY_FRAME_IDX = 100
NO_PERSON_MODEL = "yolo11x_deepsort"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--tracks-dir", default=str(project_root() / "results" / "tracks"))
    parser.add_argument("--frame", type=int, default=OVERLAY_FRAME_IDX, help="frame_idx for the overlay image")
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=DEFAULT_BOUNDARY_MARGIN_M,
        help="meters to shrink the ground rectangle when counting TP/FP/FN (default 1.0)",
    )
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


def _print_no_person_edge(
    name: str,
    feet: pd.DataFrame,
    boxes: pd.DataFrame,
    covered_feet: pd.DataFrame,
    margin_m: float,
) -> None:
    """X distribution of no_person feet and how many sit in the don't-care band."""
    nearest = nearest_gt_distance(feet, covered_feet)
    labels = classify_distant_feet(feet, boxes, nearest, DISTANT_FOOT_M)
    no_person = feet.loc[labels == "no_person"]
    if no_person.empty:
        print(f"\n{name}: no no_person feet")
        return
    x = no_person["X_m"].to_numpy(dtype=float)
    in_band = in_ground_area(no_person["X_m"], no_person["Y_m"], 0.0) & ~in_ground_area(
        no_person["X_m"], no_person["Y_m"], margin_m
    )
    print(f"\n{name} no_person feet (n={len(no_person)}): X_m min/p5/p50/p95 = "
          f"{x.min():.3f} / {np.percentile(x, 5):.3f} / {np.percentile(x, 50):.3f} / {np.percentile(x, 95):.3f}")
    print(f"{name} no_person feet inside the {margin_m:g} m boundary band: {int(in_band.sum())} "
          f"({100.0 * in_band.mean():.1f}%)")


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    tracks_dir = Path(args.tracks_dir)
    files = list(iter_track_files(tracks_dir, cam_id))
    if not files:
        raise FileNotFoundError(f"No tracks_{cam_id}_*.csv in {tracks_dir}. Run scripts/run_tracking.py first.")
    margin = float(args.boundary_margin)

    annotations = load_annotations(resolve_dataset_root(args.data_root), cam_id)
    gt = gt_boxes(annotations)
    positions = gt_positions(annotations)
    frames = annotated_frames(args.data_root, cam_id)
    image_path = frames.loc[frames["frame_idx"] == args.frame, "image_path"].iloc[0]
    overlay_dir = project_root() / "outputs" / "tracks"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        name = path.stem.removeprefix(f"tracks_{cam_id}_")
        tracks = pd.read_csv(path)
        validate_tracks(tracks)
        # Score only the frames this file covers (a --limit run covers fewer).
        last_frame = int(tracks["frame_idx"].max())
        covered_boxes = gt.loc[gt["frame_idx"] <= last_frame]
        covered_feet = positions.loc[positions["frame_idx"] <= last_frame]
        iou = tracking_summary(tracks, covered_boxes, IOU_THRESHOLD)

        world = foot_points_world(tracks, cam_id)
        on_ground, dropped_outside = drop_outside_ground_area(world)
        print(f"{name}: dropped {dropped_outside} predicted rows outside the ground area")
        ground_0 = tracking_summary_ground(on_ground, covered_feet, GROUND_MATCH_DISTANCE_M, boundary_margin_m=0.0)
        ground = tracking_summary_ground(on_ground, covered_feet, GROUND_MATCH_DISTANCE_M, boundary_margin_m=margin)
        matches = match_tracks_ground(on_ground, covered_feet, GROUND_MATCH_DISTANCE_M)
        idmap = track_to_person_map(matches, on_ground)
        idmap.to_csv(tracks_dir / f"idmap_{cam_id}_{name}.csv", index=False, float_format="%.3f")
        _relabeled, dropped_same_person = relabel_tracks(on_ground, idmap)
        print(f"{name}: dropped {dropped_same_person} rows that shared a person_id in one frame")
        rows.append(
            {
                "model": name,
                "dropped_outside": dropped_outside,
                "iou": iou,
                "ground_0": ground_0,
                "ground": ground,
                "feet": on_ground,
                "covered_feet": covered_feet,
            }
        )

        draw_overlay(
            image_path,
            tracks.loc[tracks["frame_idx"] == args.frame],
            gt.loc[gt["frame_idx"] == args.frame],
            f"{cam_id} {name} frame {args.frame}",
            overlay_dir / f"overlay_{cam_id}_{name}.png",
        )

    comparison = pd.DataFrame(
        [
            {
                "model": row["model"],
                "P_0": row["ground_0"]["det_precision"],
                "P_m": row["ground"]["det_precision"],
                "R_0": row["ground_0"]["det_recall"],
                "R_m": row["ground"]["det_recall"],
                "ids_0": row["ground_0"]["id_switches"],
                "ids_m": row["ground"]["id_switches"],
                "mota_0": row["ground_0"]["mota"],
                "mota_m": row["ground"]["mota"],
                "ignored_pred": row["ground"]["ignored_pred"],
                "ignored_gt": row["ground"]["ignored_gt"],
            }
            for row in rows
        ]
    )
    print(
        f"\n{cam_id} ground-plane scores at {GROUND_MATCH_DISTANCE_M:g} m "
        f"(margin 0 beside margin {margin:g} m; margin {margin:g} is the main result)"
    )
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    main_table = pd.DataFrame(
        [
            {
                "model": row["model"],
                "det_precision": row["ground"]["det_precision"],
                "det_recall": row["ground"]["det_recall"],
                "mean_distance_m": row["ground"]["mean_distance_m"],
                "id_switches": row["ground"]["id_switches"],
                "mota": row["ground"]["mota"],
                "pred_tracks": row["ground"]["pred_tracks"],
                "gt_people": row["ground"]["gt_people"],
                "mean_track_purity": row["ground"]["mean_track_purity"],
                "ignored_pred": row["ground"]["ignored_pred"],
                "ignored_gt": row["ground"]["ignored_gt"],
                "boundary_margin_m": margin,
            }
            for row in rows
        ]
    )
    print(f"\nMain table (boundary margin {margin:g} m)")
    print(main_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    example = next((row for row in rows if row["model"] == NO_PERSON_MODEL), None)
    if example is not None:
        last = int(example["feet"]["frame_idx"].max()) if len(example["feet"]) else -1
        _print_no_person_edge(
            example["model"],
            example["feet"],
            gt.loc[gt["frame_idx"] <= last],
            example["covered_feet"],
            margin,
        )

    saved = []
    for row in rows:
        flat = {"model": row["model"], "dropped_outside": row["dropped_outside"]}
        flat.update({f"iou_{key}": value for key, value in row["iou"].items()})
        flat.update({f"ground0_{key}": value for key, value in row["ground_0"].items()})
        flat.update({f"ground_{key}": value for key, value in row["ground"].items()})
        saved.append(flat)
    table = pd.DataFrame(saved)
    out = tracks_dir / f"tracking_summary_{cam_id}.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(f"\nSaved {out.name} (margin 0 and margin {margin:g} m) and idmap_*.csv in {tracks_dir}")
    print(f"Saved overlay_*.png in {overlay_dir}")


if __name__ == "__main__":
    main()
