#!/usr/bin/env python3
"""Run person detectors on the annotated frames of one camera.

Writes one CSV per model to results/detections/det_<cam>_<model>.csv and a
timing summary to results/detections/det_<cam>_summary.csv.

Example:
    python scripts/run_detection.py --cam C1 --models yolo11n.pt yolo11x.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, normalize_cam_id, project_root
from src.tracking import DETECTION_COLUMNS, PERSON_CLASS_ID, annotated_frames, detection_rows, save_csv, validate_boxes

# A small and a large model from the same family, so the comparison is about size.
DEFAULT_MODELS = ("yolo11n.pt", "yolo11x.pt")
# Low enough for ByteTrack's second matching round. DeepSORT filters again later.
DEFAULT_CONF = 0.1
# The images are 1920 x 1080 and far people are small, so use more than the default 640.
DEFAULT_IMGSZ = 1280


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cam", default="C1")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default=None, help="for example 0, cpu, or mps; default lets ultralytics choose")
    parser.add_argument("--out-dir", default=str(project_root() / "results" / "detections"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames, for a quick test")
    return parser.parse_args()


def detect(model_path: str, frames: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    """Run one model on every frame. Returns the detections and ms per frame."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    # Warm-up so model loading does not count in the timing.
    model.predict(frames["image_path"].iloc[0], classes=[PERSON_CLASS_ID], imgsz=args.imgsz, device=args.device, verbose=False)
    rows = []
    start = time.perf_counter()
    for frame in tqdm(frames.to_dict("records"), desc=Path(model_path).stem):
        result = model.predict(
            frame["image_path"],
            classes=[PERSON_CLASS_ID],
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        # boxes.xyxy is already in original image pixels.
        rows.extend(detection_rows(frame, result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
    ms_per_frame = 1000.0 * (time.perf_counter() - start) / len(frames)
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS), ms_per_frame


def main() -> None:
    args = parse_args()
    cam_id = normalize_cam_id(args.cam)
    frames = annotated_frames(args.data_root, cam_id)
    if args.limit:
        frames = frames.head(args.limit)
    print(f"{cam_id}: {len(frames)} annotated frames")
    out_dir = Path(args.out_dir)

    summary = []
    for model_path in args.models:
        name = Path(model_path).stem
        detections, ms_per_frame = detect(model_path, frames, args)
        validate_boxes(detections)
        path = save_csv(detections, DETECTION_COLUMNS, out_dir / f"det_{cam_id}_{name}.csv")
        per_frame = len(detections) / len(frames)
        print(f"{name}: {len(detections)} boxes ({per_frame:.1f} per frame), {ms_per_frame:.1f} ms per frame -> {path.name}")
        summary.append(
            {
                "model": name,
                "frames": len(frames),
                "boxes": len(detections),
                "boxes_per_frame": per_frame,
                "ms_per_frame": ms_per_frame,
                "conf": args.conf,
                "imgsz": args.imgsz,
            }
        )
    summary_path = out_dir / f"det_{cam_id}_summary.csv"
    table = pd.DataFrame(summary)
    if summary_path.is_file():
        # Keep rows for models that were not re-run this time.
        previous = pd.read_csv(summary_path)
        table = pd.concat([previous.loc[~previous["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(summary_path, index=False, float_format="%.3f")
    print(f"Saved {summary_path.name}")


if __name__ == "__main__":
    main()
