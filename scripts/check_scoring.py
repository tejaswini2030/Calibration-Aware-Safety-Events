#!/usr/bin/env python3
"""Score ground-truth events against themselves. Every F1 must be 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import project_root
from src.evaluation import frame_dt_from_config, score_events
from src.events import load_event_config


def _print_scores(scores: pd.DataFrame) -> None:
    """Print the score table with undefined rates shown as nan."""
    display = scores.copy()
    for column in ("precision", "recall", "f1"):
        display[column] = display[column].map(
            lambda value: "nan" if pd.isna(value) else f"{float(value):.3f}"
        )
    print(display.to_string(index=False))


def main() -> None:
    """Score the saved ground-truth events against a copy of themselves."""
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    path = project_root() / "outputs" / f"gt_events_{cam_id}.csv"
    gt = pd.read_csv(path)
    frame_dt = frame_dt_from_config(cfg)
    print(
        f"Scoring {path.name} against itself "
        f"(dwell_seconds={cfg['dwell_seconds']}, frame_dt={frame_dt:.3f} s)"
    )
    scores, matches = score_events(gt, gt, cfg)
    _print_scores(scores)

    f1 = scores["f1"].to_numpy(dtype=float)
    if not np.all(np.isclose(f1, 1.0)):
        raise SystemExit("Expected every F1 to be 1.0 when scoring ground truth against itself.")
    if len(matches) != len(gt) or not np.allclose(matches["start_delay"], 0.0):
        raise SystemExit("Self-match should pair every event with itself at zero delay.")
    print(f"Every F1 is 1.0. Matched pairs: {len(matches)}, all with start delay 0.")


if __name__ == "__main__":
    main()
