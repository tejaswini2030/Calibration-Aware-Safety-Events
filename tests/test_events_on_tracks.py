"""True positions through the track-event path must recover the answer key."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DATA_ROOT, project_root, resolve_dataset_root
from src.evaluation import score_events
from src.events import load_event_config


def _load_script():
    path = ROOT / "scripts" / "run_events_on_tracks.py"
    spec = importlib.util.spec_from_file_location("run_events_on_tracks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_positions_reproduce_ground_truth_events() -> None:
    """Annotated (X, Y) without box projection must score F1 = 1.0 against gt_events."""
    dataset = resolve_dataset_root(DATA_ROOT)
    gt_path = project_root() / "outputs" / "gt_events_C1.csv"
    if not dataset.is_dir() or not gt_path.is_file():
        pytest.skip("WILDTRACK and gt_events_C1.csv are required for this check")

    from src.data import load_annotations

    module = _load_script()
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    annotations = load_annotations(dataset, "C1")
    positions = module.positions_from_gt_world(annotations)
    events = module.detect_events_on_positions(positions, cfg)
    gt_events = __import__("pandas").read_csv(gt_path)
    scores, matches = score_events(gt_events, events, cfg)
    type_rows = scores.loc[scores["zone_id"] == "all"]
    assert len(type_rows) == 3
    assert (type_rows["f1"] == 1.0).all()
    assert len(matches) == len(gt_events)
    assert float(matches["start_delay"].abs().max()) == 0.0
