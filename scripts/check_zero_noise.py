#!/usr/bin/env python3
"""Check that zero calibration noise reproduces ground-truth events exactly."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, load_annotations, project_root, resolve_dataset_root
from src.events import load_event_config
from src.perturb import (
    choose_calibration_points,
    mark_zones_inside_calibration_hull,
    run_trial,
)

ZERO_SEEDS = (0, 1, 2, 3, 4)
NOISE_EPS_PX = 5.0
NOISE_SEED = 0
MAX_ZERO_ERROR_M = 1e-6
# Medium and Far sit inside the wide calibration rectangle. Near is closer
# to the camera than that rectangle's near edge.
EXPECTED_INSIDE = {"near": False, "medium": True, "far": True}


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _format(value: object) -> str:
    if _is_nan(value):
        return "nan"
    return f"{float(value):.4f}"


def _visible_tracks(cam_id: str) -> pd.DataFrame:
    annotations = load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)
    visible = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m"]]
    return visible.copy()


def _print_hull(cfg: dict) -> bool:
    """Print which zones lie inside the calibration hull. Return True if as expected."""
    print("Zone vs wide calibration hull (boundary counts as inside)")
    ok = True
    for zone in cfg["zones"]:
        zone_id = str(zone["id"])
        inside = bool(zone["inside_calibration_hull"])
        place = "inside" if inside else "outside"
        print(f"  {zone_id:<8} {zone['name']:<8} {place}")
        if EXPECTED_INSIDE.get(zone_id) is not inside:
            ok = False
    if ok:
        print("Hull check: PASS (Medium and Far inside, Near outside)")
    else:
        print("Hull check: FAIL (expected Medium and Far inside, Near outside)")
    return ok


def _save_zone_metadata(cfg: dict, cam_id: str) -> Path:
    """Store the hull flag with each zone for later plots."""
    payload = {
        "camera": cam_id,
        "layout": "wide",
        "zones": [
            {
                "id": zone["id"],
                "name": zone["name"],
                "inside_calibration_hull": bool(zone["inside_calibration_hull"]),
            }
            for zone in cfg["zones"]
        ],
    }
    path = project_root() / "outputs" / f"zone_metadata_{cam_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _f1_keys(result: dict) -> dict[str, float]:
    return {key: float(value) for key, value in result.items() if key.startswith("f1_")}


def _zero_noise_ok(result: dict) -> tuple[bool, list[str]]:
    """True when every F1 is 1 and the largest position error is under 1e-6 m."""
    reasons = []
    for key, value in _f1_keys(result).items():
        if value != 1.0:
            reasons.append(f"{key}={value}")
    if result["pos_max_m"] >= MAX_ZERO_ERROR_M:
        reasons.append(f"pos_max_m={result['pos_max_m']}")
    return not reasons, reasons


def _print_score_table(result: dict, zone_ids: list[str]) -> None:
    print(f"eps_px={result['eps_px']}  seed={result['seed']}")
    print(f"  {'score':<22} {'precision':>10} {'recall':>10} {'f1':>10} {'mean delay':>12}")
    rows = [("zone", "all"), ("dwell", "all"), ("proximity", "all")]
    rows.extend(("zone", zone_id) for zone_id in zone_ids)
    rows.extend(("dwell", zone_id) for zone_id in zone_ids)
    for event_type, zone_id in rows:
        suffix = event_type if zone_id == "all" else f"{event_type}_{zone_id}"
        label = event_type if zone_id == "all" else f"{event_type}/{zone_id}"
        delay = result.get(f"start_delay_mean_{event_type}", math.nan) if zone_id == "all" else math.nan
        delay_text = _format(delay) if zone_id == "all" else ""
        print(
            f"  {label:<22} {_format(result[f'precision_{suffix}']):>10} "
            f"{_format(result[f'recall_{suffix}']):>10} {_format(result[f'f1_{suffix}']):>10} "
            f"{delay_text:>12}"
        )
    print("  position error (m)")
    print(f"  {'scope':<22} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10}")
    scopes = ["overall", *zone_ids]
    for scope in scopes:
        suffix = "" if scope == "overall" else f"_{scope}"
        print(
            f"  {scope:<22} {_format(result[f'pos_mean_m{suffix}']):>10} "
            f"{_format(result[f'pos_median_m{suffix}']):>10} "
            f"{_format(result[f'pos_p95_m{suffix}']):>10} "
            f"{_format(result[f'pos_max_m{suffix}']):>10}"
        )


def main() -> None:
    """Run five zero-noise trials and one 5 px trial for camera C1."""
    cfg = load_event_config(project_root() / "configs" / "events.yaml")
    cam_id = str(cfg["camera"])
    calib_world, _pixels = choose_calibration_points(cam_id, layout="wide")
    mark_zones_inside_calibration_hull(cfg["zones"], calib_world)
    hull_ok = _print_hull(cfg)
    metadata_path = _save_zone_metadata(cfg, cam_id)
    print(f"Saved {metadata_path.name}")

    tracks = _visible_tracks(cam_id)
    gt_events = pd.read_csv(project_root() / "outputs" / f"gt_events_{cam_id}.csv")
    zone_ids = [str(zone["id"]) for zone in cfg["zones"]]

    print(f"\nZero noise, eps_px=0, seeds {list(ZERO_SEEDS)}")
    zero_ok = True
    for seed in ZERO_SEEDS:
        result = run_trial(cam_id, 0.0, seed, tracks, gt_events, cfg)
        passed, reasons = _zero_noise_ok(result)
        status = "PASS" if passed else "FAIL"
        detail = "" if passed else "  " + ", ".join(reasons)
        print(
            f"  seed {seed}: max position error {result['pos_max_m']:.3e} m  {status}{detail}"
        )
        zero_ok = zero_ok and passed
    if zero_ok and hull_ok:
        print(
            f"PASS: all {len(ZERO_SEEDS)} zero-noise trials have every F1 = 1.0 "
            f"and max position error < {MAX_ZERO_ERROR_M:g} m"
        )
    else:
        print("FAIL: zero-noise trials or the calibration-hull check did not pass")

    print(f"\nOne noisy trial, eps_px={NOISE_EPS_PX:g}, seed={NOISE_SEED}")
    noisy = run_trial(cam_id, NOISE_EPS_PX, NOISE_SEED, tracks, gt_events, cfg)
    _print_score_table(noisy, zone_ids)
    noise_changed = noisy["pos_max_m"] >= MAX_ZERO_ERROR_M or any(
        value != 1.0 for value in _f1_keys(noisy).values()
    )
    if noise_changed:
        print("eps_px=5 changes position error and event scores relative to zero noise.")
    else:
        print("FAIL: eps_px=5 did not change positions or scores.")
        zero_ok = False

    if not (zero_ok and hull_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
