#!/usr/bin/env python3
"""Save C6 wide (n4) calibration overlay and report which zones lie in the hull."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import project_root
from src.events import event_config_path, load_event_config
from src.perturb import (
    choose_calibration_points,
    mark_zones_inside_calibration_hull,
    save_calibration_overlay,
)


def main() -> None:
    cam_id = "C6"
    cfg = load_event_config(event_config_path(cam_id))
    world, pixels = choose_calibration_points(cam_id, layout="n4")
    print(f"{cam_id} wide (n4) calibration rectangle")
    print(
        f"  X [{world[:, 0].min():.1f}, {world[:, 0].max():.1f}]  "
        f"Y [{world[:, 1].min():.1f}, {world[:, 1].max():.1f}]"
    )
    for index, (xy, uv) in enumerate(zip(world, pixels)):
        print(
            f"  {index}: world ({xy[0]:.3f}, {xy[1]:.3f}) m"
            f"    pixel ({uv[0]:.2f}, {uv[1]:.2f})"
        )

    zones = copy.deepcopy(cfg["zones"])
    mark_zones_inside_calibration_hull(zones, world)
    print("Zones inside wide calibration hull:")
    for zone in zones:
        print(f"  {zone['id']}: {bool(zone['inside_calibration_hull'])}")

    n12, _ = choose_calibration_points(cam_id, layout="n12")
    print(f"n12 grid: {len(n12)} points spanning the same rectangle")

    path = project_root() / "outputs" / f"calib_points_{cam_id}.png"
    save_calibration_overlay(cam_id, world, pixels, path, layout="wide")
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
