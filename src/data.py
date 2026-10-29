"""Load WILDTRACK frame annotations and ground-plane positions."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_ROOT = "data/Wildtrack_dataset"

# Image folder to calibration filename stem. The dataset files do not state
# this correspondence. On frame 00000000 the ground grid lies on the pavement
# and projected feet land on the annotated people for every camera.
CAMERA_TO_CALIB: dict[str, str] = {
    "C1": "CVLab1",
    "C2": "CVLab2",
    "C3": "CVLab3",
    "C4": "CVLab4",
    "C5": "IDIAP1",
    "C6": "IDIAP2",
    "C7": "IDIAP3",
}

# README.txt states a 480x1440 grid, 2.5 cm cells, origin (-3.0, -9.0), and
# X = -3.0 + 0.025*(ID % 480), Y = -9.0 + 0.025*(ID / 480).
# rectangles.pom repeats ORIGINE_X -3.0 and ORIGINE_Y -9.0, with a 12 m x 36 m
# ground plane (480 * 0.025, 1440 * 0.025). The formula is already in meters.
# (-300 cm, -900 cm) is the same origin.
GRID_WIDTH = 480
GRID_HEIGHT = 1440
CELL_SIZE_M = 0.025
ORIGIN_X_M = -3.0
ORIGIN_Y_M = -9.0

# Annotated frames are sampled at 2 fps, so neighbors are 0.5 s apart.
ANNOTATED_FPS = 2.0

ANNOTATION_COLUMNS = [
    "frame_number",
    "frame_idx",
    "time_sec",
    "person_id",
    "position_id",
    "X_m",
    "Y_m",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "visible",
]


def project_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[1]


def resolve_dataset_root(root: str | Path) -> Path:
    """Resolve a dataset path from the cwd or the repository root."""
    path = Path(root)
    if path.is_dir():
        return path.resolve()
    from_repo = project_root() / path
    if from_repo.is_dir():
        return from_repo.resolve()
    raise FileNotFoundError(f"Dataset root not found: {root}")


def position_id_to_world(pos_id: int) -> tuple[float, float]:
    """Map a positionID on the 480 x 1440 grid to ground-plane meters.

    grid_x = pos_id % 480 and grid_y = pos_id // 480. Each cell is 2.5 cm.
    The origin (-3.0 m, -9.0 m) is taken from README.txt and rectangles.pom.
    """
    position = int(pos_id)
    if position < 0:
        raise ValueError(f"positionID must be non-negative, got {pos_id}")
    grid_x = position % GRID_WIDTH
    grid_y = position // GRID_WIDTH
    if grid_y >= GRID_HEIGHT:
        raise ValueError(
            f"positionID {pos_id} has grid_y {grid_y}, outside 0..{GRID_HEIGHT - 1}"
        )
    # Round off binary noise from 0.025, which is not an exact float.
    # Six decimals is 0.001 mm, far below the 2.5 cm cell size.
    x_m = round(ORIGIN_X_M + CELL_SIZE_M * grid_x, 6)
    y_m = round(ORIGIN_Y_M + CELL_SIZE_M * grid_y, 6)
    return x_m, y_m


def normalize_cam_id(cam_id: str | int) -> str:
    """Return a camera folder name such as 'C1'."""
    if isinstance(cam_id, int):
        name = f"C{cam_id}"
    else:
        name = str(cam_id).strip().upper()
        if name.isdigit():
            name = f"C{name}"
    if name not in CAMERA_TO_CALIB:
        known = ", ".join(CAMERA_TO_CALIB)
        raise ValueError(f"Unknown cam_id {cam_id!r}. Expected one of: {known}")
    return name


def view_num_for_camera(cam_id: str | int) -> int:
    """Return the 0-based viewNum for a camera. C1 is viewNum 0."""
    name = normalize_cam_id(cam_id)
    return int(name[1:]) - 1


def annotation_dir(root: str | Path) -> Path:
    """Return the annotations_positions directory."""
    path = resolve_dataset_root(root) / "annotations_positions"
    if not path.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {path}")
    return path


def detected_filename_spacing(frame_numbers: list[int]) -> int:
    """Print and return the step between consecutive annotation filenames."""
    if len(frame_numbers) < 2:
        raise ValueError("Need at least two annotation files to detect spacing.")
    steps = sorted({right - left for left, right in zip(frame_numbers, frame_numbers[1:])})
    if len(steps) != 1:
        print(f"Detected annotation filename spacing: {steps}")
        raise ValueError(f"Annotation filename spacing is not constant: {steps}")
    spacing = steps[0]
    time_step_sec = 1.0 / ANNOTATED_FPS
    print(f"Detected annotation filename spacing: {spacing}")
    print(
        f"Annotated frames are {ANNOTATED_FPS:g} fps, so consecutive "
        f"annotated frames are {time_step_sec:g} s apart."
    )
    return spacing


def _view_box(person: dict, view_num: int) -> tuple[int, int, int, int, bool]:
    """Return one camera's box and whether the person is visible in that view."""
    view = next(
        (item for item in person.get("views", []) if item.get("viewNum") == view_num),
        None,
    )
    if view is None:
        return -1, -1, -1, -1, False
    xmin = int(view["xmin"])
    ymin = int(view["ymin"])
    xmax = int(view["xmax"])
    ymax = int(view["ymax"])
    visible = xmin >= 0 and ymin >= 0 and xmax >= 0 and ymax >= 0
    return xmin, ymin, xmax, ymax, visible


def load_annotations(root: str | Path, cam_id: str | int) -> pd.DataFrame:
    """Load per-person annotations for one camera.

    Frames come only from annotations_positions JSON files, so an image with
    no annotation (for example 00002000.png) is ignored. Each row is one
    person in one annotated frame. A box of -1 is stored with visible False.
    time_sec is frame_idx / 2, so consecutive annotated frames are 0.5 s apart.
    """
    view_num = view_num_for_camera(cam_id)
    paths = sorted(annotation_dir(root).glob("*.json"))
    frame_numbers = [int(path.stem) for path in paths]
    if not frame_numbers:
        raise FileNotFoundError(f"No annotation JSON files in {annotation_dir(root)}")
    detected_filename_spacing(frame_numbers)
    time_step_sec = 1.0 / ANNOTATED_FPS

    rows: list[dict] = []
    for frame_idx, path in enumerate(paths):
        frame_number = frame_numbers[frame_idx]
        time_sec = frame_idx * time_step_sec
        people = json.loads(path.read_text(encoding="utf-8"))
        for person in people:
            x_m, y_m = position_id_to_world(person["positionID"])
            xmin, ymin, xmax, ymax, visible = _view_box(person, view_num)
            rows.append(
                {
                    "frame_number": frame_number,
                    "frame_idx": frame_idx,
                    "time_sec": time_sec,
                    "person_id": int(person["personID"]),
                    "position_id": int(person["positionID"]),
                    "X_m": x_m,
                    "Y_m": y_m,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "visible": visible,
                }
            )

    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
