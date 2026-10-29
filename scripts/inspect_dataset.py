#!/usr/bin/env python3
"""Print the WILDTRACK layout, one annotation, and camera-1 calibration files."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of scripts/)."""
    return Path(__file__).resolve().parents[1]


def resolve_dataset(explicit: Path | None) -> Path:
    """Find the dataset directory, preferring data/Wildtrack when it exists."""
    if explicit is not None:
        path = explicit if explicit.is_absolute() else repo_root() / explicit
        if not path.is_dir():
            raise SystemExit(f"Dataset path does not exist: {path}")
        print(f"Dataset root: {path}")
        return path

    preferred = repo_root() / "data" / "Wildtrack"
    if preferred.is_dir():
        print(f"Dataset root: {preferred}")
        return preferred

    data_dir = repo_root() / "data"
    candidates: list[Path] = []
    if data_dir.is_dir():
        for child in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            names = {p.name for p in child.iterdir()}
            if {"calibrations", "annotations_positions", "Image_subsets"} & names:
                candidates.append(child)

    if len(candidates) == 1:
        print(f"data/Wildtrack was not found. Using {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise SystemExit("data/Wildtrack was not found, and no dataset folder is under data/.")
    listed = "\n".join(f"  {path}" for path in candidates)
    raise SystemExit(f"data/Wildtrack was not found. Multiple candidates:\n{listed}")


def file_count(directory: Path) -> int:
    """Count non-hidden files directly inside directory."""
    return sum(1 for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))


def print_structure(dataset: Path) -> None:
    """Print the top two directory levels and how many files each holds."""
    print("\n=== 1. Folder structure (top 2 levels) ===")
    print("File counts are non-hidden files directly in that folder.")
    print(f"{dataset.name}/  ({file_count(dataset)} files)")
    for path in sorted(dataset.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_file():
            print(f"  {path.name}  ({path.stat().st_size} bytes)")
            continue
        print(f"  {path.name}/  ({file_count(path)} files)")
        for child in sorted(path.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            print(f"    {path.name}/{child.name}/  ({file_count(child)} files)")


def first_annotation(dataset: Path) -> Path:
    """Return the first annotation JSON, preferring annotations_positions/."""
    preferred = dataset / "annotations_positions"
    search_root = preferred if preferred.is_dir() else dataset
    json_files = sorted(path for path in search_root.rglob("*.json") if path.is_file())
    if not json_files:
        raise SystemExit(f"No JSON files found under {search_root}")
    return json_files[0]


def print_annotation(dataset: Path) -> None:
    """Print the keys of one annotation file and one full record."""
    print("\n=== 2. Annotation JSON ===")
    path = first_annotation(dataset)
    print(f"File: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"Top-level type: {type(payload).__name__}")

    if isinstance(payload, list):
        print(f"Number of entries: {len(payload)}")
        if not payload:
            print("File is an empty list.")
            return
        entry = payload[0]
        if isinstance(entry, dict):
            print(f"Keys of entry 0: {list(entry.keys())}")
        print("Full example entry (index 0):")
        print(json.dumps(entry, indent=2))
        return

    if isinstance(payload, dict):
        print(f"Keys: {list(payload.keys())}")
        if not payload:
            print("File is an empty object.")
            return
        key = next(iter(payload))
        print(f"Full example entry (key {key!r}):")
        print(json.dumps(payload[key], indent=2))
        return

    print("Full payload:")
    print(json.dumps(payload, indent=2))


def describe_format(text: str) -> str:
    """Name the file format from its contents."""
    head = text.lstrip()[:300]
    if head.startswith("<?xml") and "<opencv_storage>" in text:
        return "OpenCV XML (<opencv_storage>)"
    if head.startswith("<?xml") or head.startswith("<"):
        return "XML"
    if head.startswith("{") or head.startswith("["):
        return "JSON"
    return f"unrecognized text starting with {head[:60]!r}"


def print_calibrations(dataset: Path) -> tuple[Path | None, Path | None]:
    """List calibration files and print one camera-1 intrinsic and extrinsic.

    Returns the intrinsic and extrinsic paths that were printed.
    """
    print("\n=== 3. Calibration files ===")
    calib_root = dataset / "calibrations"
    if not calib_root.is_dir():
        xml_files = sorted(dataset.rglob("*.xml"))
        print(f"No calibrations/ directory. XML files found: {len(xml_files)}")
        for path in xml_files:
            print(f"  {path.relative_to(dataset)}")
        return None, None

    files = sorted(path for path in calib_root.rglob("*") if path.is_file() and not path.name.startswith("."))
    print(f"Directory: {calib_root}")
    print(f"File count: {len(files)}")
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"  {path.relative_to(dataset)}  [{describe_format(text)}]")

    readme = dataset / "README.txt"
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    if readme_text:
        print("\nREADME.txt lines that mention calibration or cameras:")
        for line in readme_text.splitlines():
            if re.search(r"calib|intrinsic|extrinsic|camera", line, re.I):
                print(f"  {line}")

    image_dirs = []
    image_root = dataset / "Image_subsets"
    if image_root.is_dir():
        image_dirs = sorted(path.name for path in image_root.iterdir() if path.is_dir())
    print(f"\nImage camera folders: {image_dirs or 'none found'}")
    print(
        "No file in this dataset maps an image folder such as C1 onto a "
        "calibration filename. Filenames use CVLab1-4 and IDIAP1-3."
    )
    print("Printing CVLab1 as camera 1 (extrinsic + intrinsic_zero, the set README says matches these images).")

    extrinsic = _match_file(files, parent_token="extrinsic", stem_token="CVLab1")
    intrinsic = _match_file(files, parent_token="intrinsic_zero", stem_token="CVLab1")
    if intrinsic is None:
        intrinsic = _match_file(files, parent_token="intrinsic", stem_token="CVLab1")

    for label, path in (("extrinsic", extrinsic), ("intrinsic", intrinsic)):
        print(f"\n--- camera 1 {label}: {path if path is None else path.relative_to(dataset)} ---")
        if path is None:
            print("Not found.")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"Format: {describe_format(text)}")
        print(text.rstrip())
    return intrinsic, extrinsic


def _match_file(files: list[Path], parent_token: str, stem_token: str) -> Path | None:
    """Return the first file whose parent path and stem contain the given tokens."""
    matches = [
        path
        for path in files
        if parent_token in str(path.parent).lower() and stem_token.lower() in path.stem.lower()
    ]
    return matches[0] if matches else None


def _tagged_numbers(text: str, tag: str) -> list[float]:
    """Return floats inside the first <tag>...</tag> block."""
    match = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.I | re.S)
    if match is None:
        return []
    return [float(value) for value in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]


def print_translation_units(dataset: Path, extrinsic: Path | None) -> None:
    """Print translation magnitudes and whether they look like meters or centimeters."""
    print("\n=== 4. Translation magnitude ===")
    if extrinsic is None:
        print("No extrinsic file was selected, so translation units were not checked.")
        return

    text = extrinsic.read_text(encoding="utf-8", errors="replace")
    tags = sorted(set(re.findall(r"</?([A-Za-z_][\w:.-]*)\b", text)))
    print(f"XML tags in {extrinsic.name}: {tags}")

    components: list[float] = []
    used_tag = None
    for tag in ("tvec", "T", "translation", "Trans", "t"):
        components = _tagged_numbers(text, tag)
        if components:
            used_tag = tag
            break
    if not components:
        print("No translation-like block with numbers was found (looked for tvec, T, translation, Trans, t).")
        return

    magnitudes = [abs(value) for value in components]
    norm = math.sqrt(sum(value * value for value in components))
    print(f"Translation tag: <{used_tag}>")
    print(f"Components: {components}")
    print(f"Absolute values: {magnitudes}")
    print(f"Max absolute value: {max(magnitudes):.6g}")
    print(f"L2 norm: {norm:.6g}")

    readme_path = dataset / "README.txt"
    readme = readme_path.read_text(encoding="utf-8", errors="replace") if readme_path.is_file() else ""
    spacing = re.search(r"([0-9.]+)\s*cm spacing", readme, re.I)
    grid = re.search(r"(\d+)\s*x\s*(\d+)\s*grid", readme, re.I)
    if spacing and grid:
        cell_m = float(spacing.group(1)) / 100.0
        width_m = int(grid.group(1)) * cell_m
        height_m = int(grid.group(2)) * cell_m
        scene_m = max(width_m, height_m)
        print(
            f"README ground plane: {grid.group(1)} x {grid.group(2)} grid, "
            f"{spacing.group(1)} cm cells -> {width_m:.3g} m x {height_m:.3g} m."
        )
        ratio = max(magnitudes) / scene_m if scene_m else float("inf")
        print(f"Max |translation| / longer ground-plane side: {ratio:.4g}")
        if ratio > 20:
            print(
                "The translation magnitudes are much larger than this ground plane in meters, "
                "so they look like centimeters."
            )
        elif ratio < 5:
            print(
                "The translation magnitudes are on the same scale as this ground plane in meters, "
                "so they look like meters."
            )
        else:
            print("The translation magnitudes are between those two scales; units are ambiguous from size alone.")
    else:
        print("README did not state a grid size and centimeter spacing, so no unit comparison was made.")


def image_size(path: Path) -> tuple[int, int, str]:
    """Read width and height from a PNG or JPEG header."""
    with path.open("rb") as handle:
        header = handle.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return width, height, "PNG IHDR"
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_start = handle.read(1)
                if marker_start != b"\xff":
                    break
                marker = handle.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                segment_length = struct.unpack(">H", handle.read(2))[0]
                if marker in {b"\xc0", b"\xc1", b"\xc2"}:
                    precision_and_size = handle.read(5)
                    height, width = struct.unpack(">HH", precision_and_size[1:5])
                    return width, height, f"JPEG SOF {marker.hex()}"
                handle.seek(segment_length - 2, 1)
    raise SystemExit(f"Could not read image dimensions from header: {path}")


def first_image(dataset: Path) -> Path:
    """Return the first image file under the dataset."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(
        path
        for path in dataset.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith(".")
    )
    if not images:
        raise SystemExit(f"No image files found under {dataset}")
    return images[0]


def print_image_size(dataset: Path) -> None:
    """Print the pixel size of one frame."""
    print("\n=== 5. Image frame size ===")
    path = first_image(dataset)
    width, height, source = image_size(path)
    print(f"File: {path}")
    print(f"Size: {width} x {height} pixels ({source})")


def main() -> None:
    """Run every inspection section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        help="Dataset root. Defaults to data/Wildtrack, or the WILDTRACK folder found under data/.",
    )
    args = parser.parse_args()
    dataset = resolve_dataset(args.dataset)
    print_structure(dataset)
    print_annotation(dataset)
    _intrinsic, extrinsic = print_calibrations(dataset)
    print_translation_units(dataset, extrinsic)
    print_image_size(dataset)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
