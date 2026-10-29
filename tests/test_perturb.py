"""Tests for normalized DLT and calibration-point perturbation."""

from __future__ import annotations

import cv2
import numpy as np

from src.data import DATA_ROOT
from src.geometry import (
    apply_homography,
    ground_homography,
    invert_homography,
    load_camera_calibration,
    normalized_dlt,
    world_to_image,
)
from src.perturb import choose_calibration_points, estimate_H_img_to_world, perturb_pixels


def _scale_h22(H: np.ndarray) -> np.ndarray:
    """Scale a homography so that H[2, 2] is 1."""
    return np.asarray(H, dtype=np.float64) / H[2, 2]


def test_zero_noise_recovers_ground_truth_homography() -> None:
    """Four exact wide-layout correspondences recover H_img_to_world."""
    world, pixels = choose_calibration_points("C1", layout="wide")
    assert world.shape == (4, 2)
    assert pixels.shape == (4, 2)
    K, R, t = load_camera_calibration(DATA_ROOT, "C1")
    H_world_to_img = ground_homography(K, R, t)
    exact_pixels = world_to_image(H_world_to_img, world)
    assert np.max(np.abs(pixels - exact_pixels)) < 1e-8
    assert np.all(pixels[:, 0] >= 0.0) and np.all(pixels[:, 0] < 1920.0)
    assert np.all(pixels[:, 1] >= 0.0) and np.all(pixels[:, 1] < 1080.0)

    H_true = invert_homography(H_world_to_img)
    H_est = estimate_H_img_to_world(pixels, world)
    difference = np.max(np.abs(_scale_h22(H_est) - _scale_h22(H_true)))
    assert difference < 1e-8, f"max element difference {difference}"


def test_normalized_dlt_matches_opencv_on_noisy_correspondences() -> None:
    """Ten noisy correspondences project the same way as cv2.findHomography(method=0)."""
    rng = np.random.default_rng(0)
    src = rng.normal(size=(10, 2))
    true_h = np.array(
        [[1.05, -0.10, 0.20], [0.08, 0.95, -0.15], [0.01, -0.02, 1.0]],
        dtype=np.float64,
    )
    # Noise keeps the correspondences off the exact homography, so N=10 is
    # least squares. OpenCV stores points as float32 and then runs a few
    # geometric-refinement steps; this noise stays inside the 1e-6 agreement.
    dst = apply_homography(true_h, src) + rng.normal(0.0, 1e-5, size=(10, 2))
    ours = normalized_dlt(src, dst)
    found = cv2.findHomography(src, dst, method=0)
    opencv_h = found[0] if isinstance(found, tuple) else found
    projected_ours = apply_homography(ours, src)
    projected_opencv = apply_homography(opencv_h, src)
    error = float(np.max(np.abs(projected_ours - projected_opencv)))
    assert error < 1e-6, f"projected point difference {error}"
    assert np.max(np.abs(dst - apply_homography(true_h, src))) > 0.0


def test_perturb_pixels_moves_each_point_by_exactly_eps() -> None:
    """Every point moves by eps_px, not by a shorter random radius."""
    pixels = np.array([[10.0, 20.0], [100.0, 200.0], [0.0, 0.0], [1919.0, 1079.0]])
    eps_px = 5.0
    moved = perturb_pixels(pixels, eps_px, np.random.default_rng(1))
    distance = np.linalg.norm(moved - pixels, axis=1)
    assert np.max(np.abs(distance - eps_px)) < 1e-9
    assert not np.allclose(moved, pixels)


def test_perturb_pixels_is_determined_by_the_rng_seed() -> None:
    """Two generators created with the same seed produce the same perturbation."""
    pixels = np.array([[0.0, 0.0], [10.0, -3.0], [50.0, 80.0]])
    first = perturb_pixels(pixels, 3.0, np.random.default_rng(7))
    second = perturb_pixels(pixels, 3.0, np.random.default_rng(7))
    assert np.array_equal(first, second)
    other = perturb_pixels(pixels, 3.0, np.random.default_rng(8))
    assert not np.array_equal(first, other)
