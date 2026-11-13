"""Tests for exhaustive RANSAC homography estimation."""

from __future__ import annotations

import numpy as np

from src.geometry import ransac_homography_exhaustive, transfer_errors_px
from src.perturb import choose_calibration_points, estimate_H_img_to_world


def test_ransac_keeps_all_points_when_there_is_no_outlier() -> None:
    """Exact n12 correspondences: every point is an inlier under a 3 px gate."""
    world, pixels = choose_calibration_points("C1", layout="n12")
    assert len(world) == 12
    result = ransac_homography_exhaustive(pixels, world, threshold_px=3.0)
    assert result["n_inliers"] == 12
    assert bool(np.all(result["inlier_mask"]))
    assert np.max(result["transfer_errors_px"]) < 1e-6


def test_ransac_finds_a_30px_outlier() -> None:
    """Moving one clicked pixel by 30 px should drop that point as an outlier."""
    world, pixels = choose_calibration_points("C1", layout="n12")
    clicked = pixels.copy()
    outlier_index = 5
    clicked[outlier_index] = clicked[outlier_index] + np.array([30.0, 0.0])

    # Least squares is pulled by the outlier; RANSAC should reject it.
    H_dlt = estimate_H_img_to_world(clicked, world)
    dlt_errors = transfer_errors_px(H_dlt, clicked, world)
    assert dlt_errors[outlier_index] > 3.0

    result = ransac_homography_exhaustive(clicked, world, threshold_px=3.0)
    assert result["n_inliers"] == 11
    assert not bool(result["inlier_mask"][outlier_index])
    assert bool(np.all(np.delete(result["inlier_mask"], outlier_index)))
    assert result["transfer_errors_px"][outlier_index] > 3.0
    assert np.max(result["transfer_errors_px"][result["inlier_mask"]]) <= 3.0 + 1e-9
