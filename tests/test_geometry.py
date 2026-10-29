"""Geometry checks for all seven WILDTRACK cameras."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data import CAMERA_TO_CALIB, DATA_ROOT
from src.geometry import (
    ground_homography,
    image_to_world,
    invert_homography,
    load_camera_calibration,
    world_to_image,
)

# Ground rectangle used by the annotations: origin (-3, -9) m and a
# 12 m x 36 m plane, so X is -3..9 and Y is -9..27.
X_MIN_M = -3.0
X_MAX_M = 9.0
Y_MIN_M = -9.0
Y_MAX_M = 27.0
N_POINTS = 100
SEED = 0
CAMERAS = list(CAMERA_TO_CALIB)


def world_points() -> np.ndarray:
    """Return the same 100 ground points for every camera."""
    rng = np.random.default_rng(SEED)
    return np.column_stack(
        [
            rng.uniform(X_MIN_M, X_MAX_M, N_POINTS),
            rng.uniform(Y_MIN_M, Y_MAX_M, N_POINTS),
        ]
    )


@pytest.fixture(scope="module", params=CAMERAS)
def camera(request: pytest.FixtureRequest) -> dict[str, np.ndarray | str]:
    """Load K, R, t and both homographies for one camera."""
    cam_id = request.param
    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    H_world_to_img = ground_homography(K, R, t)
    H_img_to_world = invert_homography(H_world_to_img)
    return {
        "cam_id": cam_id,
        "K": K,
        "R": R,
        "t": t,
        "H_world_to_img": H_world_to_img,
        "H_img_to_world": H_img_to_world,
    }


def test_round_trip(camera: dict[str, np.ndarray | str]) -> None:
    """World to image to world stays within 1e-6 m."""
    xy = world_points()
    uv = world_to_image(camera["H_world_to_img"], xy)
    back = image_to_world(camera["H_img_to_world"], uv)
    error_m = float(np.max(np.abs(back - xy)))
    assert error_m < 1e-6, f"{camera['cam_id']} round-trip error {error_m} m"


def test_matches_project_points(camera: dict[str, np.ndarray | str]) -> None:
    """world_to_image matches cv2.projectPoints for Z=0 and zero distortion."""
    xy = world_points()
    object_points = np.column_stack([xy, np.zeros(len(xy))]).reshape(-1, 1, 3)
    rvec, _jacobian = cv2.Rodrigues(camera["R"])
    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        camera["t"],
        camera["K"],
        np.zeros(5),
    )
    uv = world_to_image(camera["H_world_to_img"], xy)
    error_px = float(np.max(np.abs(projected.reshape(-1, 2) - uv)))
    assert error_px < 0.01, f"{camera['cam_id']} projectPoints error {error_px} px"


def test_homography_invertible(camera: dict[str, np.ndarray | str]) -> None:
    """H_world_to_img has a determinant bounded away from zero."""
    determinant = float(np.linalg.det(camera["H_world_to_img"]))
    assert abs(determinant) > 1e-6, f"{camera['cam_id']} det(H) = {determinant}"
