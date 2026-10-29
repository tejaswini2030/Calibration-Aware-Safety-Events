"""Homography estimation and ground-plane projection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data import CAMERA_TO_CALIB, normalize_cam_id, resolve_dataset_root


def _read_opencv_node(node: cv2.FileNode) -> np.ndarray:
    """Read an OpenCV XML node that is either a matrix or a sequence of reals."""
    if node.empty():
        raise ValueError("OpenCV XML node is empty.")
    if node.isSeq():
        values = [float(node.at(index).real()) for index in range(node.size())]
        return np.asarray(values, dtype=np.float64)
    matrix = node.mat()
    if matrix is None or matrix.size == 0:
        raise ValueError("OpenCV XML node is neither a sequence nor a matrix.")
    return np.asarray(matrix, dtype=np.float64)


def _read_opencv_xml(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load named nodes from an OpenCV XML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"Could not open OpenCV XML: {path}")
    try:
        values: dict[str, np.ndarray] = {}
        for key in keys:
            node = storage.getNode(key)
            if node.empty():
                raise KeyError(f"{path} has no '{key}' node")
            values[key] = _read_opencv_node(node)
        return values
    finally:
        storage.release()


def _normalize_homography(H: np.ndarray) -> np.ndarray:
    """Scale a 3x3 homography so that H[2, 2] is 1."""
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got {matrix.shape}")
    scale = matrix[2, 2]
    if abs(scale) < 1e-12:
        raise ValueError("Cannot normalize homography because H[2, 2] is too close to 0.")
    return matrix / scale


def load_camera_calibration(
    root: str | Path,
    cam_id: str | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load K, R, and t for one camera.

    K comes from calibrations/intrinsic_zero (the undistorted images in this
    dataset). Distortion coefficients are ignored. R is cv2.Rodrigues(rvec).
    t is the extrinsic tvec converted from centimeters to meters.

    Returns K (3, 3), R (3, 3), and t (3, 1), with t in meters.
    """
    dataset = resolve_dataset_root(root)
    calib_name = CAMERA_TO_CALIB[normalize_cam_id(cam_id)]
    calib_dir = dataset / "calibrations"
    intrinsics = _read_opencv_xml(
        calib_dir / "intrinsic_zero" / f"intr_{calib_name}.xml",
        ("camera_matrix",),
    )
    extrinsics = _read_opencv_xml(
        calib_dir / "extrinsic" / f"extr_{calib_name}.xml",
        ("rvec", "tvec"),
    )
    K = intrinsics["camera_matrix"].reshape(3, 3)
    rvec = extrinsics["rvec"].reshape(3, 1)
    t_cm = extrinsics["tvec"].reshape(3, 1)
    R, _jacobian = cv2.Rodrigues(rvec)
    # WILDTRACK stores tvec in centimeters (components of a few hundred to
    # about 1000). The ground plane is only 12 m by 36 m, and our world
    # coordinates are meters, so divide by 100.
    t = t_cm / 100.0
    return K, np.asarray(R, dtype=np.float64), t


def ground_homography(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build H_world_to_img for the Z = 0 ground plane.

    H_world_to_img = K @ [r1 r2 t], where r1 and r2 are the first two
    columns of R. The result maps (X, Y) in meters to pixels and is scaled
    so that H[2, 2] = 1. t must already be in meters.
    """
    rotation = np.asarray(R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(t, dtype=np.float64).reshape(3, 1)
    intrinsics = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose = np.hstack([rotation[:, 0:1], rotation[:, 1:2], translation])
    return _normalize_homography(intrinsics @ pose)


def invert_homography(H_world_to_img: np.ndarray) -> np.ndarray:
    """Return H_img_to_world, the inverse of H_world_to_img, with H[2, 2] = 1."""
    inverse = np.linalg.inv(np.asarray(H_world_to_img, dtype=np.float64))
    return _normalize_homography(inverse)


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography.

    Each point becomes homogeneous, is multiplied by H, and is divided by
    its last coordinate. Returns (N, 2).
    """
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got {matrix.shape}")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    mapped = homogeneous @ matrix.T
    scale = mapped[:, 2]
    if np.any(np.abs(scale) < 1e-12):
        raise ValueError("Homography sent a point to infinity (zero third coordinate).")
    return mapped[:, :2] / scale[:, None]


def world_to_image(H: np.ndarray, XY: np.ndarray) -> np.ndarray:
    """Project ground-plane points to image pixels.

    H is H_world_to_img, mapping meters to pixels. XY is (N, 2) in meters
    on Z = 0. Returns (N, 2) pixel coordinates (u, v).
    """
    return apply_homography(H, XY)


def image_to_world(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Project image pixels onto the ground plane.

    H is H_img_to_world, mapping pixels to meters. uv is (N, 2) in pixels.
    Returns (N, 2) ground coordinates (X, Y) in meters.
    """
    return apply_homography(H, uv)
