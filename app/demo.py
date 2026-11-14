#!/usr/bin/env python3
"""Streamlit demo: calibration noise, camera bump, and event scoring on WILDTRACK.

Run from the repository root:
  streamlit run app/demo.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DATA_ROOT, load_annotations, normalize_cam_id, project_root, resolve_dataset_root
from src.evaluation import score_events
from src.events import (
    detect_dwell_events,
    detect_proximity_events,
    detect_zone_events,
    event_config_path,
    load_event_config,
)
from src.geometry import (
    ground_homography,
    image_to_world,
    invert_homography,
    load_camera_calibration,
    normalized_dlt,
    ransac_homography_exhaustive,
    world_to_image,
)
from src.perturb import choose_calibration_points, perturb_pixels
from src.tracking import annotated_frames, camera_ground_xy, relabel_tracks, validate_tracks

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZONE_COLORS = {
    "near": (0.95, 0.45, 0.10),
    "medium": (0.15, 0.65, 0.30),
    "far": (0.20, 0.40, 0.85),
}
EVENT_COLORS = {
    "zone": "#e07a2f",
    "dwell": "#2e9b4f",
    "proximity": "#3d6fd8",
}
AXIS_VECTORS = {
    "pitch": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "yaw": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "roll": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}
PRED_TRACKS = {
    "yolo11n + ByteTrack": "yolo11n_bytetrack",
    "yolo11n + DeepSORT": "yolo11n_deepsort",
    "yolo11x + ByteTrack": "yolo11x_bytetrack",
    "yolo11x + DeepSORT": "yolo11x_deepsort",
}
TRANSFER_THRESH_PX = 3.0
PROX_CIRCLE_R_M = 0.75


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_frames(cam_id: str) -> pd.DataFrame:
    return annotated_frames(DATA_ROOT, cam_id)


@st.cache_data(show_spinner=False)
def cached_annotations(cam_id: str) -> pd.DataFrame:
    return load_annotations(resolve_dataset_root(DATA_ROOT), cam_id)


@st.cache_data(show_spinner=False)
def cached_config(cam_id: str) -> dict[str, Any]:
    return load_event_config(event_config_path(cam_id))


@st.cache_data(show_spinner=False)
def cached_pose(cam_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K, R, t = load_camera_calibration(DATA_ROOT, cam_id)
    H_w2i = ground_homography(K, R, t)
    H_i2w = invert_homography(H_w2i)
    return K, R, t, H_w2i, H_i2w


@st.cache_data(show_spinner=False)
def cached_calib_points(cam_id: str, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    layout = "n4" if int(n_points) == 4 else "n12"
    world, pixels = choose_calibration_points(cam_id, layout=layout)
    return world, pixels


@st.cache_data(show_spinner=False)
def cached_gt_events(cam_id: str) -> pd.DataFrame:
    path = project_root() / "outputs" / f"gt_events_{cam_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/make_gt_events.py --cam {cam_id}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def cached_image(cam_id: str, frame_idx: int) -> np.ndarray:
    frames = cached_frames(cam_id)
    row = frames.loc[frames["frame_idx"] == int(frame_idx)]
    if row.empty:
        raise ValueError(f"No frame_idx {frame_idx} for {cam_id}")
    path = Path(str(row.iloc[0]["image_path"]))
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


@st.cache_data(show_spinner=False)
def cached_pred_feet(model_key: str) -> pd.DataFrame:
    """Bottom-center pixels with H*-matched person_id for one C1 tracker."""
    tracks_dir = project_root() / "results" / "tracks"
    tracks = pd.read_csv(tracks_dir / f"tracks_C1_{model_key}.csv")
    idmap = pd.read_csv(tracks_dir / f"idmap_C1_{model_key}.csv")
    validate_tracks(tracks)
    relabeled, _dropped = relabel_tracks(tracks, idmap)
    u = (relabeled["xmin"].to_numpy(dtype=float) + relabeled["xmax"].to_numpy(dtype=float)) / 2.0
    v = relabeled["ymax"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "time_sec": relabeled["time_sec"].to_numpy(dtype=float),
            "person_id": relabeled["person_id"].to_numpy(dtype=int),
            "u": u,
            "v": v,
        }
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def bump_homography(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    axis: str,
    angle_deg: float,
    seed: int,
) -> np.ndarray:
    """World-to-image H after a small camera-frame rotation (or H* if none/0°)."""
    if axis == "none" or float(angle_deg) <= 0.0:
        return ground_homography(K, R, t)
    rng = np.random.default_rng(int(seed))
    sign = float(rng.choice([-1.0, 1.0]))
    rvec = AXIS_VECTORS[axis] * (sign * math.radians(float(angle_deg)))
    R_delta, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    R_bumped = R_delta @ np.asarray(R, dtype=np.float64)
    t_bumped = R_delta @ np.asarray(t, dtype=np.float64).reshape(3, 1)
    return ground_homography(K, R_bumped, t_bumped)


def clicked_pixels(
    exact_pixels: np.ndarray,
    eps_px: float,
    outlier_px: float,
    seed: int,
) -> tuple[np.ndarray, int | None]:
    """Perturb every point by eps_px; optionally shove one point by outlier_px."""
    rng = np.random.default_rng(int(seed))
    clicked = perturb_pixels(exact_pixels, float(eps_px), rng)
    outlier_index: int | None = None
    if float(outlier_px) > 0.0:
        outlier_index = int(rng.integers(0, len(exact_pixels)))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        clicked[outlier_index] = clicked[outlier_index] + float(outlier_px) * np.array(
            [math.cos(angle), math.sin(angle)], dtype=np.float64
        )
    return clicked, outlier_index


def estimate_H_system(
    pixels: np.ndarray,
    world_pts: np.ndarray,
    estimator: str,
) -> np.ndarray:
    if estimator == "RANSAC":
        return ransac_homography_exhaustive(pixels, world_pts, threshold_px=TRANSFER_THRESH_PX)["H"]
    return normalized_dlt(pixels, world_pts)


def detect_all(tracks: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    zone = detect_zone_events(tracks, cfg["zones"], cfg)
    dwell = detect_dwell_events(zone, cfg)
    prox = detect_proximity_events(tracks, cfg)
    return pd.concat([zone, dwell, prox], ignore_index=True)


def _rgb(color: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in color)  # type: ignore[return-value]


def _bgr(color: tuple[float, float, float]) -> tuple[int, int, int]:
    r, g, b = _rgb(color)
    return b, g, r


# ---------------------------------------------------------------------------
# Scenario compute (cached — rerun only when calibration settings change)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Recomputing calibration / events…")
def compute_scenario(
    cam_id: str,
    track_source: str,
    n_points: int,
    eps_px: float,
    seed: int,
    outlier_px: float,
    estimator: str,
    bump_axis: str,
    bump_angle: float,
) -> dict[str, Any]:
    """Estimate H, lift feet for the whole clip, detect events, and score."""
    cfg = cached_config(cam_id)
    annotations = cached_annotations(cam_id)
    K, R, t, H_star_w2i, H_star_i2w = cached_pose(cam_id)
    world_pts, exact_pixels = cached_calib_points(cam_id, n_points)
    gt_events = cached_gt_events(cam_id)

    clicked, outlier_index = clicked_pixels(exact_pixels, eps_px, outlier_px, seed)
    H_system = estimate_H_system(clicked, world_pts, estimator)
    H_project = bump_homography(K, R, t, bump_axis, bump_angle, seed)

    visible = annotations.loc[annotations["visible"], ["time_sec", "person_id", "X_m", "Y_m", "frame_idx"]].copy()
    true_xy = visible[["X_m", "Y_m"]].to_numpy(dtype=np.float64)

    if track_source == "ground_truth":
        # True image feet under (possibly bumped) pose; lift with the system H.
        true_uv = world_to_image(H_project, true_xy)
        est_xy = image_to_world(H_system, true_uv)
        est_uv = true_uv.copy()
        persons = visible["person_id"].to_numpy(dtype=int)
        times = visible["time_sec"].to_numpy(dtype=float)
        frame_idxs = visible["frame_idx"].to_numpy(dtype=int)
        true_xy_aligned = true_xy
    else:
        feet = cached_pred_feet(track_source)
        # Align predicted feet to GT by (time, person) for error / green markers.
        est_uv = feet[["u", "v"]].to_numpy(dtype=np.float64)
        est_xy = image_to_world(H_system, est_uv)
        pred = pd.DataFrame(
            {
                "time_sec": feet["time_sec"].to_numpy(dtype=float),
                "person_id": feet["person_id"].to_numpy(dtype=int),
                "X_est": est_xy[:, 0],
                "Y_est": est_xy[:, 1],
                "u": est_uv[:, 0],
                "v": est_uv[:, 1],
            }
        )
        # Round times to match annotation grid (2 fps).
        pred["_t"] = np.round(pred["time_sec"] * 2.0) / 2.0
        gt = visible.copy()
        gt["_t"] = np.round(gt["time_sec"] * 2.0) / 2.0
        merged = pred.merge(
            gt[["_t", "person_id", "X_m", "Y_m", "frame_idx", "time_sec"]],
            on=["_t", "person_id"],
            how="inner",
            suffixes=("", "_gt"),
        )
        if merged.empty:
            raise RuntimeError("No overlapping predicted/GT feet after idmap matching.")
        true_xy_aligned = merged[["X_m", "Y_m"]].to_numpy(dtype=np.float64)
        est_xy = merged[["X_est", "Y_est"]].to_numpy(dtype=np.float64)
        est_uv = merged[["u", "v"]].to_numpy(dtype=np.float64)
        true_uv = world_to_image(H_star_w2i, true_xy_aligned)
        persons = merged["person_id"].to_numpy(dtype=int)
        times = merged["time_sec"].to_numpy(dtype=float)
        frame_idxs = merged["frame_idx"].to_numpy(dtype=int)

    positions = pd.DataFrame(
        {
            "frame_idx": frame_idxs,
            "time_sec": times,
            "person_id": persons,
            "X_true": true_xy_aligned[:, 0],
            "Y_true": true_xy_aligned[:, 1],
            "X_est": np.round(est_xy[:, 0], 6),
            "Y_est": np.round(est_xy[:, 1], 6),
            "u_true": true_uv[:, 0],
            "v_true": true_uv[:, 1],
            "u_est": est_uv[:, 0],
            "v_est": est_uv[:, 1],
        }
    )
    err = np.linalg.norm(
        positions[["X_est", "Y_est"]].to_numpy() - positions[["X_true", "Y_true"]].to_numpy(),
        axis=1,
    )
    mean_pos_error = float(np.mean(err)) if len(err) else float("nan")

    est_tracks = positions[["time_sec", "person_id", "X_est", "Y_est"]].rename(
        columns={"X_est": "X_m", "Y_est": "Y_m"}
    )
    pred_events = detect_all(est_tracks, cfg)
    scores, _matches = score_events(gt_events, pred_events, cfg)
    score_all = scores.loc[scores["zone_id"] == "all", ["event_type", "precision", "recall", "f1"]].copy()

    return {
        "positions": positions,
        "pred_events": pred_events,
        "gt_events": gt_events,
        "scores": score_all,
        "mean_pos_error": mean_pos_error,
        "world_pts": world_pts,
        "exact_pixels": exact_pixels,
        "clicked_pixels": clicked,
        "outlier_index": outlier_index,
        "H_star_w2i": H_star_w2i,
        "cfg": cfg,
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_camera_panel(
    image_bgr: np.ndarray,
    scenario: dict[str, Any],
    frame_idx: int,
) -> np.ndarray:
    canvas = image_bgr.copy()
    H = scenario["H_star_w2i"]
    cfg = scenario["cfg"]
    # Zones with true H*.
    for zone in cfg["zones"]:
        poly = np.asarray(zone["polygon"], dtype=np.float64)
        uv = world_to_image(H, poly)
        pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        color = _bgr(ZONE_COLORS.get(str(zone["id"]), (0.8, 0.8, 0.8)))
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)
        cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)

    # Calibration points: true green, clicked red.
    exact = np.round(scenario["exact_pixels"]).astype(np.int32)
    clicked = np.round(scenario["clicked_pixels"]).astype(np.int32)
    for i, (t_pt, c_pt) in enumerate(zip(exact, clicked)):
        cv2.circle(canvas, (int(t_pt[0]), int(t_pt[1])), 7, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(c_pt[0]), int(c_pt[1])), 7, (0, 0, 220), 2, cv2.LINE_AA)
        cv2.line(
            canvas,
            (int(t_pt[0]), int(t_pt[1])),
            (int(c_pt[0]), int(c_pt[1])),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if scenario["outlier_index"] is not None and i == scenario["outlier_index"]:
            cv2.circle(canvas, (int(c_pt[0]), int(c_pt[1])), 12, (0, 0, 255), 2, cv2.LINE_AA)

    # Feet for this frame.
    frame_rows = scenario["positions"].loc[scenario["positions"]["frame_idx"] == int(frame_idx)]
    for row in frame_rows.itertuples(index=False):
        tu, tv = int(round(row.u_true)), int(round(row.v_true))
        eu, ev = int(round(row.u_est)), int(round(row.v_est))
        cv2.circle(canvas, (tu, tv), 5, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (eu, ev), 5, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.line(canvas, (tu, tv), (eu, ev), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{cfg['camera']}  frame {frame_idx}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{cfg['camera']}  frame {frame_idx}",
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_birdseye(scenario: dict[str, Any], frame_idx: int, cam_id: str) -> plt.Figure:
    cfg = scenario["cfg"]
    positions = scenario["positions"]
    frame_rows = positions.loc[positions["frame_idx"] == int(frame_idx)]
    t_now = float(frame_rows["time_sec"].iloc[0]) if len(frame_rows) else frame_idx / 2.0

    fig, ax = plt.subplots(figsize=(5.2, 6.2))
    for zone in cfg["zones"]:
        poly = np.asarray(zone["polygon"], dtype=np.float64)
        color = ZONE_COLORS.get(str(zone["id"]), (0.7, 0.7, 0.7))
        closed = np.vstack([poly, poly[0]])
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.25)
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        center = poly.mean(axis=0)
        ax.text(center[0], center[1], str(zone["name"]), ha="center", va="center", fontsize=8)

    if len(frame_rows):
        xt = frame_rows["X_true"].to_numpy()
        yt = frame_rows["Y_true"].to_numpy()
        xe = frame_rows["X_est"].to_numpy()
        ye = frame_rows["Y_est"].to_numpy()
        ax.scatter(xt, yt, c="green", s=28, zorder=3, label="true")
        ax.scatter(xe, ye, c="red", s=28, zorder=3, label="estimated")
        for x0, y0, x1, y1 in zip(xt, yt, xe, ye):
            ax.plot([x0, x1], [y0, y1], color="0.45", linewidth=0.8, zorder=2)

    # Active proximity alarms (estimated).
    prox = scenario["pred_events"]
    if not prox.empty:
        active = prox.loc[
            (prox["event_type"] == "proximity")
            & (prox["start_time"] <= t_now)
            & (prox["end_time"] >= t_now)
        ]
        est = frame_rows.set_index("person_id")
        for row in active.itertuples(index=False):
            a, b = int(row.person_a), int(row.person_b)
            if a not in est.index or b not in est.index:
                continue
            pa = est.loc[a]
            pb = est.loc[b]
            if isinstance(pa, pd.DataFrame):
                pa = pa.iloc[0]
            if isinstance(pb, pd.DataFrame):
                pb = pb.iloc[0]
            mx = 0.5 * (float(pa["X_est"]) + float(pb["X_est"]))
            my = 0.5 * (float(pa["Y_est"]) + float(pb["Y_est"]))
            circle = plt.Circle((mx, my), PROX_CIRCLE_R_M, fill=False, color="red", linewidth=1.8, zorder=4)
            ax.add_patch(circle)

    cam_xy = camera_ground_xy(cam_id)
    ax.scatter([cam_xy[0]], [cam_xy[1]], c="black", marker="^", s=60, zorder=5, label="camera")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Bird's-eye (current frame)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    fig.tight_layout()
    return fig


def draw_timeline(scenario: dict[str, Any], frame_idx: int) -> plt.Figure:
    gt = scenario["gt_events"]
    pred = scenario["pred_events"]
    t_now = frame_idx / 2.0
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True)
    for ax, event_type in zip(axes, ("zone", "dwell", "proximity")):
        for source, table, y0, alpha in (
            ("GT", gt, 0.55, 0.85),
            ("pred", pred, 0.15, 0.55),
        ):
            if table.empty:
                continue
            subset = table.loc[table["event_type"] == event_type]
            color = EVENT_COLORS[event_type]
            for row in subset.itertuples(index=False):
                ax.broken_barh(
                    [(float(row.start_time), max(float(row.duration), 0.25))],
                    (y0, 0.3),
                    facecolors=color,
                    alpha=alpha,
                    edgecolors="none",
                )
        ax.axvline(t_now, color="black", linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.3, 0.7])
        ax.set_yticklabels(["pred", "GT"])
        ax.set_ylabel(event_type)
        ax.grid(True, axis="x", alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title("Event timeline (whole clip)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


def sidebar() -> dict[str, Any]:
    st.sidebar.header("Settings")
    cam_id = st.sidebar.selectbox("Camera", ["C1", "C6"], index=0)

    track_labels = ["ground truth"]
    if cam_id == "C1":
        track_labels.extend(PRED_TRACKS.keys())
    track_label = st.sidebar.selectbox("Tracks", track_labels, index=0)
    track_source = "ground_truth" if track_label == "ground truth" else PRED_TRACKS[track_label]
    if cam_id == "C6" and track_source != "ground_truth":
        st.sidebar.warning("Predicted tracks are only available for C1.")
        track_source = "ground_truth"

    st.sidebar.subheader("Calibration")
    n_points = st.sidebar.selectbox("Number of points", [4, 12], index=0)
    eps_px = st.sidebar.slider("Click noise ε (px)", 0.0, 10.0, 0.0, 0.5)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)
    outlier_px = st.sidebar.slider("One bad point (px)", 0.0, 50.0, 0.0, 1.0)
    estimator = st.sidebar.selectbox("Estimator", ["DLT", "RANSAC"], index=0)

    st.sidebar.subheader("Camera bump")
    bump_axis = st.sidebar.selectbox("Axis", ["none", "pitch", "yaw", "roll"], index=0)
    bump_angle = st.sidebar.slider("Angle (deg)", 0.0, 1.0, 0.0, 0.05)

    st.sidebar.subheader("Frame")
    frames = cached_frames(cam_id)
    n_frames = int(frames["frame_idx"].max())
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    frame_idx = st.sidebar.slider(
        "Frame",
        0,
        n_frames,
        int(st.session_state.frame_idx),
        1,
    )
    st.session_state.frame_idx = int(frame_idx)

    cols = st.sidebar.columns(2)
    playing = cols[0].checkbox("Play", value=False)
    if cols[1].button("Reset"):
        st.session_state.frame_idx = 0
        st.rerun()

    return {
        "cam_id": normalize_cam_id(cam_id),
        "track_source": track_source,
        "n_points": int(n_points),
        "eps_px": float(eps_px),
        "seed": int(seed),
        "outlier_px": float(outlier_px),
        "estimator": estimator,
        "bump_axis": bump_axis,
        "bump_angle": float(bump_angle),
        "frame_idx": int(st.session_state.frame_idx),
        "playing": playing,
        "n_frames": n_frames,
    }


def main() -> None:
    st.set_page_config(page_title="Calibration-Aware Safety Events", layout="wide")
    st.title("Calibration-Aware Safety Events")
    st.caption("Zones use true H*. Feet and events use the estimated homography (and optional camera bump).")

    opts = sidebar()
    scenario = compute_scenario(
        opts["cam_id"],
        opts["track_source"],
        opts["n_points"],
        opts["eps_px"],
        opts["seed"],
        opts["outlier_px"],
        opts["estimator"],
        opts["bump_axis"],
        opts["bump_angle"],
    )

    image = cached_image(opts["cam_id"], opts["frame_idx"])
    cam_rgb = draw_camera_panel(image, scenario, opts["frame_idx"])
    bird = draw_birdseye(scenario, opts["frame_idx"], opts["cam_id"])
    timeline = draw_timeline(scenario, opts["frame_idx"])

    left, mid, right = st.columns([1.15, 0.85, 1.2])
    with left:
        st.subheader("Camera")
        st.image(cam_rgb, use_container_width=True)
    with mid:
        st.subheader("Bird's-eye")
        st.pyplot(bird, clear_figure=True)
    with right:
        st.subheader("Events")
        st.pyplot(timeline, clear_figure=True)
    plt.close(bird)
    plt.close(timeline)

    st.subheader("Scores (whole clip)")
    scores = scenario["scores"].copy()
    scores["precision"] = scores["precision"].map(lambda x: f"{x:.3f}")
    scores["recall"] = scores["recall"].map(lambda x: f"{x:.3f}")
    scores["f1"] = scores["f1"].map(lambda x: f"{x:.3f}")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(scores.set_index("event_type"), use_container_width=True)
    with c2:
        st.metric("Mean position error", f"{scenario['mean_pos_error']:.3f} m")
        st.caption(
            f"{opts['n_points']} pts · ε={opts['eps_px']:g} px · "
            f"outlier={opts['outlier_px']:g} px · {opts['estimator']} · "
            f"bump={opts['bump_axis']} {opts['bump_angle']:g}°"
        )

    if opts["playing"]:
        time.sleep(0.15)
        st.session_state.frame_idx = (opts["frame_idx"] + 1) % (opts["n_frames"] + 1)
        st.rerun()


if __name__ == "__main__":
    main()
