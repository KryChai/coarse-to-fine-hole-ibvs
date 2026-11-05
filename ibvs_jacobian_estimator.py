"""Hardware-independent FD initialization, gated Broyden and bounded DLS.

J maps actual base-frame TCP XY millimetres to raw-image UV pixels.
No robot commands or camera I/O belong in this module.
"""

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JacobianSettings:
    mode: str = "online"  # online | fixed (DLS comparison) | calibrate
    hole_diameter_mm: float | None = None
    probe_mm: float | None = None  # Explicit field-approved distance, not the servo limit.
    max_probe_mm: float | None = None  # Field-approved excursion from the calibration origin.
    camera_id: str = ""  # Physical camera identity, not just its enumerated index.
    camera_index: int = 0
    calibration_path: str = "calibration/ibvs_jacobian.json"
    image_size: tuple = (2560, 1440)
    target_uv: tuple = (1280.0, 720.0)
    sample_frames: int = 5
    gain: float = 0.4
    damping_ratio: float = 0.03
    max_step_mm: float = 1.0
    beta: float = 0.3
    min_motion_mm: float = 0.05
    min_pixel_motion_px: float = 2.0
    huber_px: float = 3.0
    max_residual_px: float = 30.0
    association_px: float = 40.0
    max_condition: float = 100.0
    min_singular: float = 0.01
    max_singular: float = 10000.0
    max_relative_update: float = 0.3
    max_reference_change: float = 0.5
    min_confidence: float = 0.5
    max_orientation_change_deg: float = 0.1
    max_z_change_mm: float = 0.1
    height_tolerance_mm: float = 2.0
    stationary_tolerance_mm: float = 0.02
    arrival_tolerance_mm: float = 0.1
    max_sample_age_s: float = 30.0
    max_pair_interval_s: float = 60.0
    align_tol_px: float = 2.0
    align_tol_mm: float = 0.05
    max_iterations: int = 100
    max_detection_failures: int = 3
    max_non_improving_steps: int = 5
    max_rejected_updates: int = 8

    def validate(self):
        if self.mode not in ("online", "fixed", "calibrate"):
            raise ValueError("jacobian mode must be online, fixed or calibrate")
        exempt = {"mode", "camera_id", "calibration_path", "image_size", "target_uv",
                  "hole_diameter_mm", "probe_mm", "max_probe_mm", "camera_index"}
        for key, value in asdict(self).items():
            if key not in exempt and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("sample_frames", "max_iterations", "max_detection_failures",
                    "max_non_improving_steps", "max_rejected_updates"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.beta > 1 or self.gain > 1 or self.min_confidence > 1:
            raise ValueError("beta, gain and confidence must not exceed 1")
        size, target = vector(self.image_size, 2), vector(self.target_uv, 2)
        if np.any(size <= 0) or np.any(size != np.floor(size)):
            raise ValueError("invalid image_size")
        if np.any(target < 0) or np.any(target >= size):
            raise ValueError("target_uv outside raw image")
        if self.max_step_mm > 1 or self.min_singular >= self.max_singular:
            raise ValueError("max_step_mm must be <= 1; singular value bounds invalid")
        if not isinstance(self.camera_index, int) or self.camera_index < 0:
            raise ValueError("camera_index must be a nonnegative integer")
        if self.max_probe_mm is not None and (not np.isfinite(self.max_probe_mm) or self.max_probe_mm <= 0):
            raise ValueError("max_probe_mm must be finite and positive")
        if self.mode != "fixed" or self.camera_id:
            if not isinstance(self.camera_id, str) or not self.camera_id.strip():
                raise ValueError("请在配置中填写物理相机 camera_id")
            if self.hole_diameter_mm is None or not np.isfinite(self.hole_diameter_mm) or self.hole_diameter_mm <= 0:
                raise ValueError("请在配置中填写真实 hole_diameter_mm")
            if self.sample_frames < 3:
                raise ValueError("online/calibrate requires at least three stationary frames")
        if self.mode == "calibrate":
            if self.probe_mm is None or not np.isfinite(self.probe_mm) or not 0 < self.probe_mm <= self.max_step_mm:
                raise ValueError("请填写现场确认的 probe_mm，且不超过 max_step_mm")
            if self.probe_mm < 3 * self.min_motion_mm:
                raise ValueError("probe_mm must be >= 3 * min_motion_mm")
            if self.max_probe_mm is not None and self.probe_mm > self.max_probe_mm:
                raise ValueError("probe_mm exceeds field-approved max_probe_mm")
        if not str(self.calibration_path).strip():
            raise ValueError("calibration_path is empty")
        return self


def vector(value, length):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"expected {length} finite values")
    return result


def angle_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def validate_matrix(J, settings):
    matrix = np.asarray(J, dtype=float)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("J must be a finite 2x2 matrix")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] < settings.min_singular or singular[0] > settings.max_singular:
        raise ValueError("J singular values outside permitted scale")
    if singular[0] / singular[-1] > settings.max_condition:
        raise ValueError("J is ill-conditioned")
    return matrix.copy()


def estimate_initial_jacobian(samples, settings):
    """Fit an intercept plus J to absolute samples (no noisy shared reference)."""
    Q = np.asarray([vector(s["q"], 2) for s in samples])
    S = np.asarray([vector(s["s"], 2) for s in samples])
    if len(Q) < 3:
        raise ValueError("at least three independent probe positions required")
    Qc, Sc = Q - Q.mean(axis=0), S - S.mean(axis=0)
    singular = np.linalg.svd(Qc, compute_uv=False)
    if singular[-1] < settings.min_motion_mm or singular[0] / singular[-1] > 20:
        raise ValueError("probe data lack sufficient independent XY excitation")
    J = validate_matrix(np.linalg.lstsq(Qc, Sc, rcond=None)[0].T, settings)
    noise = max(float(s.get("noise_px", 0)) for s in samples)
    span = np.min(np.linalg.svd(Qc @ J.T, compute_uv=False))
    if span < max(settings.min_pixel_motion_px, 3 * np.sqrt(2) * noise):
        raise ValueError("probe pixel displacement below measured noise floor")
    residual = Sc - Qc @ J.T
    if np.max(np.linalg.norm(residual, axis=1)) > max(settings.huber_px, 3 * noise):
        raise ValueError("probe fit residual too large; check association and motion")
    return J


class OnlineJacobianEstimator:
    def __init__(self, J0, settings=None):
        self.settings = settings or JacobianSettings(mode="fixed")
        self.settings.validate()
        self.J = validate_matrix(J0, self.settings)
        self.J_reference = self.J.copy()
        self.update_count = self.reject_count = 0

    def get_jacobian(self):
        return self.J.copy()

    def update(self, before, after, *, robot_settled, now):
        cfg = self.settings
        old = self.J.copy()
        audit = {
            "J_before": old.tolist(), "J_candidate": None,
            "dq": None, "ds": None, "predicted_ds_before": None,
            "residual": None, "residual_norm": None, "weight": None,
            "denominator": None, "candidate_delta_J": None,
            "candidate_relative_update": None, "condition_before": float(np.linalg.cond(old)),
            "condition_candidate": None, "singular_values_candidate": None,
            "beta": cfg.beta, "noise_px": None, "pixel_motion_threshold": None,
            "gates": {key: getattr(cfg, key) for key in (
                "min_motion_mm", "min_pixel_motion_px", "max_residual_px", "huber_px",
                "max_condition", "max_relative_update", "max_reference_change",
                "max_orientation_change_deg", "max_z_change_mm", "max_sample_age_s",
                "max_pair_interval_s", "arrival_tolerance_mm", "max_step_mm", "min_confidence",
                "min_singular", "max_singular")},
        }

        def finish(accepted, reason):
            actual = self.J.copy()
            dq, ds = audit["dq"], audit["ds"]
            if dq is not None and ds is not None:
                prediction = actual @ np.asarray(dq)
                audit["predicted_ds_after"] = prediction.tolist()
                audit["residual_after"] = (np.asarray(ds) - prediction).tolist()
                audit["residual_after_norm"] = float(np.linalg.norm(audit["residual_after"]))
            change = actual - old
            audit.update(accepted=accepted, reason=reason, J=actual.tolist(),
                         applied_delta_J=change.tolist(),
                         relative_update=float(np.linalg.norm(change) / np.linalg.norm(old)),
                         reference_drift=float(np.linalg.norm(actual - self.J_reference) / np.linalg.norm(self.J_reference)),
                         condition=float(np.linalg.cond(actual)),
                         singular_values=np.linalg.svd(actual, compute_uv=False).tolist(),
                         update_count=self.update_count, reject_count=self.reject_count)
            return audit

        def reject(reason):
            self.reject_count += 1
            return finish(False, reason)

        try:
            # Compute available diagnostics even if a subsequent gate rejects learning.
            qb, qa = vector(before["tcp"], 6), vector(after["tcp"], 6)
            dq = qa[:2] - qb[:2]
            ds = vector(after["center"], 2) - vector(before["center"], 2)
            residual = ds - old @ dq
            audit.update(dq=dq.tolist(), ds=ds.tolist(),
                         predicted_ds_before=(old @ dq).tolist(), residual=residual.tolist(),
                         residual_norm=float(np.linalg.norm(residual)),
                         denominator=float(np.dot(dq, dq)))
            if not robot_settled:
                return reject("robot_not_settled")
            for sample in (before, after):
                if not sample["valid"] or not cfg.min_confidence <= sample["conf"] <= 1:
                    return reject("invalid_detection")
                if tuple(sample["image_size"]) != tuple(cfg.image_size):
                    return reject("resolution_changed")
                if not np.isfinite(sample["timestamp"]) or not 0 <= now - sample["timestamp"] <= cfg.max_sample_age_s:
                    return reject("stale_image")
                if not np.isfinite(sample["noise_px"]) or sample["noise_px"] < 0:
                    return reject("invalid_noise")
            dt = after["timestamp"] - before["timestamp"]
            if not 0 < dt <= cfg.max_pair_interval_s or after["frame_id"] <= before["frame_id"]:
                return reject("unpaired_frames")
            qb, qa = vector(before["tcp"], 6), vector(after["tcp"], 6)
            if abs(qa[2] - qb[2]) > cfg.max_z_change_mm:
                return reject("height_changed")
            if np.max(np.abs(angle_difference(qa[3:], qb[3:]))) > cfg.max_orientation_change_deg:
                return reject("orientation_changed")
            dq = qa[:2] - qb[:2]
            ds = vector(after["center"], 2) - vector(before["center"], 2)
            if np.linalg.norm(dq) < cfg.min_motion_mm:
                return reject("motion_too_small")
            if np.linalg.norm(dq) > cfg.max_step_mm + cfg.arrival_tolerance_mm:
                return reject("unexpected_motion")
            if "commanded_tcp" in after and np.linalg.norm(qa[:3] - vector(after["commanded_tcp"], 6)[:3]) > cfg.arrival_tolerance_mm:
                return reject("command_feedback_mismatch")
            noise = max(before["noise_px"], after["noise_px"])
            audit["noise_px"] = float(noise)
            audit["pixel_motion_threshold"] = max(cfg.min_pixel_motion_px, 3 * np.sqrt(2) * noise)
            if np.linalg.norm(ds) < max(cfg.min_pixel_motion_px, 3 * np.sqrt(2) * noise):
                return reject("visual_change_too_small")
            residual = ds - old @ dq
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm > cfg.max_residual_px:
                return reject("residual_outlier")
            weight = min(1.0, cfg.huber_px / max(residual_norm, 1e-12))
            audit["weight"] = weight
            change = cfg.beta * weight * np.outer(residual, dq) / np.dot(dq, dq)
            candidate = old + change
            audit.update(J_candidate=candidate.tolist(), candidate_delta_J=change.tolist(),
                         candidate_relative_update=float(np.linalg.norm(change) / np.linalg.norm(old)),
                         condition_candidate=float(np.linalg.cond(candidate)),
                         singular_values_candidate=np.linalg.svd(candidate, compute_uv=False).tolist())
            if np.linalg.norm(change) > cfg.max_relative_update * np.linalg.norm(old):
                return reject("update_too_large")
            candidate = validate_matrix(old + change, cfg)
            if np.linalg.det(candidate) * np.linalg.det(self.J_reference) <= 0:
                return reject("jacobian_sign_flip")
            # A determinant alone cannot detect a simultaneous reversal of both columns.
            if any(np.dot(candidate[:, i], self.J_reference[:, i]) <= 0 for i in range(2)):
                return reject("axis_direction_flip")
            if np.linalg.norm(candidate - self.J_reference) > cfg.max_reference_change * np.linalg.norm(self.J_reference):
                return reject("reference_drift")
        except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
            audit["validation_error"] = str(error)
            return reject("invalid_sample_or_matrix")
        self.J = candidate
        self.update_count += 1
        return finish(True, "updated")


def calculate_dls_step(J, error, settings, *, return_diagnostics=False):
    matrix = validate_matrix(J, settings)
    e = vector(error, 2)
    U, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
    mu = settings.damping_ratio * singular[0]
    # SVD avoids squaring the condition number in normal equations.
    step = -settings.gain * Vt.T @ ((singular / (singular**2 + mu**2)) * (U.T @ e))
    raw_step = step.copy()
    norm = np.linalg.norm(step)
    if norm > settings.max_step_mm:
        step *= settings.max_step_mm / norm
    if return_diagnostics:
        return step, {"gain": settings.gain, "damping_ratio": settings.damping_ratio,
                      "mu": float(mu), "singular_values": singular.tolist(),
                      "error_px": e.tolist(), "raw_delta_xy": raw_step.tolist(),
                      "limited_delta_xy": step.tolist(), "max_step_mm": settings.max_step_mm,
                      "limit_scale": min(1., settings.max_step_mm / max(float(norm), 1e-12))}
    return step


def save_calibration(path, J, metadata):
    """Atomic replacement; callers keep initial and learned matrices in separate files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump({"version": 1, "J": np.asarray(J).tolist(), "metadata": metadata},
                      stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_calibration(path, settings, *, tool=None, tcp=None):
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("version") != 1:
        raise ValueError("unsupported calibration version")
    meta = payload["metadata"]
    expected = {"camera_id": settings.camera_id, "image_size": list(settings.image_size),
                "target_uv": list(settings.target_uv), "hole_diameter_mm": settings.hole_diameter_mm,
                "camera_index": settings.camera_index}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"calibration mismatch: {key}; recalibrate")
    vector(meta["reference_tcp"], 6)
    if tool is not None and meta["tool"] != tool:
        raise ValueError("calibration TCP tool changed; recalibrate")
    if tcp is not None:
        validate_work_pose(tcp, meta["reference_tcp"], settings)
    return validate_matrix(payload["J"], settings), meta


def validate_work_pose(tcp, reference, settings):
    q, ref = vector(tcp, 6), vector(reference, 6)
    if abs(q[2] - ref[2]) > settings.height_tolerance_mm:
        raise ValueError("working height outside calibration range")
    if np.max(np.abs(angle_difference(q[3:], ref[3:]))) > settings.max_orientation_change_deg:
        raise ValueError("working orientation outside calibration range")
