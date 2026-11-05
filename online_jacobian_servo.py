"""Shared, callback-driven calibration and online servo state machine."""

from datetime import datetime
import json
from pathlib import Path
import time
import os

import numpy as np

from ibvs_jacobian_estimator import (
    OnlineJacobianEstimator, angle_difference, calculate_dls_step,
    estimate_initial_jacobian, load_calibration, save_calibration,
    validate_work_pose, vector,
)


def validate_observation(sample, settings, now):
    vector(sample["tcp"], 6)
    center = vector(sample["center"], 2)
    if (sample.get("valid") is not True or sample.get("settled") is not True
            or not settings.min_confidence <= sample["conf"] <= 1
            or tuple(sample["image_size"]) != tuple(settings.image_size)
            or not np.isfinite(sample["timestamp"])
            or not 0 <= now - sample["timestamp"] <= settings.max_sample_age_s
            or not isinstance(sample["frame_id"], int)
            or sample["frame_id"] < 1
            or not np.isfinite(sample["noise_px"]) or sample["noise_px"] < 0
            or sample["noise_px"] > settings.align_tol_px
            or np.any(center < 0) or np.any(center >= np.asarray(settings.image_size))):
        raise RuntimeError("不可信的静止视觉观测，禁止控制或标定")


class JacobianRunLog:
    def __init__(self, directory, source="unspecified"):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.path = path / ("jacobian_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}.jsonl")
        self.source = source
        self.sequence = 0

    def __call__(self, event, **data):
        invalid = []
        def clean(value, path=""):
            if isinstance(value, (np.ndarray,)):
                return clean(value.tolist(), path)
            if isinstance(value, np.generic):
                return clean(value.item(), path)
            if isinstance(value, dict):
                return {str(k): clean(v, f"{path}.{k}") for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v, f"{path}[{i}]") for i, v in enumerate(value)]
            if isinstance(value, float) and not np.isfinite(value):
                invalid.append({"path": path, "value": repr(value)})
                return None
            if isinstance(value, Path):
                return str(value)
            return value
        record = clean({"schema_version": 2, "sequence": self.sequence + 1,
                        "source": self.source, "event": event, "time": time.monotonic(), **data})
        if invalid:
            record["nonfinite_values"] = invalid
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.sequence += 1
        from run_logging import write_run_event
        write_run_event("[JacobianAudit] " + line)

    def export_figures(self):
        """Plot only after robot control returns; raw logging never depends on matplotlib."""
        try:
            from plot_online_jacobian import export_figures
            manifest = export_figures(self.path)
            self("figures_exported", manifest=str(manifest))
            print(f"[Online J] 中英文独立图像: {manifest.parent}")
            return manifest
        except Exception as error:
            self("figure_export_failed", message=str(error))
            print(f"[Online J] 绘图失败，完整日志已保留，可离线重绘: {error}")
            return None


def run_calibration(settings, *, measure, move, tool, log, clock=time.monotonic):
    """Only explicitly selected calibration mode executes the probe motions.

    measure(expected_uv) returns a stationary multi-frame observation or None.
    move(target_tcp) must check SDK response, stopped state and actual arrival.
    """
    settings.validate()
    if settings.mode != "calibrate":
        raise ValueError("probe motions require calibrate mode")
    if Path(settings.calibration_path).exists():
        # Preserve initial calibration rather than silently replacing a trusted file.
        raise FileExistsError("标定文件已存在；请使用新的 calibration_path 保存新标定")
    origin = measure(None)
    if origin is None:
        raise RuntimeError("标定初始检测失败，不执行探测")
    validate_observation(origin, settings, clock())
    q0 = vector(origin["tcp"], 6)
    center = vector(origin["center"], 2)
    if origin.get("bbox_wh") is not None:
        scale = vector(origin["bbox_wh"], 2) / settings.hole_diameter_mm
        if np.any(scale <= 0):
            raise ValueError("invalid bbox scale prior")
        if np.min(scale) * settings.probe_mm < max(settings.min_pixel_motion_px, 3 * np.sqrt(2) * origin["noise_px"]):
            raise ValueError("孔径先验预测探测变化不足以超过视觉噪声，请重新选择探测距离")
        margin = min(np.min(center), np.min(np.asarray(settings.image_size) - center))
        if 2 * np.max(scale) * settings.probe_mm >= margin:
            raise ValueError("孔径先验预测探测可能离开视野")
    samples = [{"q": q0[:2].tolist(), "s": center.tolist(), "noise_px": origin["noise_px"]}]
    log("calibration_start", origin=origin, settings=vars(settings), tool=tool)
    latest = origin
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        target = q0.copy()
        target[axis] += sign * settings.probe_mm
        log("calibration_motion_requested", axis=axis, sign=sign, target_tcp=target.tolist())
        move(target.tolist())
        sample = measure(center)
        if sample is None:
            raise RuntimeError("探测后检测失败，停止标定；不执行自动回位或装配")
        validate_observation(sample, settings, clock())
        if sample["frame_id"] <= latest["frame_id"] or sample["timestamp"] <= latest["timestamp"]:
            raise RuntimeError("探测重复使用旧图像")
        actual = vector(sample["tcp"], 6)
        if np.linalg.norm(actual[:3] - target[:3]) > settings.arrival_tolerance_mm:
            raise RuntimeError("探测实际位移与目标不符")
        if abs(actual[2] - q0[2]) > settings.max_z_change_mm or np.max(np.abs(angle_difference(actual[3:], q0[3:]))) > settings.max_orientation_change_deg:
            raise RuntimeError("探测改变了高度或姿态")
        samples.append({"q": actual[:2].tolist(), "s": sample["center"], "noise_px": sample["noise_px"]})
        log("calibration_probe", target_tcp=target.tolist(), sample=sample)
        log("calibration_return_requested", target_tcp=q0.tolist())
        move(q0.tolist())
        returned = measure(center)
        if returned is None or np.linalg.norm(np.asarray(returned["center"]) - center) > max(settings.huber_px, 3 * origin["noise_px"], 3 * (returned or {}).get("noise_px", 0)):
            raise RuntimeError("探测回位图像不一致；检查目标关联和机械臂重复定位")
        validate_observation(returned, settings, clock())
        if returned["frame_id"] <= sample["frame_id"] or returned["timestamp"] <= sample["timestamp"]:
            raise RuntimeError("探测回位重复使用旧图像")
        if np.linalg.norm(vector(returned["tcp"], 6)[:3] - q0[:3]) > settings.arrival_tolerance_mm:
            raise RuntimeError("探测回位 TCP 不一致")
        log("calibration_return", sample=returned)
        latest = returned
    J = estimate_initial_jacobian(samples, settings)
    meta = {"camera_id": settings.camera_id, "camera_index": settings.camera_index, "image_size": list(settings.image_size),
            "target_uv": list(settings.target_uv), "hole_diameter_mm": settings.hole_diameter_mm,
            "tool": tool, "reference_tcp": q0.tolist(), "samples": samples,
            "noise_px": max(s["noise_px"] for s in samples), "created_at": datetime.now().isoformat()}
    bbox = origin.get("bbox_wh")
    if bbox is not None:
        prior = vector(bbox, 2) / settings.hole_diameter_mm
        meta["bbox_scale_prior_px_per_mm"] = prior.tolist()
        # Diameter informs probe planning/scale diagnostics, not a second Broyden factor.
        log("scale_prior_check", bbox_scale=prior.tolist(), fitted_column_norms=np.linalg.norm(J, axis=0).tolist())
    save_calibration(settings.calibration_path, J, meta)
    log("calibration_saved", path=str(settings.calibration_path), J=J.tolist(), metadata=meta)
    return J


def run_online_servo(settings, *, measure, move, tool, log, clock=time.monotonic):
    settings.validate()
    # Fail before any correction if calibration is absent, malformed or incompatible.
    J0, meta = load_calibration(settings.calibration_path, settings, tool=tool)
    estimator = OnlineJacobianEstimator(J0, settings)
    failures = non_improving = rejections = 0
    best_error = float("inf")
    pending = None
    hint = None
    latest_frame = 0
    latest_time = -float("inf")
    log("servo_start", settings=vars(settings), initial_J=J0.tolist(), metadata=meta)
    for iteration in range(settings.max_iterations):
        log("observation_requested", iteration=iteration,
            expected_uv=None if hint is None else np.asarray(hint).tolist())
        sample = measure(hint)
        if sample is None:
            failures += 1
            # A detection gap breaks pairing; never learn from a pre-gap sample.
            if pending is not None:
                log("update_pair_discarded", iteration=iteration, before=pending[0],
                    commanded_tcp=pending[1], reason="detection_gap")
            pending = None
            log("detection_failed", iteration=iteration, consecutive=failures)
            if failures >= settings.max_detection_failures:
                raise RuntimeError("连续拍照或孔洞检测失败，终止视觉对准")
            continue
        failures = 0
        log("observation_received", iteration=iteration, sample=sample)
        validate_observation(sample, settings, clock())
        if sample["frame_id"] <= latest_frame or sample["timestamp"] <= latest_time:
            raise RuntimeError("重复或乱序的视觉观测，禁止控制")
        latest_frame, latest_time = sample["frame_id"], sample["timestamp"]
        if hint is not None and np.linalg.norm(np.asarray(sample["center"]) - hint) > settings.association_px:
            raise RuntimeError("目标关联偏离预测位置，禁止后续运动")
        tcp, center = vector(sample["tcp"], 6), vector(sample["center"], 2)
        validate_work_pose(tcp, meta["reference_tcp"], settings)
        if pending is not None:
            before, commanded = pending
            sample["commanded_tcp"] = commanded
            if settings.mode == "online":
                log("jacobian_update_requested", iteration=iteration, before=before, after=sample,
                    J=estimator.get_jacobian().tolist())
                result = estimator.update(before, sample, robot_settled=sample["settled"], now=clock())
                log("jacobian_update", iteration=iteration, before=before, after=sample, **result)
                if result["accepted"]:
                    rejections = 0
                elif result["reason"] not in ("motion_too_small", "visual_change_too_small"):
                    rejections += 1
                    if rejections >= settings.max_rejected_updates:
                        raise RuntimeError("连续雅可比更新异常，终止视觉对准")
            else:
                log("fixed_pair", iteration=iteration, before=before, after=sample,
                    J=estimator.get_jacobian().tolist())
            log("motion_feedback", iteration=iteration, before=before, after=sample,
                commanded_tcp=commanded)
            pending = None
        J_used = estimator.get_jacobian()
        error = center - vector(settings.target_uv, 2)
        error_norm = float(np.linalg.norm(error))
        mm_error = float(np.linalg.norm(np.linalg.solve(J_used, error)))
        log("observation", iteration=iteration, sample=sample, error_px=error.tolist(),
            error_mm=mm_error, J=J_used.tolist())
        print(f"[Online J {iteration + 1}] error={error_norm:.3f} px, correction={mm_error:.4f} mm, updates={estimator.update_count}")
        if error_norm < settings.align_tol_px and mm_error < settings.align_tol_mm:
            # Diagnostic learned copy never overwrites or auto-replaces the initial J.
            if settings.mode == "online":
                save_calibration(str(settings.calibration_path) + ".online.json", J_used, meta)
            log("aligned_frozen", tcp=tcp.tolist(), J=J_used.tolist())
            return tcp.tolist()
        if error_norm < best_error - 0.01:
            best_error, non_improving = error_norm, 0
        else:
            non_improving += 1
            if non_improving >= settings.max_non_improving_steps:
                raise RuntimeError("视觉误差连续未改善，终止视觉对准")
        delta, control_diagnostics = calculate_dls_step(J_used, error, settings, return_diagnostics=True)
        target = tcp.copy()
        target[:2] += delta
        hint = center + J_used @ delta
        if np.any(hint < 0) or np.any(hint >= np.asarray(settings.image_size)):
            raise RuntimeError("预测孔中心超出视野，禁止运动")
        log("motion", iteration=iteration, J_used=J_used.tolist(), delta_xy=delta.tolist(),
            target_tcp=target.tolist(), predicted_uv=hint.tolist(), control=control_diagnostics)
        move(target.tolist())
        log("motion_command_completed", iteration=iteration, target_tcp=target.tolist())
        pending = (sample, target.tolist())
    raise RuntimeError("视觉对准达到最大迭代次数，终止装配")
