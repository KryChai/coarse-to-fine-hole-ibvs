"""Client for yolo_service/dev circle_refine and bounded second-stage XY servo.

No robot SDK imports. The caller supplies fresh captures and blocking, checked
TCP motion. Detection-box centres associate targets; fitted centres control XY.
"""

from dataclasses import dataclass
import json
import math
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests


class CircleProtocolError(RuntimeError):
    """Incompatible endpoint or malformed response; stop without retrying."""


class CircleMeasurementError(RuntimeError):
    """Missing, ambiguous or low-quality measurement; no motion is allowed."""


@dataclass(frozen=True)
class CircleServoSettings:
    tolerance_px: float = 1.0
    stable_frames: int = 3
    stability_px: float = 0.35
    gain: float = 0.1
    max_step_mm: float = 0.15
    association_px: float = 20.0
    ambiguity_px: float = 3.0
    max_iterations: int = 80
    max_failures: int = 3
    arrival_tolerance_mm: float = 0.1  # XY 位移范数到位容差，Z 使用独立阈值
    z_tolerance_mm: float = 1.0
    arrival_timeout_s: float = 3.0  # 阻塞运动返回后，按精调精度继续等待连续到位
    pose_tolerance_deg: float = 0.1
    # Fitting includes all detected holes and four perturbation fits per hole.
    timeout_s: float = 60.0
    allow_smoothing_sensitive: bool = False
    smoothing_max_shift_px: float = 2.0
    smoothing_confirmation_frames: int = 3

    def validate(self):
        values = {key: value for key, value in vars(self).items()
                  if key != "allow_smoothing_sensitive"}
        if type(self.allow_smoothing_sensitive) is not bool:
            raise ValueError("allow_smoothing_sensitive 必须为布尔值")
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("圆心精调参数必须为有限正数")
        if self.gain > 1 or self.max_step_mm > 1 or self.ambiguity_px >= self.association_px:
            raise ValueError("圆心精调增益、位移上限或关联门限无效")
        if not 1 < self.smoothing_max_shift_px <= 2 or self.smoothing_confirmation_frames < 3:
            raise ValueError("平滑容错上限须在 (1, 2] px，确认帧数至少为 3")
        for key in ("stable_frames", "max_iterations", "max_failures", "smoothing_confirmation_frames"):
            if type(values[key]) is not int:
                raise ValueError(f"{key} 必须是正整数")


def fine_log_summary(event, **data):
    """Compact console output; full response/telemetry is recorded separately."""
    if event == "fine_start":
        settings = data["settings"]
        return (f"开始精调：误差 < {settings['tolerance_px']:g} px，"
                f"连续 {settings['stable_frames']} 帧，散布 ≤ {settings['stability_px']:g} px，"
                f"增益={settings['gain']:g}，单步 ≤ {settings['max_step_mm']:g} mm，"
                f"平滑容错={'开启' if settings.get('allow_smoothing_sensitive') else '关闭'}")
    if event == "fine_measurement":
        u, v = data["center"]
        rmse = data["quality"].get("rmse_px")
        quality = f"，拟合 RMSE={rmse:.3f} px" if isinstance(rmse, (int, float)) else ""
        return (f"第 {data['iteration']} 轮：圆心=({u:.2f}, {v:.2f})，"
                f"误差={data['error_px']:.3f} px，半径={data['radius_px']:.2f} px{quality}")
    if event == "fine_control":
        dx, dy = data["delta_xy"]
        return f"第 {data['iteration']} 轮修正：dx={dx:.4f}，dy={dy:.4f} mm"
    if event == "fine_rejected":
        return f"第 {data['iteration']} 轮测量拒绝：{data['reason']}"
    if event == "fine_fallback_check":
        state = "确认通过" if data["confirmed"] else "等待稳定确认"
        return (f"平滑敏感性受限容错：尺度偏移={data['scale_center_shift_px']:.3f} px，"
                f"{data['frames']}/{data['required_frames']} 帧，{state}")
    if event == "fine_aligned":
        note = "，本次使用平滑敏感性容错" if data.get("smoothing_fallback_used") else ""
        return f"精调完成：第 {data['iteration']} 轮，稳定散布={data['spread_px']:.3f} px{note}"
    # Responses contain hundreds of edge/outline points and perturbation details.
    # Captures and motion feedback remain in events.jsonl without console copies.
    return None


def refinement_url(detection_url):
    """Only map plain YOLO detection to the matching per-hole refinement API."""
    parts = urlsplit(detection_url)
    for model in ("usb_hole7", "usb_hole11"):
        if parts.path == f"/api/{model}_detect":
            return urlunsplit((parts.scheme, parts.netloc, f"/api/{model}_circle_refine", "", ""))
    raise ValueError("精调自动路由要求 usb_hole7_detect 或 usb_hole11_detect")


def _vector(value, count):
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"expected {count} finite numbers")
    return array


def _point(value):
    return _vector([value["x"], value["y"]], 2)


def _quality_policy(hole, settings):
    """Keep the server verdict intact; permit only one explicit local exception."""
    quality = hole["fit_quality"]
    if (hole["status"] == "ok" and quality.get("valid") is True
            and quality.get("stability_checked") is True and quality.get("flags") == []):
        return "strict"
    eligible = (settings.allow_smoothing_sensitive and hole["status"] == "review"
                and quality.get("valid") is False and quality.get("stability_checked") is True
                and quality.get("flags") == ["smoothing_sensitive"])
    if not eligible:
        raise CircleMeasurementError(f"目标外圈拟合未通过质量检查: {quality.get('flags')}")
    # Mirror the other server-side quality bounds and fail closed on missing,
    # nonfinite or inconsistent values. A flag alone is insufficient evidence.
    bounds = {
        "scale_center_shift_px": (1.0, settings.smoothing_max_shift_px),
        "rmse_px": (0.0, 2.0), "relative_rmse_radius": (0.0, .03),
        "angular_coverage": (.85, 1.0), "inlier_fraction": (.65, 1.0),
        "candidate_center_spread_px": (0.0, 1.0),
        "bbox_jitter_max_center_shift_px": (0.0, 1.0),
        "ellipse_axis_ratio": (.95, 1.0),
    }
    for name, (lower, upper) in bounds.items():
        value = quality.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not lower <= value <= upper
                or (name == "scale_center_shift_px" and value <= lower)):
            raise CircleMeasurementError(f"平滑敏感性容错拒绝: {name}={value} 不符合 ({lower}, {upper})")
    radius = hole.get("circle", {}).get("radius")
    if (type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0
            or not math.isclose(quality["relative_rmse_radius"], quality["rmse_px"] / radius,
                                rel_tol=1e-4, abs_tol=1e-6)):
        raise CircleMeasurementError("平滑敏感性容错拒绝: 拟合相对残差与半径不一致")
    perturbations = quality.get("perturbations")
    if not isinstance(perturbations, list) or len(perturbations) != 4:
        raise CircleMeasurementError("平滑敏感性容错拒绝: 缺少四方向检测框扰动结果")
    shifts = []
    directions = []
    for item in perturbations:
        if not isinstance(item, dict):
            raise CircleMeasurementError("平滑敏感性容错拒绝: 检测框扰动结果无效")
        shift = item.get("center_shift_px")
        if (type(shift) not in (int, float) or not math.isfinite(shift)
                or not 0 <= shift <= 1 or item.get("reason") != ""):
            raise CircleMeasurementError("平滑敏感性容错拒绝: 检测框扰动未通过")
        shifts.append(shift)
        directions.append((item.get("dx"), item.get("dy")))
    if (set(directions) != {(-3, 0), (3, 0), (0, -3), (0, 3)}
            or not math.isclose(max(shifts), quality["bbox_jitter_max_center_shift_px"],
                                rel_tol=1e-4, abs_tol=1e-6)):
        raise CircleMeasurementError("平滑敏感性容错拒绝: 检测框扰动指标不一致")
    return "smoothing_sensitive"


def select_circle(payload, image_size, expected_detection, settings, confidence):
    """Associate across ALL holes before considering quality, never skip a bad
    target in favour of another successful hole. Current-frame indices aren't IDs.
    """
    try:
        if not isinstance(payload, dict) or payload.get("code") != 200:
            raise ValueError("expected code=200")
        data = payload["data"]
        contract = {"algorithm": "polar_dp_soft_l1_circle_v1",
                    "center_definition": "fitted_outer_circle_center",
                    "coordinate_frame": "original_image_pixels", "units": "px"}
        if any(data.get(k) != v for k, v in contract.items()):
            raise ValueError("incompatible algorithm, centre definition or coordinate frame")
        if data["image_size"] != dict(zip(("width", "height"), image_size)):
            raise ValueError(f"image size mismatch: expected {image_size}, got {data['image_size']}")
        if data.get("check_stability") is not True:
            raise ValueError("stability checking must be enabled")
        if not math.isclose(float(data["bbox_expansion_fraction"]), .10):
            raise ValueError(
                f"ROI expansion mismatch: expected 10% (0.1), got {data['bbox_expansion_fraction']}"
            )
        holes = data["holes"]
        if not isinstance(holes, list):
            raise ValueError("holes must be an array")
        expected = _vector(expected_detection, 2)
        ranked = []
        for hole in holes:
            detection = _point(hole["detection_center"])
            score = float(hole["confidence"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid confidence")
            if hole["status"] not in ("ok", "review", "failed"):
                raise ValueError("invalid status")
            if np.any(detection < 0) or np.any(detection >= image_size):
                raise ValueError("detection centre outside image")
            if score >= confidence:
                ranked.append((float(np.linalg.norm(detection - expected)), hole))
        ranked.sort(key=lambda item: item[0])
        if not ranked or ranked[0][0] > settings.association_px:
            raise CircleMeasurementError("目标孔不在预测检测框中心的关联范围内")
        if (len(ranked) > 1 and ranked[1][0] <= settings.association_px
                and ranked[1][0] - ranked[0][0] < settings.ambiguity_px):
            raise CircleMeasurementError("多个孔与预测位置接近，目标关联存在歧义")
        hole = ranked[0][1]
        policy = _quality_policy(hole, settings)
        center = _point(hole["center"])
        circle = hole["circle"]
        parameters = _vector([circle[k] for k in ("center_x", "center_y", "radius", "diameter")], 4)
        if (np.any(center < 0) or np.any(center >= image_size) or not 0 < parameters[2] <= max(image_size)
                or not np.allclose(center, parameters[:2], rtol=0, atol=1e-4)
                or not math.isclose(parameters[3], 2 * parameters[2], rel_tol=1e-5)):
            raise ValueError("invalid fitted circle")
        box = _vector([hole["detection_bbox"][k] for k in ("x1", "y1", "x2", "y2")], 4)
        if np.any(box[2:] <= box[:2]) or np.any(center < box[:2]) or np.any(center > box[2:]):
            raise ValueError("fitted centre outside detection box")
        if policy != "strict":
            return {**hole, "client_quality_policy": policy}
        return hole
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise CircleProtocolError(f"圆拟合响应协议错误: {error}") from error


class CircleRefinementClient:
    def __init__(self, url, image_size, confidence=.5, settings=None, record=None):
        self.url = url
        self.image_size = tuple(image_size)
        self.confidence = confidence
        self.settings = settings or CircleServoSettings()
        self.settings.validate()
        self.record = record or (lambda event, **data: None)

    def measure(self, image_path, expected_detection):
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None or (image.shape[1], image.shape[0]) != self.image_size:
            raise CircleProtocolError("精调原图无法读取或分辨率与配置不一致")
        try:
            with path.open("rb") as stream:
                response = requests.post(self.url, params={"conf": self.confidence,
                                         "check_stability": "true"},
                                         files={"file": (path.name, stream,
                                                mimetypes.guess_type(path)[0] or "application/octet-stream")},
                                         timeout=self.settings.timeout_s)
        except (OSError, requests.RequestException) as error:
            raise CircleMeasurementError(f"圆拟合请求失败: {error}") from error
        if response.status_code != 200:
            message = f"圆拟合 HTTP {response.status_code}: {response.text[:2000]}"
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                raise CircleProtocolError(message)
            raise CircleMeasurementError(message)
        try:
            payload = response.json()
        except ValueError as error:
            raise CircleProtocolError("圆拟合接口未返回有效 JSON") from error
        self.record("circle_response", image=str(path.resolve()), response=payload,
                    expected_detection=list(map(float, expected_detection)))
        # Persist every received business response, including failed/review targets.
        try:
            path.with_suffix(".circle.json").write_text(json.dumps({
                "url": self.url, "expected_detection": list(map(float, expected_detection)),
                "response": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            print(f"[CircleFine] 响应保存失败，原图保留: {error}")
        hole = select_circle(payload, self.image_size, expected_detection, self.settings, self.confidence)
        self._save_overlay(path, image, hole)
        return hole

    @staticmethod
    def _save_overlay(path, image, hole):
        try:
            # Only draw on the upload's in-memory copy; never overwrite the original.
            circle = hole["circle"]
            center = tuple(int(round(circle[k])) for k in ("center_x", "center_y"))
            cv2.circle(image, center, int(round(circle["radius"])), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            box_center = tuple(int(round(v)) for v in _point(hole["detection_center"]))
            cv2.drawMarker(image, box_center, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
            target = (image.shape[1] // 2, image.shape[0] // 2)
            cv2.drawMarker(image, target, (255, 180, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.line(image, target, center, (0, 255, 255), 2, cv2.LINE_AA)
            if not cv2.imwrite(str(path.with_name(path.stem + "_circle.jpg")), image):
                raise OSError("写入精调标注失败")
        except (ValueError, OSError, cv2.error) as error:
            print(f"[CircleFine] 精调标注保存失败，原图保留: {error}")


def run_fine_alignment(seed_detection, *, measure, get_pose, move, jacobian,
                       target_uv, settings=None, record=None):
    """measure(hint) captures a NEW stationary frame; move(tcp) verifies arrival.

    Require consecutive fresh, sub-threshold measurements without intervening
    motion. On loss/review clear this window. Quality rejection never moves XY.
    """
    cfg = settings or CircleServoSettings()
    cfg.validate()
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e6:
        raise ValueError("精调 J 必须是非奇异、有限的 2×2 矩阵")
    inverse = np.linalg.inv(matrix)
    target = _vector(target_uv, 2)
    hint = _vector(seed_detection, 2)
    stable = []
    fallback_centers = []
    fallback_used = False
    previous_policy = None
    reference_pose = _vector(get_pose(), 6).copy()
    failures = 0
    record = record or (lambda event, **data: None)
    record("fine_start", target_uv=target.tolist(), settings=vars(cfg), jacobian=matrix.tolist(),
           reference_tcp=reference_pose.tolist())

    def checked_pose(checkpoint, iteration):
        pose = _vector(get_pose(), 6).copy()
        angles = (pose[3:] - reference_pose[3:] + 180) % 360 - 180
        delta_z = float(pose[2] - reference_pose[2])
        if (abs(delta_z) > cfg.z_tolerance_mm
                or np.max(np.abs(angles)) > cfg.pose_tolerance_deg):
            record("fine_pose_violation", iteration=iteration, checkpoint=checkpoint,
                   reference_tcp=reference_pose.tolist(), tcp=pose.tolist(),
                   delta_z_mm=delta_z, angle_error_deg=angles.tolist())
            raise RuntimeError(
                "圆心精调期间 Z 或姿态变化，禁止继续 "
                f"({checkpoint}: dZ={delta_z:+.4f} mm, "
                f"最大姿态偏差={np.max(np.abs(angles)):.4f} deg)")
        return pose

    for iteration in range(1, cfg.max_iterations + 1):
        checked_pose("before_measurement", iteration)
        try:
            hole = measure(hint.copy())
            if hole is None:
                raise CircleMeasurementError("精调拍照或测量失败")
            policy = _quality_policy(hole, cfg)
        except CircleMeasurementError as error:
            stable.clear()
            fallback_centers.clear()
            previous_policy = None
            failures += 1
            record("fine_rejected", iteration=iteration, reason=str(error))
            if failures >= cfg.max_failures:
                raise RuntimeError("圆心精调连续测量失败，禁止后续偏移与装配") from error
            continue
        # Capture/inference may take several seconds. Recheck before accepting
        # a measurement, including stationary sub-threshold verification frames.
        pose = checked_pose("after_measurement", iteration)
        failures = 0
        if policy != previous_policy:
            stable.clear()
            fallback_centers.clear()
            previous_policy = policy
        center = _point(hole["center"])
        detection = _point(hole["detection_center"])
        error = center - target
        size = float(np.linalg.norm(error))
        record("fine_measurement", iteration=iteration, center=center.tolist(),
               detection_center=detection.tolist(), error_px=size, quality=hole["fit_quality"],
               target_uv=target.tolist(), confidence=hole.get("confidence"),
               radius_px=hole["circle"]["radius"], tcp=pose.tolist(), quality_policy=policy)
        hint = detection
        confirmed = True
        if policy == "smoothing_sensitive":
            fallback_centers.append(center)
            fallback_centers = fallback_centers[-cfg.smoothing_confirmation_frames:]
            spread = float(np.max(np.linalg.norm(
                fallback_centers - np.mean(fallback_centers, axis=0), axis=1)))
            confirmed = (len(fallback_centers) == cfg.smoothing_confirmation_frames
                         and spread <= cfg.stability_px)
            fallback_used = fallback_used or confirmed
            record("fine_fallback_check", iteration=iteration, frames=len(fallback_centers),
                   required_frames=cfg.smoothing_confirmation_frames, spread_px=spread,
                   scale_center_shift_px=hole["fit_quality"]["scale_center_shift_px"],
                   confirmed=confirmed)
        else:
            fallback_centers.clear()
        if size < cfg.tolerance_px:
            stable.append(center)
            stable = stable[-cfg.stable_frames:]
            if len(stable) == cfg.stable_frames:
                spread = float(np.max(np.linalg.norm(stable - np.mean(stable, axis=0), axis=1)))
                if spread <= cfg.stability_px and confirmed:
                    aligned = checked_pose("before_accept", iteration).tolist()
                    record("fine_aligned", iteration=iteration, tcp=aligned, spread_px=spread,
                           smoothing_fallback_used=fallback_used)
                    return aligned
            # No motion while validating a stationary sub-threshold centre.
            continue
        stable.clear()
        if not confirmed:
            continue
        delta = -cfg.gain * (inverse @ error)
        requested = delta.copy()
        length = float(np.linalg.norm(delta))
        if not math.isfinite(length):
            raise RuntimeError("圆心精调计算出无效位移")
        if length > cfg.max_step_mm:
            delta *= cfg.max_step_mm / length
        before = checked_pose("before_move", iteration)
        # Only XY follows current feedback; Z/orientation stay at the immutable
        # fine-stage reference instead of inheriting and accumulating drift.
        destination = reference_pose.copy()
        destination[:2] = before[:2] + delta
        record("fine_control", iteration=iteration, requested_xy=requested.tolist(),
               delta_xy=delta.tolist(), target_tcp=destination.tolist())
        move(destination.tolist())
        fallback_centers.clear()  # Every new correction needs a fresh stationary confirmation.
        # A blocking SDK/outer motion waiter can use looser acceptance bounds.
        # Confirm the fine-stage bounds on consecutive fresh feedback instead
        # of aborting on the first sample while the robot is still settling.
        deadline = time.monotonic() + cfg.arrival_timeout_s
        arrived_frames = 0
        while True:
            actual = _vector(get_pose(), 6).copy()
            delta_pose = actual - destination
            position_error = float(np.linalg.norm(delta_pose[:3]))
            xy_error = float(np.linalg.norm(delta_pose[:2]))
            angles = (delta_pose[3:] + 180) % 360 - 180
            angle_error = float(np.max(np.abs(angles)))
            arrived = (xy_error <= cfg.arrival_tolerance_mm
                       and abs(delta_pose[2]) <= cfg.z_tolerance_mm
                       and angle_error <= cfg.pose_tolerance_deg)
            arrived_frames = arrived_frames + 1 if arrived else 0
            if arrived_frames >= 3:
                break
            if time.monotonic() >= deadline:
                record("fine_motion_failed", iteration=iteration,
                       target_tcp=destination.tolist(), tcp=actual.tolist(),
                       delta_xyz_mm=delta_pose[:3].tolist(), position_error_mm=position_error,
                       xy_error_mm=xy_error, z_tolerance_mm=cfg.z_tolerance_mm,
                       angle_error_deg=angles.tolist(), timeout_s=cfg.arrival_timeout_s)
                raise RuntimeError(
                    "圆心精调运动反馈未到达目标 "
                    f"(等待 {cfg.arrival_timeout_s:g} s，位置误差={position_error:.4f} mm，"
                    f"XY误差={xy_error:.4f} mm，"
                    f"dZ={delta_pose[2]:+.4f} mm，最大姿态偏差={angle_error:.4f} deg)")
            time.sleep(.05)
        record("fine_motion_complete", iteration=iteration, commanded=delta.tolist(),
               actual=(actual[:2]-before[:2]).tolist(), tcp=actual.tolist())
        # Prediction must use actual TCP travel, not an assumed perfect command.
        hint = detection + matrix @ (actual[:2] - before[:2])
    raise RuntimeError("圆心精调达到最大轮数仍未稳定收敛，禁止后续偏移与装配")
