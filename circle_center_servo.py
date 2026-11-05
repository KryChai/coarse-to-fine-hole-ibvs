"""Client for yolo_service/dev circle_refine and bounded second-stage XY servo.

No robot SDK imports. The caller supplies fresh captures and blocking, checked
TCP motion. Detection-box centres associate targets; fitted centres control XY.
"""

from dataclasses import dataclass
import json
import math
import mimetypes
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
    max_non_improving: int = 8
    arrival_tolerance_mm: float = 0.1
    pose_tolerance_deg: float = 0.1
    # Fitting includes all detected holes and four perturbation fits per hole.
    timeout_s: float = 60.0

    def validate(self):
        values = vars(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("圆心精调参数必须为有限正数")
        if self.gain > 1 or self.max_step_mm > 1 or self.ambiguity_px >= self.association_px:
            raise ValueError("圆心精调增益、位移上限或关联门限无效")
        for key in ("stable_frames", "max_iterations", "max_failures", "max_non_improving"):
            if type(values[key]) is not int:
                raise ValueError(f"{key} 必须是正整数")


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
        if (data["image_size"] != dict(zip(("width", "height"), image_size))
                or data.get("check_stability") is not True
                or not math.isclose(float(data["bbox_expansion_fraction"]), .05)):
            raise ValueError("image size, 5% ROI or stability contract mismatch")
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
        quality = hole["fit_quality"]
        if (hole["status"] != "ok" or quality.get("valid") is not True
                or quality.get("stability_checked") is not True or quality.get("flags") != []):
            raise CircleMeasurementError(f"目标外圈拟合未通过质量检查: {quality.get('flags')}")
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
    reference_pose = _vector(get_pose(), 6)
    failures = non_improving = 0
    best_error = math.inf
    record = record or (lambda event, **data: None)
    record("fine_start", target_uv=target.tolist(), settings=vars(cfg))
    for iteration in range(1, cfg.max_iterations + 1):
        pose = _vector(get_pose(), 6)
        angles = (pose[3:] - reference_pose[3:] + 180) % 360 - 180
        if (abs(pose[2] - reference_pose[2]) > cfg.arrival_tolerance_mm
                or np.max(np.abs(angles)) > cfg.pose_tolerance_deg):
            raise RuntimeError("圆心精调期间 Z 或姿态变化，禁止继续")
        try:
            hole = measure(hint.copy())
            if hole is None:
                raise CircleMeasurementError("精调拍照或测量失败")
        except CircleMeasurementError as error:
            stable.clear()
            failures += 1
            record("fine_rejected", iteration=iteration, reason=str(error))
            if failures >= cfg.max_failures:
                raise RuntimeError("圆心精调连续测量失败，禁止后续偏移与装配") from error
            continue
        failures = 0
        center = _point(hole["center"])
        detection = _point(hole["detection_center"])
        error = center - target
        size = float(np.linalg.norm(error))
        record("fine_measurement", iteration=iteration, center=center.tolist(),
               detection_center=detection.tolist(), error_px=size, quality=hole["fit_quality"],
               target_uv=target.tolist(), confidence=hole.get("confidence"),
               radius_px=hole["circle"]["radius"], tcp=pose.tolist())
        hint = detection
        if size < cfg.tolerance_px:
            stable.append(center)
            stable = stable[-cfg.stable_frames:]
            if len(stable) == cfg.stable_frames:
                spread = float(np.max(np.linalg.norm(stable - np.mean(stable, axis=0), axis=1)))
                if spread <= cfg.stability_px:
                    aligned = _vector(get_pose(), 6).tolist()
                    record("fine_aligned", iteration=iteration, tcp=aligned, spread_px=spread)
                    return aligned
            # No motion while validating a stationary sub-threshold centre.
            continue
        stable.clear()
        if size < best_error - 1e-3:
            best_error, non_improving = size, 0
        else:
            non_improving += 1
            if non_improving >= cfg.max_non_improving:
                raise RuntimeError("圆心精调误差连续未改善，禁止后续偏移与装配")
        delta = -cfg.gain * (inverse @ error)
        length = float(np.linalg.norm(delta))
        if not math.isfinite(length):
            raise RuntimeError("圆心精调计算出无效位移")
        if length > cfg.max_step_mm:
            delta *= cfg.max_step_mm / length
        before = _vector(get_pose(), 6)
        destination = before.copy()
        destination[:2] += delta
        record("fine_control", iteration=iteration, delta_xy=delta.tolist(), target_tcp=destination.tolist())
        move(destination.tolist())
        actual = _vector(get_pose(), 6)
        record("fine_motion_complete", iteration=iteration, commanded=delta.tolist(),
               actual=(actual[:2]-before[:2]).tolist(), tcp=actual.tolist())
        angles = (actual[3:] - destination[3:] + 180) % 360 - 180
        if (np.linalg.norm(actual[:3] - destination[:3]) > cfg.arrival_tolerance_mm
                or np.max(np.abs(angles)) > cfg.pose_tolerance_deg):
            raise RuntimeError("圆心精调运动反馈未到达目标")
        # Prediction must use actual TCP travel, not an assumed perfect command.
        hint = detection + matrix @ (actual[:2] - before[:2])
    raise RuntimeError("圆心精调达到最大轮数仍未稳定收敛，禁止后续偏移与装配")
