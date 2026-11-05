"""Outer-rim image-center HTTP contract and bounded fine-alignment loop.

No robot SDK imports. All observation/target coordinates are ORIGINAL image pixels.
Absolute hole sizes are not required; no metric 3D position is inferred here.
"""
from dataclasses import asdict, dataclass
import json
import math
import mimetypes
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import requests

from servo_control import ServoSettings, ServoStepController


ALGORITHM = "outer_rim_ellipse_v1"
REFERENCE = 'outer_rim'
CENTER_DEFINITION = 'fitted_outer_ellipse_center'
BOUNDARY_METHOD = "normal_caliper_ransac_v1"
BOUNDARY_REVISION = 2


@dataclass(frozen=True)
class FineSettings:
    tolerance_px: float = 1.0
    stable_frames: int = 3
    stable_spread_px: float = .35
    max_iterations: int = 80
    max_failures: int = 3
    max_non_improving: int = 8
    gain: float = .03
    max_gain: float = .06
    max_step_mm: float = .15
    association_px: float = 20.0

    def controller(self, jacobian):
        if (not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 1
                       for v in (self.stable_frames, self.max_iterations, self.max_failures, self.max_non_improving))
                or self.stable_frames < 2 or self.stable_spread_px >= self.tolerance_px):
            raise ValueError("Invalid fine alignment settings")
        return ServoStepController(jacobian, ServoSettings(
            mode="adaptive", gain=self.gain, max_gain=self.max_gain,
            align_tol_px=self.tolerance_px, transition_error_px=10,
            slowdown_error_px=5, max_step_mm=self.max_step_mm))


class RefinementError(RuntimeError):
    """Permanent configuration/protocol failure; do not retry as a missing hole."""


def service_urls(coarse_url):
    parts = urlsplit(coarse_url)
    name = parts.path.rsplit("/", 1)[-1]
    model = next((m for m in ("usb_hole7", "usb_hole11") if name.startswith(m + "_")), None)
    if parts.scheme not in ("http", "https") or not parts.netloc or model is None:
        raise ValueError("Fine alignment requires a usb_hole7/usb_hole11 detection URL")
    return (urlunsplit((parts.scheme, parts.netloc, f"/api/{model}_refine_center", "", "")),
            urlunsplit((parts.scheme, parts.netloc, "/api/refinement/status", "", "")))


class RefinementClient:
    def __init__(self, coarse_url, image_size, association_px=20, confidence_threshold=.5):
        self.url, self.status_url = service_urls(coarse_url)
        self.task = self.url.rsplit("/", 1)[-1]
        self.image_size = tuple(image_size)
        self.association_px = association_px
        self.confidence_threshold = confidence_threshold
        self.calibration_id = None
        self.preflight_done = False
        self.last_data = None

    def preflight(self):
        try:
            response = requests.get(self.status_url, params={'mode': 'outer_rim'}, timeout=3)
            response.raise_for_status()
            data = response.json()
            if (data.get("code") != 200 or data.get("ready") is not True
                    or data.get("algorithm") != ALGORITHM
                    or data.get("boundary_method") != BOUNDARY_METHOD
                    or data.get("boundary_revision") != BOUNDARY_REVISION
                    or data.get("reference") != REFERENCE
                    or data.get('center_definition') != CENTER_DEFINITION
                    or data.get('calibration_required') is not False
                    or data.get('calibration_id') is not None
                    or data.get('image_size_policy') != 'from_upload'
                    or (data.get('image_size') is not None and tuple(data['image_size']) != self.image_size)):
                raise ValueError("incompatible outer-rim algorithm, reference or image-size policy")
            self.preflight_done = True
            return data
        except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
            raise RefinementError(f"外圈精调服务预检失败，请启动配套的最外圈拟合接口: {exc}") from exc

    def measure(self, image_path, expected):
        self.last_data = None
        if not self.preflight_done:
            self.preflight()
        try:
            with Path(image_path).open("rb") as stream:
                response = requests.post(self.url, params={"mode": "outer_rim", "conf": self.confidence_threshold, "target_u": float(expected[0]),
                    "target_v": float(expected[1]), "max_distance": self.association_px},
                    files={"file": (Path(image_path).name, stream,
                           mimetypes.guess_type(str(image_path))[0] or "application/octet-stream")}, timeout=3)
        except (OSError, requests.RequestException):
            return None
        if response.status_code != 200:
            if response.status_code in (408, 429) or response.status_code >= 500:
                return None
            raise RefinementError(f"精定位协议错误 HTTP {response.status_code}: {response.text[:1000]}")
        try:
            payload = response.json()
            data = payload["data"]
            self.last_data = data
            if (payload.get("code") != 200 or payload.get("task") != self.task or data.get("algorithm") != ALGORITHM
                    or data.get("boundary_method") != BOUNDARY_METHOD
                    or data.get("boundary_revision") != BOUNDARY_REVISION
                    or data.get("coordinate_frame") != "original_image_pixels"
                    or data.get("calibration_id") != self.calibration_id
                    or data.get('calibration_required') is not False
                    or data.get('reference') != REFERENCE
                    or data.get('center_definition') != CENTER_DEFINITION
                    or tuple(data.get("image_size", ())) != self.image_size):
                raise ValueError("response algorithm/calibration/coordinates changed")
            target = data["target_hole"]
            if target is None:
                return None
            quality = target["fit_quality"]
            if quality["valid"] is not True or target["refined_center"] is None:
                return None
            refined = target["refined_center"]
            if (refined["method"] != ALGORITHM or refined["reference"] != REFERENCE
                    or refined.get('center_definition') != CENTER_DEFINITION):
                raise ValueError("wrong reference circle")
            p = np.array([refined["center_x"], refined["center_y"]], dtype=float)
            coarse = np.array([target["detection_center"][k] for k in ("center_x", "center_y")], dtype=float)
            bbox = np.array([target[k] for k in ("x1", "y1", "x2", "y2")], dtype=float)
            conf = float(target["conf"])
            rms, coverage = float(quality["residual_rms_px"]), float(quality["angular_coverage_deg"])
            support = float(quality['caliper_support_fraction'])
            if (not np.isfinite(np.r_[p, coarse, bbox, conf, rms, coverage, support]).all()
                    or not self.confidence_threshold <= conf <= 1 or not 0 <= rms <= 1.2 or not 270 <= coverage <= 360
                    or not .65 <= support <= 1
                    or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]
                    or not 0 <= p[0] < self.image_size[0] or not 0 <= p[1] < self.image_size[1]
                    or not bbox[0] <= p[0] <= bbox[2] or not bbox[1] <= p[1] <= bbox[3]
                    or np.linalg.norm(coarse-expected) > self.association_px):
                raise ValueError("invalid geometry or target association")
            geometry = target["geometry"]
            for key in ("outlines_px", "edge_points_px"):
                if not isinstance(geometry[key], list) or len(geometry[key]) != 1:
                    raise ValueError("missing fitted geometry")
                for points in geometry[key]:
                    points = np.asarray(points, dtype=float)
                    if points.ndim != 2 or points.shape[1] != 2 or not 5 <= len(points) <= 1000 or not np.isfinite(points).all():
                        raise ValueError("invalid fitted geometry")
                    if np.max(np.abs(points)) > 2*max(self.image_size):
                        raise ValueError("fitted geometry outside image domain")
            return {"center": p, "coarse": coarse, "bbox": bbox, "confidence": conf,
                    "rms": rms, "coverage": coverage, "data": data}
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise RefinementError(f"精定位响应无效: {exc}") from exc


def run_fine_alignment(*, capture, measure, move, get_tcp, record, set_phase,
                       jacobian, initial_target, target_pixel, settings=None, iteration_offset=0):
    settings = settings or FineSettings()
    controller = settings.controller(jacobian)  # fresh history, fine tolerance inside controller
    expected = np.asarray(initial_target, dtype=float)
    target_pixel = np.asarray(target_pixel, dtype=float)
    if not np.isfinite(np.r_[expected, target_pixel]).all() or expected.shape != (2,) or target_pixel.shape != (2,):
        raise ValueError("Invalid fine target")
    set_phase("servo_fine")
    failures = stalled = 0
    best = math.inf
    stable = []
    for local_step in range(1, settings.max_iterations+1):
        iteration = iteration_offset + local_step
        path = capture()
        sample = measure(path, expected) if path else None
        if sample is None:
            stable.clear()
            controller.detection_lost()
            failures += 1
            record("detection_failed", iteration=iteration, message="fine fit/target unavailable")
            if failures >= settings.max_failures:
                raise RuntimeError("精定位连续失败，禁止进入装配偏移")
            continue
        failures = 0
        center = sample["center"]
        error = center-target_pixel
        size = float(np.linalg.norm(error))
        record("vision", iteration=iteration, u_px=float(center[0]), v_px=float(center[1]),
               confidence=sample["confidence"], error_u_px=float(error[0]), error_v_px=float(error[1]),
               error_norm_px=size, message=json.dumps({"stage": "fine", "image": str(path),
                   "target_pixel": target_pixel.tolist(), "tolerance_px": settings.tolerance_px,
                   "refinement": sample["data"]}, ensure_ascii=False, allow_nan=False))
        print(f"[精调 {local_step}] error={size:.3f} px, residual={sample['rms']:.3f} px")
        expected = np.asarray(sample["coarse"], dtype=float).copy()
        if size < settings.tolerance_px:
            stable.append(center)
            stable = stable[-settings.stable_frames:]
            if len(stable) == settings.stable_frames:
                spread = float(np.max(np.linalg.norm(np.asarray(stable)-np.mean(stable, axis=0), axis=1)))
                if spread <= settings.stable_spread_px:
                    record("fine_aligned", iteration=iteration, message=f"stable_spread_px={spread:.6f}")
                    return get_tcp()
            continue  # no motion while verifying a stationary target
        stable.clear()
        if size < best-.01:
            best, stalled = size, 0
        else:
            stalled += 1
            if stalled >= settings.max_non_improving:
                raise RuntimeError("精调误差未改善，请检查边缘、J 和最小有效运动")
        step = controller.step(error)
        tcp = list(get_tcp())
        tcp[0] += float(step.selected[0])
        tcp[1] += float(step.selected[1])
        record("control", iteration=iteration, target_tcp=tcp, **step.telemetry())
        move(tcp)
        expected += np.asarray(jacobian) @ step.selected
    raise RuntimeError("精调达到最大迭代次数或中心未稳定，禁止进入装配偏移")
