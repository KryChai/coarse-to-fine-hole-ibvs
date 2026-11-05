"""Backend ellipse-refine client. No image edges or ellipses are fitted here."""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import requests

from .geometry import finite, MeasurementError
from auto_cam_level.parameters import BBOX_EXPANSION_FRACTION, PERCEPTION_VERSION

PERCEPTION_CONTRACT = PERCEPTION_VERSION


def ellipse_refinement_url(detect_url, override=None):
    parts = urlsplit(detect_url)
    for model in ("usb_hole7", "usb_hole11"):
        if parts.path in (f"/api/{model}_detect", f"/api/{model}_ellipse_refine"):
            expected = f"/api/{model}_ellipse_refine"
            result = urlsplit(override) if override else parts._replace(path=expected, query="", fragment="")
            if result.scheme not in ("http", "https") or not result.netloc or result.path != expected or result.query or result.fragment:
                raise ValueError("ellipse endpoint must match detection model and have no query/fragment")
            return urlunsplit(result)
    raise ValueError("expected usb_hole7/11_detect or matching ellipse_refine URL")


def parse_ellipses(payload, camera, count):
    """Accept the entire ring or reject it; never skip a failed/review hole."""
    try:
        if not isinstance(payload, dict) or payload.get("code") != 200:
            raise ValueError("expected code=200")
        data = payload["data"]
        contract = dict(algorithm="polar_radial_algebraic_ellipse_v1",
                        center_definition="fitted_outer_ellipse_center",
                        coordinate_frame="original_image_pixels", units="px")
        if any(data.get(k) != v for k, v in contract.items()):
            raise ValueError("incompatible ellipse algorithm/coordinates/units")
        if (data["image_size"] != dict(zip(("width", "height"), camera.image_size))
                or data.get("check_stability") is not True
                or not np.isclose(float(data["bbox_expansion_fraction"]), BBOX_EXPANSION_FRACTION)):
            raise ValueError("image size, ROI or stability contract mismatch")
        holes = data["holes"]
        if not isinstance(holes, list) or len(holes) != count:
            raise ValueError("need complete holes ring; single-hole/all_holes response is insufficient")
        fits = []
        for hole in holes:
            quality = hole["fit_quality"]
            if (hole["status"] != "ok" or quality.get("valid") is not True
                    or quality.get("stability_checked") is not True or quality.get("flags") != []):
                raise ValueError(f"ellipse rejected: {hole['status']} {quality.get('flags')}")
            conf = float(hole["confidence"])
            center = finite([hole["center"][k] for k in ("x", "y")], (2,))
            box = finite([hole["detection_bbox"][k] for k in ("x1", "y1", "x2", "y2")], (4,))
            ellipse = hole["ellipse"]
            params = finite([ellipse[k] for k in ("center_x", "center_y", "semi_major_axis_px",
                            "semi_minor_axis_px", "rotation_angle_deg")], (5,))
            a, b, angle = params[2:]
            if (not np.isfinite(conf) or not .5 <= conf <= 1 or not 0 < b <= a <= max(camera.image_size)
                    or not np.allclose(params[:2], center, atol=1e-4, rtol=0)
                    or np.any(center < 0) or np.any(center >= camera.image_size)
                    or np.any(box[2:] <= box[:2]) or np.any(box[:2] < 0) or np.any(box[2:] > camera.image_size)
                    or np.any(center < box[:2]) or np.any(center > box[2:])
                    or not np.isclose(float(ellipse["major_axis_length_px"]), 2*a)
                    or not np.isclose(float(ellipse["minor_axis_length_px"]), 2*b)):
                raise ValueError("invalid ellipse centre/axes/bbox/confidence")
            # Only transform the backend's analytic ellipse, never refit edges.
            theta = np.arange(256)*2*np.pi/256
            phi = np.deg2rad(angle)
            R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            outline = np.column_stack((a*np.cos(theta), b*np.sin(theta)))@R.T+center
            if np.any(outline < 0) or np.any(outline >= camera.image_size):
                raise ValueError("ellipse outline outside raw image")
            xy = camera.normalize(outline)
            xy -= xy.mean(axis=0)
            area = .5*abs(np.sum(xy[:, 0]*np.roll(xy[:, 1], -1)-xy[:, 1]*np.roll(xy[:, 0], -1)))
            diameter = float(2*np.sqrt(area/np.pi))
            if not np.isfinite(diameter) or diameter <= 0:
                raise ValueError("degenerate normalized ellipse area")
            fits.append(dict(center=center, axes=np.array([2*a, 2*b]), angle=float(angle),
                             diameter=diameter, quality=quality))
        return fits
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as exc:
        raise MeasurementError(f"ellipse response rejected: {exc}") from exc


class ArrayDetector:
    def __init__(self, url, camera, count, record=None, *, ellipse_url=None, timeout_s=60.):
        self.url = ellipse_refinement_url(url, ellipse_url)
        self.camera, self.count = camera, count
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("ellipse timeout must be positive")
        self.record = record or (lambda event, **data: None)

    def detect(self, path):
        path = Path(path)
        frame = cv2.imread(str(path))
        if frame is None or (frame.shape[1], frame.shape[0]) != self.camera.image_size:
            raise MeasurementError("raw image dimensions differ from calibration")
        import mimetypes
        self.record("array_request", image=str(path), endpoint=self.url, timeout_s=self.timeout_s,
                    bbox_expansion_fraction=BBOX_EXPANSION_FRACTION)
        with path.open("rb") as stream:
            response = requests.post(self.url, params={"conf": .5, "check_stability": "true"},
                files={"file": (path.name, stream, mimetypes.guess_type(path.name)[0] or "application/octet-stream")},
                timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()
        self.record("array_response", image=str(path), endpoint=self.url, response=payload)
        fits = parse_ellipses(payload, self.camera, self.count)
        overlay = frame.copy()
        for i, fit in enumerate(fits):
            center = tuple(fit["center"].astype(int))
            cv2.ellipse(overlay, (tuple(fit["center"]), tuple(fit["axes"]), fit["angle"]), (0, 255, 0), 2)
            cv2.putText(overlay, str(i), center, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        if not cv2.imwrite(str(path.with_name(path.stem+"_ellipses.jpg")), overlay):
            raise OSError("cannot save ellipse diagnostic image")
        self.record("array_fits", image=str(path), perception_contract=PERCEPTION_CONTRACT,
                    centres=[f["center"].tolist() for f in fits],
                    normalized_diameters=[f["diameter"] for f in fits], quality=[f["quality"] for f in fits])
        return np.array([f["center"] for f in fits]), np.array([f["diameter"] for f in fits])
