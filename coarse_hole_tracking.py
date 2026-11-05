"""Lock a coarse-alignment session to the first selected hole ID."""

from math import dist, isfinite

import numpy as np


class TrackingProtocolError(ValueError):
    """The detector did not return valid, distinct track IDs and measurements."""


class CoarseHoleTracker:
    """Select the nearest visible hole once, then gate its subsequent measurements.

    Use one instance per detector session. Input records carry a stable positive
    integer hole_id, a missing flag, and, when visible, center_x, center_y and conf.
    """

    def __init__(self, target_uv, association_px=40.0):
        self.target_uv = _point(target_uv)
        if not isfinite(association_px) or association_px <= 0:
            raise ValueError("association_px must be finite and positive")
        self.association_px = association_px
        self.hole_id = None

    def select(self, holes, predicted_uv=None):
        if not isinstance(holes, list):
            raise TrackingProtocolError("holes must be an array")
        visible = {}
        seen = set()
        for item in holes:
            if not isinstance(item, dict):
                raise TrackingProtocolError("hole record must be an object")
            hole_id, missing = item.get("hole_id"), item.get("missing")
            if type(hole_id) is not int or hole_id <= 0 or hole_id in seen or type(missing) is not bool:
                raise TrackingProtocolError("invalid or duplicate hole_id or invalid missing flag")
            seen.add(hole_id)
            if missing:
                continue
            try:
                measurement = tuple(float(item[key]) for key in ("center_x", "center_y", "conf"))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise TrackingProtocolError("invalid hole center or confidence") from exc
            if not all(map(isfinite, measurement)) or not 0 <= measurement[2] <= 1:
                raise TrackingProtocolError("invalid hole center or confidence")
            visible[hole_id] = measurement

        if self.hole_id is None and visible:
            self.hole_id = min(visible, key=lambda key: (dist(visible[key][:2], self.target_uv), key))
        measurement = visible.get(self.hole_id)
        if measurement is not None and predicted_uv is not None:
            if dist(measurement[:2], _point(predicted_uv)) > self.association_px:
                return None
        return measurement


def predict_target_uv(previous_uv, jacobian, measured_xy_delta):
    """Predict the next raw-image center from measured XY motion in millimetres."""
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("jacobian must be a finite 2×2 matrix")
    return np.asarray(_point(previous_uv)) + matrix @ _point(measured_xy_delta)


def _point(value):
    coordinates = tuple(map(float, value))
    if len(coordinates) != 2 or not all(map(isfinite, coordinates)):
        raise ValueError("expected two finite coordinates")
    return coordinates
