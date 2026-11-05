"""Calibrated image coordinates and complete circular hole arrays."""
from dataclasses import dataclass

import cv2
import numpy as np


class MeasurementError(RuntimeError):
    """An observation cannot safely drive the robot."""


def finite(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"expected finite array of shape {shape}")
    return result


@dataclass
class CameraModel:
    K: object
    D: object
    image_size: object

    def __post_init__(self):
        self.K = finite(self.K, (3, 3))
        self.D = np.asarray(self.D, dtype=float).reshape(-1)
        size = finite(self.image_size, (2,))
        if (np.any(size <= 0) or np.any(size != size.astype(int))
                or self.D.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.D).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0, 0, 1])
                or self.K[0, 1] != 0 or self.K[1, 0] != 0):
            raise ValueError("invalid pinhole camera K/D/image_size")
        self.image_size = tuple(size.astype(int))

    def normalize(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise MeasurementError("nonfinite image points")
        return cv2.undistortPoints(points.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)


def complete_array(points, count, camera):
    points = finite(points, (count, 2))
    if count < 6 or np.any(points < 0) or np.any(points >= camera.image_size):
        raise MeasurementError("need >=6 complete, visible ring centres inside raw image")
    distances = np.linalg.norm(points[:, None]-points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 3:
        raise MeasurementError("duplicate/overlapping hole centres")
    xy = camera.normalize(points)
    singular = np.linalg.svd(xy-xy.mean(axis=0), compute_uv=False)
    if singular[-1] < 1e-4 or singular[-1]/singular[0] < .15:
        raise MeasurementError("degenerate or excessively tilted array")
    # Every hole in a perspective-projected circular array must be on its hull.
    if len(cv2.convexHull(xy.astype(np.float32))) != count:
        raise MeasurementError("array contains an interior outlier or missing ring geometry")
    return points, xy


def ring_points(count, radius_mm):
    if type(count) is not int or count < 6 or not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("ring needs >=6 holes and a positive radius_mm")
    phase = np.arange(count)*2*np.pi/count
    return np.column_stack((radius_mm*np.cos(phase), radius_mm*np.sin(phase), np.zeros(count)))


def normal_error(normal):
    n = finite(normal, (3,))
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("plane normal must be nonzero")
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    cross = np.array([-n[1], n[0]])  # z_C cross n_C; radians
    sine = np.linalg.norm(cross)
    angle = float(np.arctan2(sine, n[2]))
    return cross * (angle/sine if sine > 1e-12 else 1.), n, angle
