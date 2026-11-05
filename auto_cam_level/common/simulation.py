"""Perspective projection plant with known normals; never connects hardware."""
import cv2
import numpy as np

from .control import Observation
from auto_cam_level.parameters import MEASUREMENT_SAMPLES, SIMULATION
from .geometry import ring_points, normal_error


class SyntheticPlant:
    def __init__(self, camera, feature, seed=1, tilt=(6., -4.), noise_px=None):
        self.camera, self.feature = camera, feature
        self.rng, self.noise_px = np.random.default_rng(seed), SIMULATION["noise_px"] if noise_px is None else noise_px
        self.pose = np.array([0., 0., SIMULATION["distance_mm"], 0., 0., 0.])
        self.tilt = np.asarray(tilt, dtype=float)
        self.coupling = np.array(SIMULATION["coupling"])
        self.now, self.frame, self.moves = 0., 0, []

    def rotation(self):
        r = np.deg2rad(self.tilt+self.coupling@self.pose[3:5])
        return cv2.Rodrigues(np.r_[r, 0.])[0]

    def truth_tilt_deg(self):
        return float(np.rad2deg(normal_error(self.rotation()[:, 2])[2]))

    def project(self):
        radius = np.linalg.norm(self.feature.model[0]) if hasattr(self.feature, "model") else SIMULATION["array_radius_mm"]
        model = ring_points(self.feature.count, radius)
        R, t = self.rotation(), np.array([*SIMULATION["offset_xy_mm"], SIMULATION["distance_mm"]])
        rv = cv2.Rodrigues(R)[0]
        points = cv2.projectPoints(model, rv, t, self.camera.K, self.camera.D)[0].reshape(-1, 2)
        phase = np.arange(64)*2*np.pi/64
        rim = np.column_stack((SIMULATION["hole_radius_mm"]*np.cos(phase), SIMULATION["hole_radius_mm"]*np.sin(phase), np.zeros(64)))
        diameters = []
        for point in model:
            pixel_rim = cv2.projectPoints(rim+point, rv, t, self.camera.K, self.camera.D)[0].reshape(-1, 2)
            C = np.cov(self.camera.normalize(pixel_rim), rowvar=False, bias=True)
            diameters.append(2*np.sqrt(2)*np.linalg.det(C)**.25)
        points += self.rng.normal(0, self.noise_px, points.shape)
        diameters = np.array(diameters)*np.exp(self.rng.normal(0, self.noise_px/100, len(model)))
        order = self.rng.permutation(len(model))
        return points[order], diameters[order]

    def measure(self):
        # Same three-sample stationary aggregate as the field adapter.
        values, metadata = [], None
        for _ in range(MEASUREMENT_SAMPLES):
            points, diameters = self.project()
            value, metadata = self.feature(points, diameters)
            values.append(value)
            self.frame += 1
            self.now += .1
        values = np.array(values)
        return Observation(np.median(values, axis=0), self.pose.copy(), self.frame,
                           self.now, np.std(values, axis=0), metadata)

    def move(self, target):
        self.moves.append(list(target))
        self.pose = np.asarray(target, dtype=float).copy()
        self.now += .5
