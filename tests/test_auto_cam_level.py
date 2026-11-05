"""Projection-grounded levelling tests; no real camera, SDK or HTTP service."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from auto_cam_level.common.control import ControlSettings, Observation, RotationServo
from auto_cam_level.common.geometry import CameraModel, MeasurementError, normal_error, ring_points
from auto_cam_level.common.runtime import load_config
from auto_cam_level.common.simulation import SyntheticPlant
from auto_cam_level.common.vision import ArrayDetector, parse_ellipses, ellipse_refinement_url
from auto_cam_level.scheme1_pose.features import PoseFeatures
from auto_cam_level.scheme2_shape.features import ShapeFeatures


ROOT = Path(__file__).resolve().parents[1]


def camera(distorted=False):
    return CameraModel([[1800., 0, 1280], [0, 1500., 720], [0, 0, 1]],
                       [.1, -.03, .001, -.002, 0] if distorted else [0]*5, [2560, 1440])


def projection(cam, tilt=(12., -8.), count=8):
    R = cv2.Rodrigues(np.deg2rad([*tilt, 0.]))[0]
    xy = cv2.projectPoints(ring_points(count, 40), cv2.Rodrigues(R)[0], np.array([20., -15., 450.]), cam.K, cam.D)[0].reshape(-1, 2)
    return xy, R


class FeatureTests(unittest.TestCase):
    def test_both_pose_methods_recover_normal_with_distortion_and_unordered_ring(self):
        cam = camera(True)
        xy, R = projection(cam)
        truth = normal_error(R[:, 2])[1]
        for method in ("pnp", "homography"):
            feature = PoseFeatures(cam, method=method)
            for permutation in (np.arange(8), [4, 2, 7, 1, 0, 5, 3, 6], np.arange(7, -1, -1)):
                error, metadata = feature(xy[permutation])
                np.testing.assert_allclose(metadata["normal"], truth, atol=2e-5)
                np.testing.assert_allclose(error, normal_error(truth)[0], atol=2e-5)
                self.assertLess(metadata["reprojection_rms_px"], .001)

    def test_frontoparallel_normal_and_sign_are_finite(self):
        np.testing.assert_array_equal(normal_error([0, 0, -1])[0], [0, 0])
        for method in ("pnp", "homography"):
            xy, _ = projection(camera(), (0., 0.))
            error, _ = PoseFeatures(camera(), method=method)(xy)
            self.assertLess(np.linalg.norm(error), 1e-6)

    def test_ambiguous_pnp_normals_are_rejected(self):
        cam = camera()
        xy, R = projection(cam)
        other = cv2.Rodrigues(np.deg2rad([-12., 8., 0.]))[0]
        result = (2, [cv2.Rodrigues(R)[0], cv2.Rodrigues(other)[0]],
                  [np.array([20., -15., 450.])]*2)
        with patch("cv2.solvePnPGeneric", return_value=result), \
             patch("cv2.solvePnPRefineLM", side_effect=lambda o, p, k, d, r, t: (r, t)):
            with self.assertRaisesRegex(MeasurementError, "ambiguous"):
                PoseFeatures(cam, max_reprojection_px=100, ambiguity_px=100)(xy)

    def test_bad_array_geometry_never_produces_a_feature(self):
        xy, _ = projection(camera())
        bad_sets = [xy[:-1], np.zeros((8, 2)), xy.copy(), xy.copy()]
        bad_sets[2][0] = [1280, 720]
        bad_sets[3][0] = [float("nan"), 0]
        for cls in (PoseFeatures, ShapeFeatures):
            for points in bad_sets:
                with self.assertRaises((MeasurementError, ValueError)):
                    cls(camera())(points, np.ones(len(points)))

    def test_wrong_correspondence_geometry_fails_reprojection(self):
        xy, _ = projection(camera())
        xy[0] += [18, 8]
        with self.assertRaises(MeasurementError):
            PoseFeatures(camera())(xy)

    def test_shape_uses_calibrated_coordinates_with_unequal_focal_lengths(self):
        cam = camera(True)
        xy, _ = projection(cam, (0., 0.))
        s, metadata = ShapeFeatures(cam, enhanced=False)(xy)
        self.assertLess(np.linalg.norm(s), 1e-7)
        self.assertEqual(metadata["coordinate_frame"], "undistorted_normalized")
        self.assertGreater(abs(np.var(xy[:, 0])-np.var(xy[:, 1])), 100)

    def test_size_gradient_is_signed_and_no_implicit_dimension_fallback(self):
        cam = camera()
        xy, _ = projection(cam, (0., 0.))
        n = cam.normalize(xy)
        centered = n-n.mean(0)
        radius = np.sqrt(np.sum(centered**2)/8)
        d = .01*np.exp(centered@np.array([.04, -.02])/radius)
        feature = ShapeFeatures(cam)
        s, _ = feature(xy, d)
        np.testing.assert_allclose(s[2:], [.04, -.02], atol=1e-9)
        self.assertEqual(s.shape, (4,))
        with self.assertRaises(ValueError):
            feature(xy)
        d[0] *= 1.5
        with self.assertRaisesRegex(MeasurementError, "inconsistent"):
            feature(xy, d)

    def test_camera_rejects_invalid_calibration(self):
        with self.assertRaises(ValueError):
            CameraModel(np.eye(3), [np.nan]*5, [2560, 1440])


class ServoTests(unittest.TestCase):
    def run_plant(self, feature=None, tilt=(6., -4.), seed=1, mutate=None, move_override=None, cfg=None):
        feature = feature or PoseFeatures(camera())
        plant = SyntheticPlant(feature.camera, feature, seed, tilt)
        tol = [.002, .002, .0008, .0008] if isinstance(feature, ShapeFeatures) and feature.enhanced else [np.deg2rad(.5)]*2
        events = []
        def measure():
            obs = plant.measure()
            return mutate(obs, plant) if mutate else obs
        servo = RotationServo(measure, lambda: plant.pose.copy(), move_override or plant.move, tol,
                              cfg, lambda event, **v: events.append(dict(event=event, **v)), clock=lambda: plant.now)
        return servo, plant, events

    def test_perspective_closed_loops_cover_both_directions_and_four_by_two_jacobian(self):
        for method in ("pnp", "homography", "shape"):
            for seed, tilt in ((1, (6., -4.)), (7, (-5., 7.))):
                feature = ShapeFeatures(camera()) if method == "shape" else PoseFeatures(camera(), method=method)
                servo, plant, events = self.run_plant(feature, tilt, seed)
                result = servo.run()
                self.assertLess(plant.truth_tilt_deg(), .6)
                np.testing.assert_allclose(np.array(result)[[0, 1, 2, 5]], [0, 0, 450, 0])
                controls = [e for e in events if e["event"] == "control"]
                self.assertTrue(controls)
                self.assertTrue(any(e["event"] == "broyden" and e["reason"] == "accepted" for e in events))
                self.assertTrue(all(np.linalg.norm(e["delta_deg"]) <= .5+1e-9 for e in controls))
                self.assertTrue(any(e["event"] == "rotation_calibration" for e in events))

    def test_already_level_requires_stable_frames_without_motion_or_probe(self):
        for feature in (PoseFeatures(camera()), ShapeFeatures(camera()), ShapeFeatures(camera(), enhanced=False)):
            servo, plant, events = self.run_plant(feature, (0., 0.))
            servo.run()
            self.assertEqual(plant.moves, [])
            self.assertGreaterEqual(len([e for e in events if e["event"] == "observation"]), 3)

    def test_confirmation_window_is_cleared_when_initial_level_condition_is_lost(self):
        frame = [0]
        def measure():
            frame[0] += 1
            value = [.1, 0] if frame[0] == 1 else ([2., 0] if frame[0] == 2 else [0., 0])
            return Observation(value, [0.]*6, frame[0], float(frame[0]), [0., 0.])
        move = Mock()
        servo = RotationServo(measure, lambda: [0.]*6, move, [1., 1.], clock=lambda: float(frame[0]))
        servo.run(np.eye(2))
        self.assertEqual(frame[0], 5)  # three NEW confirmations after losing level
        move.assert_not_called()

    def test_duplicate_frame_and_old_timestamp_block_all_motion(self):
        for mutate in (lambda o, p: replace(o, timestamp=-100), lambda o, p: replace(o, frame_id=1)):
            servo, plant, _ = self.run_plant(mutate=mutate)
            with self.assertRaisesRegex(MeasurementError, "stale"):
                servo.run()

    def test_tool_axis_drift_is_not_learned(self):
        def drift(obs, plant):
            plant.pose[2] += 1
            return replace(obs, tcp=plant.pose.copy())
        servo, plant, _ = self.run_plant(mutate=drift)
        with self.assertRaisesRegex(RuntimeError, "XYZ"):
            servo.run()
        self.assertFalse(plant.moves)

    def test_actual_feedback_must_reach_target(self):
        servo, _, events = self.run_plant(move_override=lambda target: None)
        with self.assertRaisesRegex(RuntimeError, "not reached"):
            servo.run()
        self.assertFalse(any(e["event"] == "broyden" for e in events))

    def test_detection_failure_during_probe_does_not_trigger_blind_rollback(self):
        def fail(obs, plant):
            if plant.moves:
                raise MeasurementError("occluded")
            return obs
        servo, plant, _ = self.run_plant(mutate=fail)
        with self.assertRaisesRegex(MeasurementError, "occluded"):
            servo.run()
        self.assertEqual(len(plant.moves), 1)

    def test_observability_and_total_travel_limits(self):
        servo, plant, _ = self.run_plant(cfg=ControlSettings(max_travel_deg=.4))
        with self.assertRaisesRegex(RuntimeError, "travel"):
            servo.run()
        servo, _, _ = self.run_plant()
        with self.assertRaisesRegex(MeasurementError, "unobservable"):
            servo.run(np.zeros((2, 2)))
        servo4, _, _ = self.run_plant(ShapeFeatures(camera()))
        with self.assertRaisesRegex(MeasurementError, "unobservable"):
            servo4.validate_jacobian(np.ones((4, 2)))

    def test_high_noise_and_non_improvement_abort(self):
        servo, plant, _ = self.run_plant(mutate=lambda o, p: replace(o, noise=[.1, .1]))
        with self.assertRaisesRegex(MeasurementError, "noise"):
            servo.run()
        self.assertFalse(plant.moves)
        servo, _, _ = self.run_plant(cfg=ControlSettings(max_non_improving=2),
            mutate=lambda o, p: replace(o, feature=[.1, .1], noise=[0, 0]))
        with self.assertRaisesRegex(RuntimeError, "not improving"):
            servo.run(np.eye(2))  # deliberately wrong signs/scales


