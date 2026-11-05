"""Numerical and closed-loop tests; all motion/camera feedback is simulated."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ibvs_jacobian_estimator import (
    JacobianSettings, OnlineJacobianEstimator, calculate_dls_step,
    estimate_initial_jacobian, load_calibration, save_calibration, validate_matrix,
)
from online_jacobian_servo import run_calibration, run_online_servo


class SimulatedPlant:
    def __init__(self):
        self.J = np.array([[-20., 1.5], [2., 18.]])
        self.q0 = np.array([100., 200., 140., 90., -88.5, 90.])
        self.q = self.q0.copy()
        self.center = np.array([1200., 800.])
        self.t = 0
        self.moves = []
        self.events = []

    def measure(self, hint):
        self.t += 1
        uv = self.center + self.J @ (self.q[:2] - self.q0[:2])
        return {"tcp": self.q.tolist(), "center": uv.tolist(), "timestamp": float(self.t),
                "frame_id": self.t, "image_size": [2560, 1440], "conf": .95,
                "valid": True, "settled": True, "noise_px": .05, "bbox_wh": [160., 158.]}

    def move(self, target):
        target = np.asarray(target)
        if np.linalg.norm(target[:2] - self.q[:2]) > 1.000001:
            raise AssertionError("motion exceeds XY limit")
        np.testing.assert_allclose(target[2:], self.q[2:])
        self.moves.append(target.tolist())
        self.q = target.copy()

    def log(self, event, **data):
        self.events.append({"event": event, **data})

    def clock(self):
        return float(self.t)


class OnlineJacobianTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "initial.json"
        self.cfg = JacobianSettings(mode="online", camera_id="sim-camera", hole_diameter_mm=8.,
                                    probe_mm=.25, calibration_path=str(self.path))
        self.plant = SimulatedPlant()
        self.tool = {"name": "sim-tool", "offset": [1., 2., 3., 0., 0., 0.]}
        self.meta = {"camera_id": "sim-camera", "camera_index": 0,
                     "hole_diameter_mm": 8., "image_size": [2560, 1440],
                     "target_uv": [1280., 720.], "tool": self.tool,
                     "reference_tcp": self.plant.q0.tolist()}

    def save(self, J=None):
        save_calibration(self.path, self.plant.J if J is None else J, self.meta)

    def run_servo(self, **kwargs):
        callbacks = dict(measure=self.plant.measure, move=self.plant.move, tool=self.tool,
                         log=self.plant.log, clock=self.plant.clock)
        callbacks.update(kwargs)
        return run_online_servo(self.cfg, **callbacks)

    def pair(self):
        before = self.plant.measure(None)
        self.plant.q[:2] += [.3, .2]
        after = self.plant.measure(None)
        after["commanded_tcp"] = after["tcp"].copy()
        return before, after

    def test_fd_fits_full_matrix_with_coupled_actual_motion(self):
        offsets = [[0, 0], [.25, .015], [-.25, -.01], [.02, .25], [-.01, -.25]]
        samples = [{"q": q, "s": (self.plant.J @ q + [300, 400]).tolist(), "noise_px": .05}
                   for q in map(np.asarray, offsets)]
        J = estimate_initial_jacobian(samples, self.cfg)
        np.testing.assert_allclose(J, self.plant.J, atol=1e-10)

    def test_fd_rejects_collinear_or_too_small_excitation(self):
        for offsets in ([[0, 0], [.3, 0], [-.3, 0]], [[0, 0], [.001, 0], [0, .001]]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                estimate_initial_jacobian([{"q": q, "s": (self.plant.J @ q).tolist()} for q in offsets], self.cfg)

    def test_fd_rejects_visual_noise_and_inconsistent_fit(self):
        samples = [{"q": q, "s": (self.plant.J @ q).tolist(), "noise_px": 5.}
                   for q in map(np.asarray, [[0, 0], [.25, 0], [-.25, 0], [0, .25], [0, -.25]])]
        with self.assertRaises(ValueError):
            estimate_initial_jacobian(samples, self.cfg)
        for sample in samples:
            sample["noise_px"] = .05
        samples[0]["s"] = [100, 100]
        with self.assertRaises(ValueError):
            estimate_initial_jacobian(samples, self.cfg)

    def test_dls_has_correct_sign_and_preserves_vector_direction_when_clipped(self):
        e = np.array([100., -70.])
        dq = calculate_dls_step(self.plant.J, e, self.cfg)
        self.assertAlmostEqual(np.linalg.norm(dq), 1.)
        self.assertLess(np.linalg.norm(e + self.plant.J @ dq), np.linalg.norm(e))
        self.assertGreater(np.dot(dq, -np.linalg.solve(self.plant.J, e)), 0)

    def test_invalid_matrix_and_parameters_block_control(self):
        for J in (np.zeros((2, 2)), np.eye(2) * 1e-6, np.eye(2) * 1e5, [[np.nan, 0], [0, 1]]):
            with self.subTest(J=J), self.assertRaises(ValueError):
                calculate_dls_step(J, [1, 2], self.cfg)
        for values in ({"beta": 2}, {"gain": -1}, {"max_step_mm": 2}, {"sample_frames": 1}, {"hole_diameter_mm": None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.cfg, **values).validate()

    def test_broyden_uses_actual_displacement_and_reduces_prediction_residual(self):
        initial = self.plant.J * .85
        estimator = OnlineJacobianEstimator(initial, self.cfg)
        before, after = self.pair()
        dq = np.asarray(after["tcp"][:2]) - before["tcp"][:2]
        ds = np.asarray(after["center"]) - before["center"]
        result = estimator.update(before, after, robot_settled=True, now=self.plant.clock())
        self.assertTrue(result["accepted"])
        self.assertLess(np.linalg.norm(ds - estimator.J @ dq), np.linalg.norm(ds - initial @ dq))
        # Getter returns a copy, so control cannot accidentally mutate the estimate.
        matrix = estimator.get_jacobian()
        matrix[:] = 0
        self.assertGreater(np.linalg.norm(estimator.J), 0)

    def test_update_gates_leave_last_trusted_matrix_unchanged(self):
        changes = [
            ("stale", lambda b, a: a.update(timestamp=-100.)),
            ("duplicate", lambda b, a: a.update(frame_id=b["frame_id"])),
            ("nan", lambda b, a: a.update(center=[np.nan, 100.])),
            ("confidence", lambda b, a: a.update(conf=.1)),
            ("resolution", lambda b, a: a.update(image_size=[1280, 720])),
            ("height", lambda b, a: a["tcp"].__setitem__(2, 141.)),
            ("orientation", lambda b, a: a["tcp"].__setitem__(3, 91.)),
            ("feedback", lambda b, a: a["commanded_tcp"].__setitem__(0, 150.)),
            ("outlier", lambda b, a: a.update(center=[2000., 1000.])),
            ("small_move", lambda b, a: a.update(tcp=b["tcp"].copy())),
        ]
        for name, mutate in changes:
            with self.subTest(name=name):
                self.plant = SimulatedPlant()
                before, after = self.pair()
                estimator = OnlineJacobianEstimator(self.plant.J, self.cfg)
                mutate(before, after)
                result = estimator.update(before, after, robot_settled=True, now=self.plant.clock())
                self.assertFalse(result["accepted"])
                np.testing.assert_array_equal(estimator.J, self.plant.J)

    def test_unsettled_robot_cannot_update(self):
        before, after = self.pair()
        estimator = OnlineJacobianEstimator(self.plant.J, self.cfg)
        self.assertEqual(estimator.update(before, after, robot_settled=False, now=self.plant.clock())["reason"], "robot_not_settled")

    def test_matrix_scale_and_reference_drift_are_bounded(self):
        estimator = OnlineJacobianEstimator(self.plant.J, replace(self.cfg, max_reference_change=.001))
        before, after = self.pair()
        after["center"][0] += 2
        self.assertEqual(estimator.update(before, after, robot_settled=True, now=self.plant.clock())["reason"], "reference_drift")

    def test_calibration_roundtrip_and_compatibility_checks(self):
        self.save()
        J, meta = load_calibration(self.path, self.cfg, tool=self.tool, tcp=self.plant.q0)
        np.testing.assert_array_equal(J, self.plant.J)
        for values in ({"camera_id": "other"}, {"camera_index": 1}, {"hole_diameter_mm": 9.},
                       {"image_size": [1280, 720]}, {"target_uv": [1279, 720]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                load_calibration(self.path, replace(self.cfg, **values))
        with self.assertRaises(ValueError):
            load_calibration(self.path, self.cfg, tool={"name": "wrong"})
        for index in (2, 3):
            q = self.plant.q0.copy()
            q[index] += 4
            with self.assertRaises(ValueError):
                load_calibration(self.path, self.cfg, tcp=q)

    def test_failed_save_preserves_previous_calibration(self):
        self.save()
        trusted = self.path.read_bytes()
        with self.assertRaises(ValueError):
            save_calibration(self.path, [[np.nan, 0], [0, 1]], self.meta)
        self.assertEqual(self.path.read_bytes(), trusted)
        self.assertFalse(list(self.path.parent.glob("*.tmp")))

    def test_calibration_is_bounded_returns_to_origin_and_preserves_initial_file(self):
        cfg = replace(self.cfg, mode="calibrate")
        J = run_calibration(cfg, measure=self.plant.measure, move=self.plant.move,
                            tool=self.tool, log=self.plant.log, clock=self.plant.clock)
        np.testing.assert_allclose(J, self.plant.J, atol=1e-10)
        self.assertEqual(len(self.plant.moves), 8)
        np.testing.assert_array_equal(self.plant.q, self.plant.q0)
        initial = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            run_calibration(cfg, measure=self.plant.measure, move=self.plant.move,
                            tool=self.tool, log=self.plant.log, clock=self.plant.clock)
        self.assertEqual(initial, self.path.read_bytes())

    def test_missing_calibration_blocks_motion(self):
        with self.assertRaises(FileNotFoundError):
            self.run_servo()
        self.assertFalse(self.plant.moves)

    def test_calibration_loss_stops_without_blind_return_or_save(self):
        cfg = replace(self.cfg, mode="calibrate")
        count = 0
        def measure(hint):
            nonlocal count
            count += 1
            return self.plant.measure(hint) if count == 1 else None
        with self.assertRaises(RuntimeError):
            run_calibration(cfg, measure=measure, move=self.plant.move,
                            tool=self.tool, log=self.plant.log, clock=self.plant.clock)
        self.assertEqual(len(self.plant.moves), 1)
        self.assertFalse(self.path.exists())

    def test_online_convergence_freezes_J_and_never_overwrites_initial(self):
        self.save(self.plant.J * .85)
        initial_bytes = self.path.read_bytes()
        aligned = self.run_servo()
        np.testing.assert_array_equal(aligned[2:], self.plant.q0[2:])
        self.assertGreater(len(self.plant.moves), 1)
        error = np.asarray(self.plant.measure(None)["center"]) - self.cfg.target_uv
        self.assertLess(np.linalg.norm(error), self.cfg.align_tol_px)
        self.assertTrue(any(e["event"] == "jacobian_update" and e["accepted"] for e in self.plant.events))
        self.assertEqual(self.plant.events[-1]["event"], "aligned_frozen")
        self.assertEqual(self.path.read_bytes(), initial_bytes)
        self.assertTrue(Path(str(self.path) + ".online.json").exists())

    def test_fixed_comparison_uses_same_initial_J_without_learning(self):
        self.save(self.plant.J * .85)
        self.cfg = replace(self.cfg, mode="fixed")
        self.run_servo()
        self.assertFalse(any(e["event"] == "jacobian_update" for e in self.plant.events))
        for event in self.plant.events:
            if event["event"] == "motion":
                np.testing.assert_array_equal(event["J_used"], self.plant.J * .85)
        self.assertFalse(Path(str(self.path) + ".online.json").exists())

    def test_detection_gap_breaks_update_pair(self):
        self.save()
        original = self.plant.measure
        count = 0
        def measure(hint):
            nonlocal count
            count += 1
            return None if count == 2 else original(hint)
        self.run_servo(measure=measure)
        failures = [i for i, e in enumerate(self.plant.events) if e["event"] == "detection_failed"]
        self.assertEqual(len(failures), 1)
        next_observation = next(e for e in self.plant.events[failures[0] + 1:] if e["event"] in ("observation", "jacobian_update"))
        self.assertEqual(next_observation["event"], "observation")

    def test_lost_target_never_moves_and_never_saves_learned_J(self):
        self.save()
        with self.assertRaisesRegex(RuntimeError, "连续"):
            self.run_servo(measure=lambda hint: None)
        self.assertFalse(self.plant.moves)
        self.assertFalse(Path(str(self.path) + ".online.json").exists())

    def test_untrusted_initial_observation_cannot_trigger_motion(self):
        self.save()
        for values in ({"settled": False}, {"timestamp": -100.}, {"image_size": [1280, 720]}, {"conf": .1}):
            with self.subTest(values=values):
                def measure(hint):
                    sample = self.plant.measure(hint)
                    sample.update(values)
                    return sample
                with self.assertRaises(RuntimeError):
                    self.run_servo(measure=measure)
                self.assertFalse(self.plant.moves)

    def test_repeated_frame_stops_further_motion(self):
        self.save()
        def measure(hint):
            sample = self.plant.measure(hint)
            sample["frame_id"] = 1
            return sample
        with self.assertRaisesRegex(RuntimeError, "重复"):
            self.run_servo(measure=measure)
        self.assertEqual(len(self.plant.moves), 1)

    def test_non_improving_feedback_stops_closed_loop(self):
        self.save()
        def measure(hint):
            sample = self.plant.measure(hint)
            sample["center"] = self.plant.center.tolist()
            return sample
        with self.assertRaisesRegex(RuntimeError, "未改善"):
            self.run_servo(measure=measure)
        self.assertEqual(len(self.plant.moves), self.cfg.max_non_improving_steps)

    def test_too_small_probe_is_rejected_before_motion(self):
        with self.assertRaises(ValueError):
            run_calibration(replace(self.cfg, mode="calibrate", probe_mm=.001),
                            measure=self.plant.measure, move=self.plant.move,
                            tool=self.tool, log=self.plant.log)
        self.assertFalse(self.plant.moves)

    def test_field_probe_limit_is_distinct_from_actual_probe_and_step_limit(self):
        cfg = replace(self.cfg, mode="calibrate", hole_diameter_mm=11.,
                      probe_mm=.5, max_probe_mm=30.)
        cfg.validate()
        with self.assertRaisesRegex(ValueError, "max_probe_mm"):
            replace(cfg, max_probe_mm=.4).validate()
        with self.assertRaisesRegex(ValueError, "max_step_mm"):
            replace(cfg, probe_mm=30.).validate()
        for invalid in (0., -1., float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(cfg, max_probe_mm=invalid).validate()


if __name__ == "__main__":
    unittest.main()
