"""Deterministic local-image-model tests; no hardware-performance claims."""

from dataclasses import replace
import unittest

import numpy as np

from servo_control import ServoSettings, ServoStepController


def converge(jacobian, initial_error, mode):
    error = np.array(initial_error, dtype=float)
    controller = ServoStepController(jacobian, ServoSettings(mode=mode))
    norms = []
    for iteration in range(301):
        if np.linalg.norm(error) < controller.settings.align_tol_px:
            return iteration, norms
        command = controller.step(error)
        norms.append(float(np.linalg.norm(command.selected)))
        error += jacobian @ command.selected
    raise AssertionError("Synthetic local image model did not converge")


class ServoControlTests(unittest.TestCase):
    def test_fixed_mode_preserves_formula_and_global_norm_cap(self):
        controller = ServoStepController(np.diag([-1, 1]), ServoSettings(mode="fixed"))
        small = controller.step([10, -5])
        np.testing.assert_allclose(small.selected, [0.3, 0.15])
        large = controller.step([100, -100])
        np.testing.assert_allclose(large.requested, [3, 3])
        self.assertAlmostEqual(np.linalg.norm(large.selected), 1)
        self.assertEqual(large.effective_gain, 0.03)

    def test_error_schedule_increases_gain_without_exceeding_bounds(self):
        gains = [ServoStepController(np.eye(2)).step([error, 0]).effective_gain
                 for error in (2, 5, 10, 20, 40, 100)]
        self.assertEqual(gains, sorted(gains))
        self.assertAlmostEqual(gains[0], 0.03)
        self.assertAlmostEqual(gains[-1], 0.15)

    def test_physical_step_ceiling_shrinks_near_target(self):
        near = ServoStepController(np.eye(2)).step([3, 0])
        far = ServoStepController(np.eye(2)).step([30, 0])
        self.assertLess(near.step_limit_mm, far.step_limit_mm)
        self.assertLessEqual(np.linalg.norm(near.selected), near.step_limit_mm)
        self.assertEqual(far.step_limit_mm, 1)

    def test_error_increase_reduces_scale_and_improvement_restores_it_gradually(self):
        controller = ServoStepController(np.eye(2))
        controller.step([20, 0])
        reduced = controller.step([30, 0])
        self.assertTrue(reduced.backoff_applied)
        self.assertEqual(reduced.gain_scale, 0.5)
        recovered = controller.step([20, 0])
        self.assertAlmostEqual(recovered.gain_scale, 0.55)
        self.assertFalse(recovered.backoff_applied)

    def test_direction_reversal_backs_off_even_if_error_magnitude_decreases(self):
        controller = ServoStepController(np.eye(2))
        controller.step([20, 0])
        command = controller.step([-5, 0])
        self.assertTrue(command.backoff_applied)
        self.assertEqual(command.gain_scale, 0.5)

    def test_detection_loss_clears_response_history_and_marks_next_command(self):
        controller = ServoStepController(np.eye(2))
        controller.step([10, 0])
        controller.detection_lost()
        command = controller.step([100, 0])
        self.assertEqual(command.gain_scale, 0.5)
        self.assertTrue(command.backoff_applied)
        self.assertIsNotNone(controller.previous_error)

    def test_repeated_backoff_has_a_floor_and_all_steps_obey_the_global_cap(self):
        controller = ServoStepController(np.array([[-2.0, 0.3], [0.1, 3.0]]))
        rng = np.random.default_rng(412)
        for error in rng.uniform(-1000, 1000, size=(100, 2)):
            command = controller.step(error)
            self.assertGreaterEqual(command.gain_scale, 0.125)
            self.assertLessEqual(command.gain_scale, 1)
            self.assertGreater(command.effective_gain, 0)
            self.assertLessEqual(command.effective_gain, 0.15)
            self.assertLessEqual(np.linalg.norm(command.selected), command.step_limit_mm + 1e-12)
            self.assertLessEqual(command.step_limit_mm, 1)

    def test_aligned_error_produces_no_minimum_step_or_dither(self):
        for error in ([0, 0], [1, 1]):
            np.testing.assert_array_equal(ServoStepController(np.eye(2)).step(error).selected, [0, 0])

    def test_each_alignment_gets_fresh_history(self):
        first = ServoStepController(np.eye(2))
        first.detection_lost()
        self.assertEqual(ServoStepController(np.eye(2)).step([20, 0]).gain_scale, 1)

    def test_bad_errors_and_jacobians_are_rejected(self):
        for matrix in (np.zeros((2, 2)), np.ones((3, 3)), [[float("nan"), 0], [0, 1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                ServoStepController(matrix)
        for error in ([1], [[1], [2]], [float("inf"), 1]):
            with self.subTest(error=error), self.assertRaises(ValueError):
                ServoStepController(np.eye(2)).step(error)

    def test_invalid_tuning_is_rejected_before_any_step(self):
        cases = ({"gain": 0}, {"max_gain": 1}, {"max_gain": 0.01},
                 {"transition_error_px": 1}, {"slowdown_error_px": 2},
                 {"near_step_ratio": 0}, {"max_step_mm": -1}, {"gain": float("nan")},
                 {"backoff_factor": 1}, {"recovery_factor": 1}, {"mode": "unknown"})
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ServoStepController(np.eye(2), replace(ServoSettings(), **changes))

    def test_adaptive_converges_faster_on_matched_local_models(self):
        for jacobian, error in ((np.diag([-1., 1.]), [20, -12]),
                                (np.diag([-12., 10.]), [90, -65]),
                                (np.array([[1., -9.], [8., 0.5]]), [-50, 30])):
            fixed_steps, fixed_norms = converge(jacobian, error, "fixed")
            adaptive_steps, adaptive_norms = converge(jacobian, error, "adaptive")
            with self.subTest(jacobian=jacobian):
                self.assertLess(adaptive_steps, fixed_steps)
                self.assertLessEqual(max(fixed_norms + adaptive_norms), 1 + 1e-12)

    def test_saturation_is_not_misrepresented_as_unbounded_speedup(self):
        fixed = ServoStepController(np.eye(2), ServoSettings(mode="fixed")).step([100, 0])
        adaptive = ServoStepController(np.eye(2)).step([100, 0])
        np.testing.assert_allclose(fixed.selected, adaptive.selected)


if __name__ == "__main__":
    unittest.main()
