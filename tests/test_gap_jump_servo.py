"""Fault-oriented tests of the SAME callback engine used on hardware."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gap_jump_servo import GapSettings, Observation, associate, run_alignment
from gap_simulation import SyntheticPlant, simulate


class GapEngineTests(unittest.TestCase):
    def run_plant(self, cfg=None, scenario="feature_bias", mutate=None):
        cfg = cfg or GapSettings()
        plant = SyntheticPlant(cfg, 7, scenario)
        events = []
        def measure(stage, hint):
            obs = plant.measure(stage, hint)
            return mutate(obs, plant) if mutate and obs is not None else obs
        result = run_alignment(cfg, measure=measure, move=plant.move,
            get_pose=lambda: plant.pose.tolist(), clock=lambda: plant.now,
            record=lambda event, **data: events.append(dict(event=event, **data)))
        return result, plant, events

    def test_full_loop_is_bounded_and_never_learns_across_feature_switch(self):
        _, plant, events = self.run_plant()
        self.assertLess(np.linalg.norm(plant.truth()-plant.cfg.target_uv), 1.)
        self.assertTrue(any(e["event"] == "handoff_validated" for e in events))
        handoff = next(e for e in events if e["event"] == "handoff_validated")
        np.testing.assert_allclose(handoff["fine_J"], plant.cfg.fine_J)
        updates = [e for e in events if e["event"] == "jacobian_update"]
        self.assertTrue(updates)
        self.assertTrue(all(not e["cross_feature"] for e in updates))
        controls = [e for e in events if e["event"] == "control"]
        for e in controls:
            self.assertLessEqual(np.linalg.norm(e["dls"]["limited_delta_xy"]),
                                 .15+1e-9 if e["stage"] == "fine" else 1.+1e-9)
        handoff = [i for i, e in enumerate(events) if e["event"].startswith("handoff_")]
        self.assertFalse(any(e["event"] == "control" for e in events[min(handoff):max(handoff)+1]))

    def test_missing_detection_breaks_learning_pair_and_can_recover(self):
        _, _, events = self.run_plant(scenario="dropout")
        self.assertTrue(any(e["event"] == "pair_discarded" and e["reason"] == "detection_gap" for e in events))
        for e in events:
            if e["event"] == "jacobian_update":
                self.assertFalse(e["before_frame"] < 10 and e["after_frame"] > 11)

    def test_ambiguous_target_aborts_instead_of_choosing_high_confidence_neighbour(self):
        with self.assertRaisesRegex(RuntimeError, "failures"):
            self.run_plant(scenario="ambiguous")

    def test_unstable_feature_offset_blocks_fine_motion(self):
        with self.assertRaisesRegex(RuntimeError, "offset is unstable"):
            self.run_plant(scenario="unstable_handoff")

    def test_duplicate_frame_blocks_control(self):
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.run_plant(scenario="stale")

    def test_target_identity_change_aborts(self):
        def mutate(obs, plant):
            if plant.frame > 4:
                obs.target_id = "neighbour"
            return obs
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.run_plant(mutate=mutate)

    def test_pose_or_tool_plane_change_aborts(self):
        def mutate(obs, plant):
            if plant.frame == 3:
                plant.pose[2] = obs.tcp[2] = 1.
            return obs
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            self.run_plant(mutate=mutate)

    def test_motion_not_actually_arriving_aborts(self):
        plant = SyntheticPlant(GapSettings())
        with self.assertRaisesRegex(RuntimeError, "arrival"):
            run_alignment(plant.cfg, measure=plant.measure, move=lambda _: None,
                          get_pose=lambda: plant.pose, clock=lambda: plant.now)

    def test_unsafe_profile_is_rejected_before_any_measurement_on_hardware(self):
        measure = Mock()
        with self.assertRaisesRegex(ValueError, "simulation-only"):
            run_alignment(GapSettings(profile="P3"), measure=measure, move=Mock(), get_pose=Mock(), hardware=True)
        measure.assert_not_called()

    def test_fine_learning_can_be_unplugged(self):
        _, _, events = self.run_plant(GapSettings(fine_learning=False))
        self.assertFalse(any(e["event"] == "jacobian_update" and e["stage"] == "fine" for e in events))

    def test_baseline_has_no_learning_or_fine_stage(self):
        _, _, events = self.run_plant(GapSettings(profile="P0"))
        self.assertFalse(any(e["event"] in ("jacobian_update", "feature_switch") for e in events))

    def test_naive_ablation_is_labelled_and_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            unsafe = simulate("P3", seed=7, root=tmp)
            safe = simulate("P4", seed=7, root=tmp)
        self.assertEqual(safe["cross_feature_updates"], 0)
        self.assertGreater(unsafe["cross_feature_pairs_attempted"], 0)
        # Existing numerical gates can also reject the defective pair; never
        # claim contamination/benefit solely because P3 was constructed to try it.
        self.assertEqual(unsafe["source"], "simulation")

    def test_no_quality_fallback_after_association(self):
        chosen = associate([dict(anchor=[10, 10], confidence=.8, quality=False),
                            dict(anchor=[30, 30], confidence=.99, quality=True)],
                           np.array([10, 10]), 40, 3)
        self.assertFalse(chosen["quality"])

    def test_invalid_config_and_image_geometry(self):
        for cfg in (GapSettings(fine_step_mm=.2), GapSettings(coarse_gain=float("nan")),
                    GapSettings(stable_frames=2), GapSettings(image_size=(0, 1440)),
                    GapSettings(fine_J=((0, 0), (0, 0)))):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                cfg.validate()

    def test_custom_image_size_is_used_by_estimator_gates(self):
        cfg = GapSettings(image_size=(640, 480), target_uv=(320., 240.))
        _, _, events = self.run_plant(cfg)
        self.assertFalse(any(e["event"] == "jacobian_update" and e["reason"] == "resolution_changed"
                             for e in events))

    def test_handoff_motion_is_rejected(self):
        def mutate(obs, plant):
            if obs.feature == "fine" and plant.fine_frames == 2:
                plant.pose[0] += .03
                obs.tcp = plant.pose.tolist()
            return obs
        with self.assertRaisesRegex(RuntimeError, "pose changed during feature handoff"):
            self.run_plant(mutate=mutate)


