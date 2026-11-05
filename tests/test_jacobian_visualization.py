"""Traceability and export tests: no real hardware or service calls."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ibvs_jacobian_estimator import JacobianSettings, OnlineJacobianEstimator, calculate_dls_step
from online_jacobian_servo import JacobianRunLog
from plot_online_jacobian import make_specs, read_events, export_figures


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def sample(self, tcp, center, stamp):
        return dict(tcp=tcp, center=center, timestamp=float(stamp), frame_id=stamp,
                    image_size=[2560, 1440], noise_px=.05, conf=.9, valid=True, settled=True)

    def update(self, *, reject=False):
        cfg = JacobianSettings(mode="fixed")
        estimator = OnlineJacobianEstimator([[-20., 0.], [0., 20.]], cfg)
        before = self.sample([0., 0., 140., 0., 0., 0.], [1200., 700.], 1)
        after = self.sample([.5, 0., 140., 0., 0., 0.], [1250. if reject else 1191., 701.], 2)
        return estimator.update(before, after, robot_settled=True, now=2.)

    def test_rejected_update_preserves_full_available_diagnostics(self):
        result = self.update(reject=True)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "residual_outlier")
        self.assertEqual(result["ds"], [50., 1.])
        self.assertEqual(result["predicted_ds_before"], [-10., 0.])
        self.assertEqual(result["residual"], [60., 1.])
        self.assertEqual(result["J_before"], result["J"])
        self.assertEqual(result["relative_update"], 0.)
        self.assertIsNone(result["J_candidate"])
        self.assertEqual(result["gates"]["max_residual_px"], 30.)

    def test_accepted_update_logs_candidate_and_applied_matrix(self):
        result = self.update()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["J_candidate"], result["J"])
        self.assertLess(result["residual_after_norm"], result["residual_norm"])
        self.assertGreater(result["relative_update"], 0)
        self.assertEqual(result["update_count"], 1)

    def test_control_audit_preserves_raw_and_limited_step(self):
        step, audit = calculate_dls_step([[-20, 0], [0, 20]], [100, -80],
                                        JacobianSettings(mode="fixed"), return_diagnostics=True)
        np.testing.assert_allclose(step, np.asarray(audit["raw_delta_xy"]) * audit["limit_scale"])
        self.assertAlmostEqual(np.linalg.norm(step), 1.)
        self.assertGreater(np.linalg.norm(audit["raw_delta_xy"]), 1.)

    def test_jsonl_mirrors_every_event_and_preserves_nonfinite_provenance(self):
        log = JacobianRunLog(self.root, source="simulation")
        with patch("run_logging.write_run_event") as mirror:
            log("test", matrix=np.eye(2), invalid=np.nan, infinite=float("inf"))
            log("jacobian_update", iteration=1, **self.update(reject=True))
        events, raw, warnings = read_events(log.path)
        self.assertEqual([e["sequence"] for e in events], [1, 2])
        self.assertEqual(mirror.call_count, 2)
        self.assertIsNone(events[0]["invalid"])
        self.assertEqual(len(events[0]["nonfinite_values"]), 2)
        self.assertFalse(warnings)
        self.assertIn(b'"schema_version": 2', raw)

    def test_incomplete_tail_is_reported_but_corrupt_middle_is_not_ignored(self):
        path = self.root / "partial.jsonl"
        path.write_bytes(b'{"event":"start"}\n{"event":')
        events, _, warnings = read_events(path)
        self.assertEqual(len(events), 1)
        self.assertTrue(warnings)
        path.write_bytes(b'{"event":"start"}\nBROKEN\n{"event":"end"}\n')
        with self.assertRaises(ValueError):
            read_events(path)

    def test_specs_keep_every_rejected_point_and_separate_matrix_elements(self):
        events = [{"event": "jacobian_update", "iteration": i, **self.update(reject=i == 1)} for i in range(3)]
        specs = {s["name"]: s for s in make_specs(events)}
        self.assertEqual(len(specs), 19)
        self.assertTrue({"J11", "J12", "J21", "J22"} <= specs.keys())
        self.assertEqual(specs["update_gate"]["series"][0][1], [1, 0, 1])
        self.assertEqual(len(specs["residual"]["series"][0][1]), 3)

    def test_export_is_bilingual_one_axes_per_file_and_csv_retains_rejections(self):
        log = JacobianRunLog(self.root, source="simulation")
        log("servo_start", initial_J=[[-20., 0.], [0., 20.]], settings={"target_uv": [1280, 720]})
        for i in range(2):
            log("jacobian_update", iteration=i, **self.update(reject=bool(i)))
        log("aborted", message="synthetic failure for QA")
        import matplotlib.figure
        axes_counts = []
        original = matplotlib.figure.Figure.savefig
        def save(fig, *args, **kwargs):
            axes_counts.append(len(fig.axes))
            return original(fig, *args, **kwargs)
        with patch.object(matplotlib.figure.Figure, "savefig", save):
            manifest_path = export_figures(log.path, formats=("png",))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(axes_counts, [1] * 38)
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(manifest["source"], "simulation")
        for lang in ("zh", "en"):
            self.assertEqual(len(list((manifest_path.parent / lang).glob("*.png"))), 19)
        self.assertIn("residual_outlier", (manifest_path.parent / "updates.csv").read_text(encoding="utf-8-sig"))
        self.assertEqual(len(read_events(manifest_path.parent / "events_snapshot.jsonl")[0]), 4)

    def test_plotting_failure_keeps_log_and_is_recorded(self):
        log = JacobianRunLog(self.root)
        log("aborted", message="test")
        with patch("plot_online_jacobian.export_figures", side_effect=RuntimeError("font missing")), contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(log.export_figures())
        events, _, _ = read_events(log.path)
        self.assertEqual(events[-1]["event"], "figure_export_failed")


if __name__ == "__main__":
    unittest.main()
