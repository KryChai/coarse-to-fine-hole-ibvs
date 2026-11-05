import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from circle_visualization import CircleRun, export_run, load_events, sample_rows
from circle_preview import draw_circle_preview


class CircleVisualizationTests(unittest.TestCase):
    def test_background_export_returns_before_plotting_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def slow_export(path):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test export timed out")

        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), \
                patch("circle_visualization.export_run", side_effect=slow_export) as export:
            run = CircleRun(directory)
            run("alignment_complete", tcp=[0.] * 6)
            worker = run.finish(background=True)
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(worker.is_alive())
                self.assertFalse(worker.daemon)
                self.assertIs(run.finish(background=True), worker)
                self.assertEqual(load_events(run.path)[-1]["event"], "alignment_complete")
                export.assert_called_once_with(run.path)
            finally:
                release.set()
                worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    def test_background_export_failure_is_recorded_without_losing_data(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"), \
                patch("circle_visualization.export_run", side_effect=RuntimeError("plot failure")):
            run = CircleRun(directory)
            run("alignment_complete", tcp=[0.] * 6)
            worker = run.finish(background=True)
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            events = load_events(run.path)
            self.assertEqual([e["event"] for e in events], ["alignment_complete", "export_failed"])
            self.assertEqual(events[-1]["reason"], "plot failure")

    def test_full_response_persists_without_console_dump_or_duplicate_measurements(self):
        response = {"data": {"holes": [{"visualization": {
            "edge_points_px": [[1., 2.]] * 1000}}]}}
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print") as console:
            run = CircleRun(directory)
            run("circle_response", response=response)
            run("fine_measurement", center=[10., 20.], error_px=.5)
            console.assert_not_called()
            self.assertEqual(load_events(run.path)[0]["response"], response)
            run("alignment_failed", reason="measurement rejected")
            self.assertIn("measurement rejected", console.call_args.args[0])

    def test_preview_scales_original_pixels_and_ignores_wrong_size(self):
        view = np.zeros((100, 200, 3), dtype=np.uint8)
        data = {"image_size": {"width": 400, "height": 200}, "holes": [{
            "status": "ok", "circle": {"center_x": 200., "center_y": 100.},
            "visualization": {"outline_points_px": [[180., 100.], [200., 80.], [220., 100.]]}}]}
        draw_circle_preview(view, data, (400, 200))
        self.assertTrue(np.any(view[45:55, 95:105]))
        blank = np.zeros_like(view)
        draw_circle_preview(blank, data, (800, 400))
        self.assertFalse(np.any(blank))
        draw_circle_preview(blank, {"image_size": data["image_size"], "holes": [None]}, (400, 200))

    def test_record_is_durable_strict_json_and_missing_metrics_stay_missing(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            run = CircleRun(directory, source="simulation")
            run("fine_measurement", iteration=1, center=np.array([10., 20.]), target_uv=[10., 20.],
                error_px=0., quality={"rmse_px": float("nan")})
            events = load_events(run.path)
            self.assertEqual(events[0]["sequence"], 1)
            self.assertEqual(events[0]["source"], "simulation")
            rows = sample_rows(events)
            self.assertIsNone(rows[0]["rmse_px"])
            self.assertIsNone(rows[0]["confidence"])
            self.assertEqual(rows[0]["error_px"], 0.)
            self.assertNotIn("NaN", run.path.read_text(encoding="utf8"))

    def test_export_failure_preserves_original_data(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            run = CircleRun(directory)
            run("alignment_failed", reason="motion failure")
            with patch("circle_visualization.export_run", side_effect=RuntimeError("font")):
                run.finish()
            self.assertEqual([e["event"] for e in load_events(run.path)], ["alignment_failed", "export_failed"])

    def test_bilingual_individual_exports_have_manifest_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch("builtins.print"):
            run = CircleRun(directory, source="simulation")
            run("fine_start", settings={"tolerance_px": .7})
            run("fine_measurement", iteration=1, center=[10., 20.], target_uv=[10., 20.], error_px=0., quality={})
            run("fine_rejected", reason="review")
            run("alignment_complete", tcp=[0.]*6)
            manifest = export_run(run.path)
            data = json.loads(manifest.read_text(encoding="utf8"))
            self.assertEqual(data["status"], "alignment_complete")
            self.assertEqual(len(data["files"]), 96)  # 16 independent plots * 2 languages * 3 formats
            self.assertEqual((manifest.parent/"events_snapshot.jsonl").read_bytes(), run.path.read_bytes())
            for path in data["files"]:
                self.assertTrue((manifest.parent/path).is_file())
            for language in ("zh", "en"):
                svg = (manifest.parent/language/"error_px.svg").read_text(encoding="utf8")
                self.assertIn("模拟数据" if language == "zh" else "SIMULATED DATA", svg)
