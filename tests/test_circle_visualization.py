import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from circle_visualization import CircleRun, export_run, load_events, sample_rows
from circle_preview import draw_circle_preview


class CircleVisualizationTests(unittest.TestCase):
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
