"""No hardware: verify persistence, failure isolation and the live/offline reader."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from servo_telemetry import FIELDS, LEGACY_FIELDS, GAIN_FIELDS, RunTelemetry, run_visualized
from plot_servo_run import load_run, summarize


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def new_run(self, **kwargs):
        run = RunTelemetry(self.root, **kwargs)
        self.addCleanup(run.close)
        return run

    def test_rows_are_flushed_and_missing_values_are_not_zero(self):
        run = self.new_run(label="test", config={"gain": 0.03})
        run.set_phase("servo_align")
        run.record("vision", iteration=1, error_u_px=4, error_v_px=-3,
                   error_norm_px=5, confidence=0.9)
        with run.csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]["error_u_px"], "4")
        self.assertEqual(rows[-1]["tcp_x_mm"], "")
        self.assertEqual(rows[-1]["phase"], "servo_align")
        self.assertEqual(list(rows[-1]), list(FIELDS))
        run.close("failed", "lost target")
        run.close("reported_success")
        self.assertEqual(json.loads(run.metadata_path.read_text(encoding="utf-8"))["outcome"], "failed")

    def test_feedback_keeps_multi_turn_angles_and_targets_separate(self):
        run = self.new_run()
        run.record("feedback", tcp=[1, 2, 3, 4, 5, 6], joints=[0, 1, 2, 3, 4, -270],
                   target_tcp=[2, 3, 4, 5, 6, 7])
        row = load_run(run.directory).rows[0]
        self.assertEqual(row["joint_j6_deg"], -270)
        self.assertEqual(row["tcp_x_mm"], 1)
        self.assertEqual(row["target_tcp_x_mm"], 2)

    def test_every_csv_value_is_preserved_in_the_audit_log(self):
        run = self.new_run(config={"gain": 0.03})
        run.record("feedback", tcp=[1.23456789, 2, 3, 4, 5, 6], joints=[1, 2, 3, 4, 5, -270])
        run.record("detection_failed", iteration=1, message="目标缺失")
        run.close("interrupted", "KeyboardInterrupt")
        records = [json.loads(line) for line in run.audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(records[0]["type"], "start")
        self.assertEqual(records[0]["metadata"]["config"]["gain"], 0.03)
        self.assertTrue(all(record["run_id"] == run.directory.name for record in records))
        with run.csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        samples = [record["row"] for record in records if record["type"] == "sample"]
        self.assertEqual(len(rows), len(samples))
        for row, sample in zip(rows, samples):
            self.assertEqual(list(sample), list(FIELDS))
            for field, value in row.items():
                if value == "":
                    self.assertIsNone(sample[field])
                elif field in ("phase", "event", "message"):
                    self.assertEqual(sample[field], value)
                else:
                    self.assertEqual(float(sample[field]), float(value))
        self.assertEqual(records[-1]["metadata"]["outcome"], "interrupted")

    def test_audit_log_failure_does_not_disable_csv_measurements(self):
        run = self.new_run()
        run._audit_file.close()
        run._audit_file = Mock()
        run._audit_file.write.side_effect = OSError("audit path unavailable")
        with contextlib.redirect_stdout(io.StringIO()):
            run.record("vision", iteration=1, error_norm_px=3.5)
        run.close("failed")
        self.assertFalse(run._disabled)
        self.assertEqual(load_run(run.directory).rows[0]["error_norm_px"], 3.5)
        self.assertTrue(json.loads(run.metadata_path.read_text(encoding="utf-8"))["audit_log_failed"])

    def test_disk_write_failure_isolated_from_control_and_marked(self):
        run = self.new_run()
        run._writer = Mock()
        run._writer.writerow.side_effect = OSError("disk full")
        with contextlib.redirect_stdout(io.StringIO()):
            run.record("vision", error_norm_px=5)
        self.assertTrue(run._disabled)
        run.close("failed")
        self.assertTrue(json.loads(run.metadata_path.read_text(encoding="utf-8"))["recording_failed"])

    def test_nonfinite_measurements_do_not_enter_csv(self):
        run = self.new_run()
        with contextlib.redirect_stdout(io.StringIO()):
            run.record("feedback", tcp=[float("nan")] * 6)
        self.assertEqual(load_run(run.directory).rows, [])

    def test_partial_final_record_during_live_write_is_retried_not_plotted(self):
        run = self.new_run()
        run.record("vision", iteration=1, error_norm_px=5)
        with run.csv_path.open("a", encoding="utf-8") as stream:
            stream.write("2,0.1,servo_align,vision,2")
        self.assertEqual(len(load_run(run.directory).rows), 1)

    def test_malformed_complete_rows_are_rejected(self):
        run = self.new_run()
        with run.csv_path.open("a", encoding="utf-8") as stream:
            stream.write("2,0.1,servo_align,vision,2\n")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            load_run(run.directory)

    def test_summary_distinguishes_alignment_from_failed_overall_task(self):
        run = self.new_run(config={"align_tol_px": 2})
        run.record("detection_failed", iteration=1)
        run.record("vision", iteration=2, error_norm_px=1.5)
        run.record("control", iteration=3, step_limited=1)
        run.close("failed")
        summary = summarize(load_run(run.directory))
        self.assertTrue(summary["alignment_reached"])
        self.assertEqual(summary["outcome"], "failed")
        self.assertEqual(summary["first_threshold_step"], 2)
        self.assertEqual(summary["detection_failures"], 1)
        self.assertEqual(summary["limited_steps"], 1)

    def test_cli_closes_data_and_exports_after_interrupt(self):
        callback = Mock(side_effect=KeyboardInterrupt())
        with patch.object(RunTelemetry, "launch_viewer") as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                run_visualized(callback, {}, ["--output-dir", str(self.root)])
        run = callback.call_args.kwargs["telemetry"]
        self.assertEqual(json.loads(run.metadata_path.read_text(encoding="utf-8"))["outcome"], "interrupted")
        self.assertEqual([call.kwargs["live"] for call in launch.call_args_list], [True, False])

    def test_no_live_flag_still_records_and_exports_failed_run(self):
        callback = Mock(return_value=-1)
        with patch.object(RunTelemetry, "launch_viewer") as launch, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_visualized(callback, {}, ["--no-live-plots", "--output-dir",
                                                           str(self.root)]), -1)
        launch.assert_called_once_with(live=False)
        run = callback.call_args.kwargs["telemetry"]
        self.assertEqual(json.loads(run.metadata_path.read_text(encoding="utf-8"))["outcome"], "failed")

    def test_disable_flag_has_no_recording_or_process_side_effects(self):
        callback = Mock(return_value=0)
        with patch("servo_telemetry.RunTelemetry") as constructor:
            self.assertEqual(run_visualized(callback, {}, ["--no-telemetry"]), 0)
        constructor.assert_not_called()
        callback.assert_called_once_with(telemetry=None)

    def test_cli_applies_mode_before_callback_and_records_selected_settings(self):
        callback = Mock(return_value=0)
        configure = Mock(return_value={"servo_mode": "fixed", "gain": 0.03})
        with patch.object(RunTelemetry, "launch_viewer"), contextlib.redirect_stdout(io.StringIO()):
            run_visualized(callback, {}, ["--servo-mode", "fixed", "--no-live-plots",
                                         "--output-dir", str(self.root)], configure_controller=configure)
        configure.assert_called_once_with("fixed")
        run = callback.call_args.kwargs["telemetry"]
        self.assertEqual(json.loads(run.metadata_path.read_text(encoding="utf-8"))["config"]["servo_mode"], "fixed")

    def test_invalid_controller_settings_abort_before_hardware_callback(self):
        callback = Mock()
        configure = Mock(side_effect=ValueError("invalid gain"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_visualized(callback, {}, ["--servo-mode", "adaptive"], configure_controller=configure)
        callback.assert_not_called()

    def test_schema_v1_csv_is_read_without_inventing_gain_measurements(self):
        import math
        run = self.new_run()
        run.close("reported_success")
        metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = 1
        run.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with run.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEGACY_FIELDS)
            writer.writeheader()
            writer.writerow({"sequence": 0, "elapsed_s": 1, "event": "control", "phase": "servo_align",
                             "iteration": 1, "dx_mm": 0.3, "dy_mm": 0.2})
        loaded = load_run(run.directory)
        self.assertEqual(loaded.rows[0]["dx_mm"], 0.3)
        self.assertTrue(all(math.isnan(loaded.rows[0][field]) for field in GAIN_FIELDS))
        self.assertIsNone(summarize(loaded)["effective_gain_max"])

    def test_viewer_receives_only_file_path_and_uses_independent_process(self):
        run = self.new_run()
        with patch("servo_telemetry.subprocess.Popen") as popen:
            run.launch_viewer(live=True)
        args = popen.call_args.args[0]
        self.assertIn(str(run.directory), args)
        self.assertIn("--live", args)
        self.assertIn("--live-language", args)
        self.assertEqual(args[-1], "zh")
        self.assertNotIn("robot_handle", str(popen.call_args))


class PlotRenderingTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from plot_servo_run import configure_style
        configure_style()
        self.plt = plt
        self.addCleanup(plt.close, "all")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = RunTelemetry(self.root, source="simulation")
        self.addCleanup(self.run.close)

    def test_missing_detection_remains_a_gap_and_all_steps_are_retained(self):
        from plot_servo_run import draw_servo
        self.run.record("vision", iteration=1, error_u_px=4, error_v_px=-3, error_norm_px=5)
        self.run.record("detection_failed", iteration=2)
        self.run.record("vision", iteration=3, error_u_px=1, error_v_px=0, error_norm_px=1)
        fig = self.plt.figure()
        draw_servo(fig, load_run(self.run.directory))
        line = fig.axes[0].lines[0]
        self.assertEqual(list(line.get_xdata()), [1, 2, 3])
        import math
        self.assertTrue(math.isnan(line.get_ydata()[1]))

    def test_empty_and_failed_runs_export_nonempty_vector_and_raster_files(self):
        from plot_servo_run import export_run
        from plot_panels import SERVO_PANELS, ROBOT_PANELS
        self.run.close("failed", "camera unavailable")
        with contextlib.redirect_stdout(io.StringIO()):
            output = export_run(load_run(self.run.directory))
        for name in ("servo", "robot"):
            for extension in ("png", "pdf", "svg"):
                self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
        for language in ("en", "zh"):
            for name in ("overview", "servo", "robot"):
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater((output / language / f"{name}.{extension}").stat().st_size, 1000)
            for panel in SERVO_PANELS + ROBOT_PANELS:
                for extension in ("png", "pdf", "svg"):
                    self.assertGreater((output / language / "panels" / f"{panel}.{extension}").stat().st_size, 1000)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 99)
        self.assertTrue(all((output / path).is_file() for path in manifest["files"]))
        self.assertFalse(json.loads((output / "summary.json").read_text(encoding="utf-8"))["alignment_reached"])

    def test_bilingual_individual_panels_use_the_same_observations_as_composites(self):
        from plot_panels import configure_style, draw_overview, draw_single, SERVO_PANELS, ROBOT_PANELS
        import numpy as np
        self.run.record("vision", iteration=1, error_u_px=4, error_v_px=-3, error_norm_px=5, confidence=0.9)
        self.run.record("detection_failed", iteration=2)
        self.run.record("vision", iteration=3, error_u_px=1, error_v_px=0, error_norm_px=1, confidence=0.9)
        self.run.record("feedback", tcp=[10, 20, 30, 0, 1, 2], joints=[1, 2, 3, 4, 5, -270])
        data = load_run(self.run.directory)
        for language in ("en", "zh"):
            configure_style(language)
            combined = self.plt.figure(figsize=(14, 21))
            draw_overview(combined, data, language=language)
            self.assertEqual(len(combined.axes), 12)
            if language == "zh":
                self.assertIn("图像特征误差", combined.axes[0].get_title())
                self.assertIn("检测", combined.axes[0].get_xlabel())
            else:
                self.assertIn("Image feature error", combined.axes[0].get_title())
            for index, panel in enumerate(SERVO_PANELS + ROBOT_PANELS):
                single = self.plt.figure()
                draw_single(single, data, panel, language=language)
                for actual, expected in zip(single.axes[0].lines, combined.axes[index].lines):
                    np.testing.assert_allclose(actual.get_xdata(), expected.get_xdata())
                    np.testing.assert_allclose(actual.get_ydata(), expected.get_ydata())
                self.plt.close(single)
            self.plt.close(combined)

    def test_english_figure_preserves_chinese_run_identifiers_without_missing_glyphs(self):
        import warnings
        from plot_panels import configure_style, draw_single
        configure_style("en")
        figure = self.plt.figure(figsize=(7, 5))
        data = load_run(self.run.directory)
        data.metadata["label"] = "双语窗口检查"
        draw_single(figure, data, "servo_error", language="en")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Glyph .* missing from font")
            figure.canvas.draw()

    def test_comparison_exports_both_languages_and_each_run_panel(self):
        from plot_servo_run import export_comparison
        self.run.record("vision", iteration=1, error_u_px=30, error_v_px=-40, error_norm_px=50)
        other = RunTelemetry(self.root, label="alternative", source="simulation")
        self.addCleanup(other.close)
        other.record("vision", iteration=1, error_u_px=0.6, error_v_px=-0.8, error_norm_px=1)
        files = []
        # Check the export plan and renderer without repeating dozens of raster renders.
        def capture(fig, prefix):
            paths = [str(prefix) + "." + extension for extension in ("png", "pdf", "svg")]
            files.append((str(prefix), [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]))
            for path in paths:
                Path(path).write_text("rendered", encoding="utf-8")
            return paths
        with patch("plot_servo_run.save_figure", side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
            output = export_comparison([load_run(self.run.directory), load_run(other.directory)], self.root / "comparison")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 45)
        for language in ("en", "zh"):
            combined = next(limits for path, limits in files if path.endswith(language + "\\comparison")
                            or path.endswith(language + "/comparison"))
            for index in (1, 2):
                for column, name in enumerate(("error", "convergence", "correction")):
                    single = next(limits for path, limits in files
                                  if (f"{language}\\panels\\run_{index:03d}_{name}" in path
                                      or f"{language}/panels/run_{index:03d}_{name}" in path))
                    self.assertEqual(single[0], combined[column])

    def test_live_refresh_and_closing_one_window_keeps_the_other_updating(self):
        from matplotlib.backend_bases import FigureCanvasBase
        from plot_servo_run import watch_run
        timers = []

        def make_timer(*args, **kwargs):
            timer = Mock()
            timer.add_callback.side_effect = lambda callback: setattr(timer, "callback", callback)
            timers.append(timer)
            return timer

        def interact():
            self.run.record("feedback", tcp=[1, 2, 3, 4, 5, 6], joints=[0] * 6)
            self.run.record("control", iteration=1, nominal_gain=0.15, effective_gain=0.12,
                            step_limit_mm=0.8, requested_step_mm=1, selected_step_mm=0.8)
            timers[0].callback()
            servo, robot = [self.plt.figure(number) for number in self.plt.get_fignums()]
            self.assertEqual(servo.axes[4].lines[1].get_ydata()[-1], 0.12)
            self.plt.close(servo)
            self.run.record("feedback", tcp=[7, 8, 9, 4, 5, 6], joints=[1] * 6)
            timers[1].callback()
            self.assertEqual(list(robot.axes[0].lines[0].get_ydata()), [1, 7])
            self.run.close("failed")
            timers[1].callback()
            self.assertTrue(all(timer.stop.called for timer in timers))

        with patch.object(FigureCanvasBase, "new_timer", side_effect=make_timer), \
                patch.object(self.plt, "show", side_effect=interact):
            watch_run(self.run.directory)


if __name__ == "__main__":
    unittest.main()
