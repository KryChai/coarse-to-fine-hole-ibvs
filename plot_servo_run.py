"""Live and offline matplotlib views of recorded IBVS runs. No hardware imports."""

import argparse
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path

from servo_telemetry import FIELDS, LEGACY_FIELDS, VERSION2_FIELDS, SCHEMA_VERSION
from plot_panels import (
    SERVO_PANELS, ROBOT_PANELS, configure_style, draw_servo, draw_robot,
    draw_overview, draw_single, draw_comparison, draw_comparison_panel, header, layout,
    text as plot_text, save_figure,
)


@dataclass
class RunData:
    directory: Path
    metadata: dict
    rows: list

    @property
    def label(self):
        return self.metadata.get("label", self.directory.name)


def load_run(path):
    """Ignore only the unflushed final CSV fragment while a writer is active."""
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    version = metadata.get("schema_version")
    if version not in (1, 2, SCHEMA_VERSION):
        raise ValueError("Unsupported telemetry schema_version")
    text = (directory / "samples.csv").read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    reader = csv.DictReader(io.StringIO(text))
    expected = LEGACY_FIELDS if version == 1 else VERSION2_FIELDS if version == 2 else FIELDS
    if reader.fieldnames != list(expected):
        raise ValueError("CSV columns do not match the telemetry schema")
    rows = []
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("Malformed complete CSV record")
        row = {key: math.nan for key in FIELDS}
        for key, value in raw.items():
            if key in ("event", "phase", "message"):
                row[key] = value
            else:
                row[key] = float(value) if value else math.nan
                if value and not math.isfinite(row[key]):
                    raise ValueError(f"Non-finite CSV measurement: {key}")
        rows.append(row)
    return RunData(directory, metadata, rows)


def summarize(run):
    observations = [r for r in run.rows if r["event"] == "vision"]
    config = run.metadata.get("config", {})
    fine_enabled = config.get("fine_enabled", False)
    tolerance = float(config.get("fine_align_tol_px", 0.5) if fine_enabled
                      else config.get("align_tol_px", 2.0))
    reached = [r for r in observations if r["error_norm_px"] < tolerance
               and (not fine_enabled or r["phase"] == "circle_fine")]
    stable = [r for r in run.rows if r["event"] == "fine_aligned"]
    gains = [r["effective_gain"] for r in run.rows if math.isfinite(r["effective_gain"])]
    result = {
        "label": run.label, "source": run.metadata.get("source", "unknown"),
        "outcome": run.metadata.get("outcome", "unknown"),
        "alignment_reached": bool(stable) if fine_enabled else bool(reached),
        "threshold_px": tolerance,
        "stable_alignment_step": stable[0]["iteration"] if stable else None,
        "first_threshold_step": reached[0]["iteration"] if reached else None,
        "first_threshold_elapsed_s": reached[0]["elapsed_s"] if reached else None,
        "final_observed_error_px": observations[-1]["error_norm_px"] if observations else None,
        "detection_failures": sum(r["event"] == "detection_failed" for r in run.rows),
        "limited_steps": sum(r["event"] == "control" and r["step_limited"] == 1
                             for r in run.rows),
        "recorded_rows": len(run.rows),
        "servo_mode": run.metadata.get("config", {}).get("servo_mode"),
        "effective_gain_min": min(gains) if gains else None,
        "effective_gain_max": max(gains) if gains else None,
        "backoff_steps": sum(r["backoff_applied"] == 1 for r in run.rows),
        "note": "A threshold observation is not a tightening-quality measurement.",
    }
    holes = sorted({int(r["hole_index"]) for r in run.rows if math.isfinite(r["hole_index"]) and r["hole_index"] > 0})
    if holes:
        result["holes"] = []
        for hole in holes:
            subset = [r.copy() for r in run.rows if r["hole_index"] == hole]
            completed = any(r["event"] == "hole_complete" for r in subset)
            for row in subset:
                row["hole_index"] = 0
            detail = summarize(RunData(run.directory,run.metadata,subset))
            detail.update(hole_index=hole,assembly_completed=completed)
            result["holes"].append(detail)
        result["completed_holes"] = sum(h["assembly_completed"] for h in result["holes"])
        result["alignment_reached"] = all(h["alignment_reached"] for h in result["holes"])
        result["ring_completed"] = result["completed_holes"] == config.get("hole_count",12) and run.metadata.get("outcome") == "reported_success"
    return result



def _write_export_info(directory, files, runs, x_axis):
    manifest = {
        "languages": ["en", "zh"], "formats": ["png", "pdf", "svg"], "x_axis": x_axis,
        "runs": [{"directory": str(run.directory.resolve()), "label": run.label,
                  "source": run.metadata.get("source", "unknown"), "rows": len(run.rows),
                  "last_sequence": run.rows[-1]["sequence"] if run.rows else None}
                 for run in runs],
        "files": [str(Path(file).relative_to(directory)).replace("\\", "/") for file in files],
        "legacy_root_copies": "English combined figures, for existing consumers",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                       allow_nan=False), encoding="utf-8")


def _legacy_copies(directory, names):
    import shutil
    copied = []
    for name in names:
        for extension in ("png", "pdf", "svg"):
            target = directory / f"{name}.{extension}"
            shutil.copyfile(directory / "en" / target.name, target)
            copied.append(str(target))
    return copied


def export_run(run, output=None, x_axis="step"):
    import matplotlib.pyplot as plt
    directory = Path(output) if output else run.directory / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for language in ("en", "zh"):
        configure_style(language)
        localized = directory / language
        panels = localized / "panels"
        panels.mkdir(parents=True, exist_ok=True)
        for name, size, draw in (("overview", (14, 21), draw_overview),
                                 ("servo", (11, 10), draw_servo), ("robot", (15, 8), draw_robot)):
            fig = plt.figure(figsize=size)
            try:
                if name == "robot":
                    draw(fig, run, language=language)
                else:
                    draw(fig, run, x_axis, language=language)
                files.extend(save_figure(fig, localized / name))
            finally:
                plt.close(fig)
        for panel in SERVO_PANELS + ROBOT_PANELS:
            fig = plt.figure(figsize=(7, 5))
            try:
                draw_single(fig, run, panel, x_axis, language)
                files.extend(save_figure(fig, panels / panel))
            finally:
                plt.close(fig)
    files.extend(_legacy_copies(directory, ("overview", "servo", "robot")))
    (directory / "summary.json").write_text(
        json.dumps(summarize(run), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_export_info(directory, files, [run], x_axis)
    print(f"[plots] Exported bilingual composites and individual panels: {directory}")
    return directory


def export_comparison(runs, output=None, x_axis="step"):
    import matplotlib.pyplot as plt
    output = Path(output) if output else runs[0].directory / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for language in ("en", "zh"):
        configure_style(language)
        localized = output / language
        (localized / "panels").mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(14, max(4, 3.1 * len(runs))))
        try:
            axes = draw_comparison(fig, runs, x_axis, language)
            limits = [(axes[0, col].get_xlim(), axes[0, col].get_ylim()) for col in range(3)]
            files.extend(save_figure(fig, localized / "comparison"))
        finally:
            plt.close(fig)
        for index, run in enumerate(runs, start=1):
            for column, name in enumerate(("error", "convergence", "correction")):
                fig = plt.figure(figsize=(7, 5))
                try:
                    title = ("Image feature error", "Convergence", "Selected correction")[column]
                    header(fig, run, title, language)
                    ax = fig.add_subplot(111)
                    draw_comparison_panel(ax, run, column, x_axis, language)
                    ax.set_title(plot_text(title, language))
                    ax.set_xlim(limits[column][0])
                    ax.set_ylim(limits[column][1])
                    layout(fig, "SIMULATED DEMO" if run.metadata.get("source") == "simulation" else "", language)
                    files.extend(save_figure(fig, localized / "panels" / f"run_{index:03d}_{name}"))
                finally:
                    plt.close(fig)
    files.extend(_legacy_copies(output, ("comparison",)))
    (output / "summary.json").write_text(json.dumps([summarize(run) for run in runs],
        indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    _write_export_info(output, files, runs, x_axis)
    print(f"[plots] Exported bilingual comparison and individual panels: {output}")
    return output


def watch_run(path, x_axis="step", language="en", close_on_finish=False):
    import matplotlib.pyplot as plt
    configure_style(language)
    # Each independent window owns a GUI timer.
    servo = plt.figure(figsize=(11, 10))
    robot = plt.figure(figsize=(15, 8))
    servo.canvas.manager.set_window_title(plot_text("IBVS - convergence", language))
    robot.canvas.manager.set_window_title(plot_text("IBVS - robot coordinates", language))
    signature = None

    def refresh():
        nonlocal signature
        if not plt.fignum_exists(servo.number) and not plt.fignum_exists(robot.number):
            timer.stop()
            return
        try:
            run = load_run(path)
            if close_on_finish and run.metadata.get("outcome") != "running":
                timer.stop()
                plt.close(servo)
                plt.close(robot)
                return
            current = (len(run.rows), run.metadata.get("outcome"))
            if current == signature:
                return
            signature = current
            if plt.fignum_exists(servo.number):
                draw_servo(servo, run, x_axis, language)
                servo.canvas.draw_idle()
            if plt.fignum_exists(robot.number):
                view = next(((ax.elev, ax.azim) for ax in robot.axes if hasattr(ax, "elev")), None)
                draw_robot(robot, run, view, language)
                robot.canvas.draw_idle()
            if run.metadata.get("outcome") != "running":
                timer.stop()
        except (OSError, ValueError) as error:
            print(f"[plots] Waiting for a complete run file: {error}")

    # Keep a timer on each canvas: closing either window leaves the other live.
    timers = [fig.canvas.new_timer(interval=1000) for fig in (servo, robot)]
    timer = _TimerGroup(timers)
    for item in timers:
        item.add_callback(refresh)
        item.start()
    refresh()
    if plt.fignum_exists(servo.number) or plt.fignum_exists(robot.number):
        plt.show()


class _TimerGroup:
    def __init__(self, timers):
        self.timers = timers

    def stop(self):
        for timer in self.timers:
            timer.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or samples.csv paths")
    parser.add_argument("--live", action="store_true", help="Watch one active run in live windows")
    parser.add_argument("--close-on-finish", action="store_true",
                        help="Close live windows when the recorded task finishes")
    parser.add_argument("--live-language", choices=("zh", "en"), default="zh",
                        help="Live-window language; exports always include both languages")
    parser.add_argument("--x-axis", choices=("step", "time"), default="step")
    parser.add_argument("--output", type=Path, help="Export directory for one run or comparison")
    args = parser.parse_args(argv)
    if args.live and len(args.runs) != 1:
        parser.error("--live accepts exactly one run")
    if args.close_on_finish and not args.live:
        parser.error("--close-on-finish requires --live")
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / "runs" / ".matplotlib"))
        import matplotlib
        if not args.live:
            matplotlib.use("Agg")
        elif matplotlib.get_backend().lower() in ("agg", "pdf", "svg", "ps", "pgf", "cairo", "template"):
            # A previous offline export (e.g. demo --live) may have selected Agg.
            # Tkinter ships with the normal Windows Python installer.
            matplotlib.use("TkAgg")
        configure_style()
        if args.live:
            watch_run(args.runs[0], args.x_axis, args.live_language, args.close_on_finish)
        else:
            runs = [load_run(path) for path in args.runs]
            if len(runs) == 1:
                export_run(runs[0], args.output, args.x_axis)
            else:
                export_comparison(runs, args.output, args.x_axis)
    except ImportError as error:
        parser.exit(2, f"Missing plotting dependency ({error}). Install requirements.txt; "
                       "live mode also needs Python Tkinter/Tk and a desktop session.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Cannot load / plot run: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
