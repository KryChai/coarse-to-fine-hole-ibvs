"""Append-only run measurements; this module never imports a robot or GUI SDK."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from run_logging import append_run_log_line


SCHEMA_VERSION = 3
POSE_AXES = ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")
TCP_FIELDS = tuple("tcp_" + axis for axis in POSE_AXES)
TARGET_TCP_FIELDS = tuple("target_" + field for field in TCP_FIELDS)
JOINT_FIELDS = tuple(f"joint_j{i}_deg" for i in range(1, 7))
TARGET_JOINT_FIELDS = tuple("target_" + field for field in JOINT_FIELDS)
VISION_FIELDS = ("u_px", "v_px", "error_u_px", "error_v_px", "error_norm_px",
                 "confidence", "requested_dx_mm", "requested_dy_mm", "dx_mm", "dy_mm",
                 "step_limited")
LEGACY_FIELDS = ("sequence", "elapsed_s", "phase", "event", "iteration", "message",
                 *VISION_FIELDS, *TCP_FIELDS, *TARGET_TCP_FIELDS, *JOINT_FIELDS,
                 *TARGET_JOINT_FIELDS)
GAIN_FIELDS = ("nominal_gain", "effective_gain", "gain_scale", "step_limit_mm",
               "requested_step_mm", "selected_step_mm", "backoff_applied")
VERSION2_FIELDS = (*LEGACY_FIELDS, *GAIN_FIELDS)
FIELDS = (*VERSION2_FIELDS, "hole_index", "stage_iteration")


class RunTelemetry:
    """One flushed CSV and metadata file per run, including failed/partial runs.

    Missing measurements stay empty. Plotting failures and disk errors must not
    turn into robot commands or change the result of a motion operation.
    """

    def __init__(self, output_dir=None, label="baseline", source="hardware", config=None,
                 clock=None):
        root = Path(output_dir) if output_dir else Path(__file__).parent / "runs"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=f"{stamp}_{os.getpid()}_", dir=root))
        self.csv_path = self.directory / "samples.csv"
        self.metadata_path = self.directory / "metadata.json"
        self.audit_path = self.directory / "telemetry.log"
        self._file = self.csv_path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._audit_file = self.audit_path.open("x", encoding="utf-8")
        self._audit_failed = False
        self._run_log_failed = False
        self._clock = clock or time.monotonic
        self._start = self._clock()
        self._sequence = 0
        self.hole_index = 0
        self._observation = 0
        self._iterations = {}
        self.phase = "startup"
        self._disabled = False
        self._closed = False
        self.metadata = {
            "schema_version": SCHEMA_VERSION, "run_id": self.directory.name,
            "label": label, "source": source,
            "started_at": datetime.now().astimezone().isoformat(),
            "outcome": "running", "config": config or {},
            "coordinate_frame": "robot base; translation mm, orientation deg",
            "sampling": "event/feedback samples, not a continuous motion trace",
            "wrench": "unavailable: RobotStateData has no force/torque fields",
            "audit_log": "telemetry.log (UTF-8 JSON lines)",
            "run_log_mirror": "[绘图数据] JSON records in the active entrypoint run log",
        }
        self._write_metadata()
        self._audit("start", metadata=self.metadata, columns=list(FIELDS))

    def _audit(self, kind, **values):
        """JSON lines in the per-run audit log, mirrored to the terminal run log."""
        line = json.dumps({"type": kind, "run_id": self.directory.name,
                           "timestamp": datetime.now().astimezone().isoformat(),
                           **values}, ensure_ascii=False, allow_nan=False, default=float)
        if not self._audit_failed:
            try:
                self._audit_file.write(line + "\n")
                self._audit_file.flush()
            except OSError as error:
                self._audit_failed = True
                print(f"[可视化] 审计日志写入失败，CSV 记录继续: {error}")
        if not self._run_log_failed:
            try:
                append_run_log_line("[绘图数据] " + line)
            except OSError as error:
                self._run_log_failed = True
                print(f"[可视化] 运行日志镜像写入失败，独立数据记录继续: {error}")

    def _write_metadata(self):
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2,
                                        allow_nan=False), encoding="utf-8")
        temporary.replace(self.metadata_path)

    def record(self, event, *, tcp=None, joints=None, target_tcp=None,
               target_joints=None, **values):
        if self._closed or self._disabled:
            return
        try:
            row = {"sequence": self._sequence, "elapsed_s": self._clock() - self._start,
                   "phase": self.phase, "event": event}
            unknown = set(values) - set(FIELDS)
            if unknown:
                raise ValueError(f"Unknown telemetry fields: {sorted(unknown)}")
            row.update(values)
            row["hole_index"] = self.hole_index
            if self.hole_index and "iteration" in values:
                local = values["iteration"]
                key = (self.hole_index, self.phase, local)
                if key not in self._iterations:
                    self._observation += 1
                    self._iterations[key] = self._observation
                row["stage_iteration"] = values.get("stage_iteration",local)
                row["iteration"] = self._iterations[key]
            for fields, vector in ((TCP_FIELDS, tcp), (JOINT_FIELDS, joints),
                                   (TARGET_TCP_FIELDS, target_tcp),
                                   (TARGET_JOINT_FIELDS, target_joints)):
                if vector is not None:
                    vector = list(vector)
                    if len(vector) != 6 or not all(math.isfinite(float(v)) for v in vector):
                        raise ValueError("Telemetry pose/joints must contain six finite numbers")
                    row.update(zip(fields, map(float, vector)))
            for key, value in row.items():
                if key not in ("phase", "event", "message") and value is not None:
                    if not math.isfinite(float(value)):
                        raise ValueError(f"Non-finite measurement: {key}")
            self._writer.writerow(row)
            self._file.flush()
            # Exact column names and nulls allow one-to-one CSV/log reconciliation.
            self._audit("sample", row={field: row.get(field) for field in FIELDS})
            self._sequence += 1
        except (OSError, ValueError, TypeError) as error:
            self._disabled = True
            self._audit("recording_error", event=event, message=str(error))
            print(f"[可视化] 数据记录已停用，控制流程继续: {error}")

    def begin_ring(self, hole_count):
        self.metadata["config"]["hole_count"] = hole_count
        try:
            self._write_metadata()
            self._audit("ring_start", hole_count=hole_count)
        except OSError as error:
            print(f"[可视化] 无法保存整圈元信息: {error}")

    def set_phase(self, phase):
        self.phase = phase
        self.record("phase")

    def close(self, outcome="unknown", message=""):
        if self._closed:
            return
        self.record("end", message=message)
        self._closed = True
        self.metadata.update(outcome=outcome, message=message,
                             samples=self._sequence, recording_failed=self._disabled,
                             elapsed_s=self._clock() - self._start,
                             audit_log_failed=self._audit_failed,
                             run_log_mirror_failed=self._run_log_failed)
        self._audit("end", metadata=self.metadata)
        self.metadata.update(audit_log_failed=self._audit_failed,
                             run_log_mirror_failed=self._run_log_failed)
        try:
            self._file.close()
            self._write_metadata()
        except OSError as error:
            print(f"[可视化] 无法完成数据归档，已写入的 CSV 仍保留: {error}")
        finally:
            try:
                self._audit_file.close()
            except OSError as error:
                print(f"[可视化] 关闭审计日志失败: {error}")

    def launch_viewer(self, live=False, language="zh"):
        """Independent process owns its GUI; it receives only a CSV path."""
        plotter = Path(__file__).with_name("plot_servo_run.py")
        args = [sys.executable, str(plotter), str(self.directory)]
        if live:
            args.extend(("--live", "--close-on-finish", "--live-language", language))
        try:
            # A console is unnecessary on Windows; matplotlib still creates its
            # own interactive window. No robot handles enter the plot process.
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            log = (self.directory / ("viewer.log" if live else "export.log")).open(
                "w", encoding="utf-8")
            with log:
                return subprocess.Popen(args, stdout=log, stderr=log,
                                        creationflags=flags)
        except OSError as error:
            print(f"[可视化] 启动绘图失败，原始数据已保留: {error}")
            return None


def run_visualized(callback, config, argv=None, configure_controller=None,
                   add_arguments=None, configure_run=None):
    """CLI wrapper for hardware entrypoints; callback returns 0 on completion."""
    parser = argparse.ArgumentParser(description="执行原控制流程并记录可视化数据")
    parser.add_argument("--no-live-plots", action="store_true", help="仅记录并在结束后出图")
    parser.add_argument("--no-telemetry", action="store_true", help="禁用记录与绘图")
    parser.add_argument("--run-label", default="baseline", help="算法或参数版本标签")
    parser.add_argument("--plot-language", choices=("zh", "en"), default="zh",
                        help="实时窗口语言；离线图片始终同时导出中英文")
    parser.add_argument("--output-dir", type=Path, help="运行数据根目录，默认 runs/")
    parser.add_argument("--servo-mode", choices=("adaptive", "fixed"),
                        help="变增益/步长或固定增益对照；默认为控制器配置")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args(argv)
    if configure_run is not None:
        configure_run(args)
    if args.servo_mode is not None and configure_controller is None:
        parser.error("this entrypoint has no configurable servo controller")
    if configure_controller is not None:
        try:
            config = {**config, **configure_controller(args.servo_mode)}
        except (TypeError, ValueError) as error:
            parser.error(f"invalid servo settings: {error}")
    telemetry = None
    if not args.no_telemetry:
        try:
            telemetry = RunTelemetry(args.output_dir, args.run_label, config=config)
            print(f"[可视化] 运行数据目录: {telemetry.directory}")
            if not args.no_live_plots:
                telemetry.launch_viewer(live=True, language=args.plot_language)
        except (OSError, ValueError) as error:
            print(f"[可视化] 初始化失败，继续原控制流程: {error}")
    outcome, message = "error", ""
    try:
        result = callback(telemetry=telemetry)
        outcome = "reported_success" if result == 0 else "failed"
        return result
    except KeyboardInterrupt:
        outcome, message = "interrupted", "KeyboardInterrupt"
        raise
    except BaseException as error:
        message = f"{type(error).__name__}: {error}"
        raise
    finally:
        if telemetry is not None:
            telemetry.close(outcome, message)
            telemetry.launch_viewer(live=False)
