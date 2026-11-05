"""Explicit simulated preview; never connects to camera, YOLO or robot hardware."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from servo_telemetry import RunTelemetry
from servo_control import ServoSettings, ServoStepController


def simulate(output_dir, gain=0.03, label="Demo fixed gain 0.03", failure=False, mode="fixed"):
    """Illustrate a local linear image model, not measured robot performance."""
    seconds = [0.0]
    jacobian = np.diag([-12.0, 10.0])
    controller = ServoStepController(jacobian, ServoSettings(mode=mode, gain=gain, max_gain=max(0.15, gain)))
    run = RunTelemetry(output_dir, label=label, source="simulation",
                       config={"controller": "synthetic local image model", "gain": gain,
                               "servo_mode": mode, "step_control": asdict(controller.settings),
                               "initial_error_px": [90.0, -65.0],
                               "align_tol_px": 2.0, "conf_thresh": 0.5,
                               "max_step_mm": 1.0,
                               "jacobian_px_per_mm": jacobian.tolist()},
                       clock=lambda: seconds[0])
    tcp = np.array([498.73, -816.88, 140.89, 90.0, -88.51, 90.0])
    joints = np.array([107.15, 61.15, 78.67, 218.03, -19.45, -270.03])
    run.set_phase("servo_align")
    error = np.array([90.0, -65.0])
    outcome = "failed"
    for step in range(1, 301):
        seconds[0] += 0.45
        run.record("feedback", tcp=tcp, joints=joints)
        # Deliberate missing detection: preserve the gap rather than invent a value.
        if step == 5:
            controller.detection_lost()
            run.record("detection_failed", iteration=step, message="synthetic occlusion")
            continue
        norm = float(np.linalg.norm(error))
        run.record("vision", iteration=step, u_px=1280 + error[0], v_px=720 + error[1],
                   error_u_px=error[0], error_v_px=error[1], error_norm_px=norm,
                   confidence=0.93)
        if norm < 2.0:
            outcome = "reported_success"
            break
        if failure and step >= 15:
            run.record("visual_failed", message="synthetic non-improving error")
            break
        command = controller.step(error)
        applied = command.selected
        target = tcp.copy()
        target[:2] += applied
        run.record("control", iteration=step, target_tcp=target, **command.telemetry())
        run.record("tcp_target", target_tcp=target)
        seconds[0] += 0.15
        tcp = target
        joints += np.array([0.12, -0.08, 0.1, -0.05, 0.03, 0.02]) * np.linalg.norm(applied)
        if not failure or step < 10:
            error += jacobian @ applied
    if outcome == "reported_success":
        run.set_phase("assembly_offset")
        target = tcp.copy()
        target[:3] += [-20.61, 79.72, 0]
        run.record("tcp_target", target_tcp=target)
        for fraction in np.linspace(0.1, 1.0, 10):
            seconds[0] += 0.3
            # These intermediate positions are explicitly simulated, never real traces.
            run.record("feedback", tcp=tcp + fraction * (target - tcp), joints=joints)
        tcp = target
        run.set_phase("assembly_z")
        target = tcp.copy()
        target[2] = 124.56
        run.record("tcp_target", target_tcp=target)
        seconds[0] += 1.5
        run.record("feedback", tcp=target, joints=joints)
    run.close(outcome, "Synthetic demo only; not a hardware experiment.")
    return run.directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--live", action="store_true", help="Open the generated demo in live-view windows")
    args = parser.parse_args(argv)
    from plot_servo_run import main as plot_main
    directories = [simulate(args.output_dir, label="Demo fixed gain 0.03", mode="fixed"),
                   simulate(args.output_dir, label="Demo adaptive gain 0.03-0.15", mode="adaptive"),
                   simulate(args.output_dir, label="Demo adaptive stalled target", failure=True, mode="adaptive")]
    for directory in directories:
        plot_main([str(directory)])
    plot_main([*(str(path) for path in directories), "--output", str(args.output_dir / "comparison")])
    if args.live:
        plot_main([str(directories[0]), "--live"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
