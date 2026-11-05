"""Generate explicitly labelled SIMULATED audit logs and bilingual standalone plots.

This demo never imports the robot SDK or connects to cameras/detection services.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from ibvs_jacobian_estimator import JacobianSettings, save_calibration
from online_jacobian_servo import JacobianRunLog, run_online_servo


def generate_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    J_true = np.array([[-20., 1.5], [2., 18.]])
    origin = np.array([100., 200., 140., 90., -88.5, 90.])
    tcp = origin.copy()
    count = 0
    cfg = JacobianSettings(camera_id="SIMULATED_CAMERA", hole_diameter_mm=11.,
                           calibration_path=str(directory / "simulated_calibration.json"))
    tool = {"name": "SIMULATED_TOOL", "offset": [0.] * 6}
    meta = {"camera_id": cfg.camera_id, "camera_index": 0, "image_size": list(cfg.image_size),
            "target_uv": list(cfg.target_uv), "hole_diameter_mm": 11.,
            "tool": tool, "reference_tcp": origin.tolist()}
    save_calibration(cfg.calibration_path, J_true * .85, meta)
    log = JacobianRunLog(directory, source="simulation")
    log("simulation_model", true_J=J_true.tolist(), initial_J=(J_true * .85).tolist(),
        injected_outlier_iteration=4, injected_outlier_u_px=34.,
        note="Synthetic affine plant with one artificial detector outlier; not experimental data.")

    def measure(hint):
        nonlocal count
        count += 1
        center = np.array([1200., 800.]) + J_true @ (tcp[:2] - origin[:2])
        if count == 4:
            center[0] += 34.  # Explicit synthetic detector outlier exercises rejection logging.
        return {"tcp": tcp.tolist(), "center": center.tolist(), "timestamp": float(count),
                "frame_id": count, "image_size": list(cfg.image_size), "conf": .95,
                "valid": True, "settled": True, "noise_px": .05, "bbox_wh": [220., 220.]}

    def move(target):
        nonlocal tcp
        target = np.asarray(target)
        if np.linalg.norm(target[:2] - tcp[:2]) > cfg.max_step_mm + 1e-9:
            raise ValueError("simulated motion exceeds configured bound")
        tcp = target.copy()

    try:
        run_online_servo(cfg, measure=measure, move=move, tool=tool, log=log, clock=lambda: float(count))
    except BaseException as error:
        log("aborted", message=str(error), exception_type=type(error).__name__)
        raise
    finally:
        manifest = log.export_figures()
    return log.path, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs") / ("jacobian_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    print(generate_demo(args.output))
