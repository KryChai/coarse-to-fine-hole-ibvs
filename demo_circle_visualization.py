"""Preview bilingual exports using explicitly simulated data; no hardware."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from circle_center_servo import CircleMeasurementError, CircleServoSettings, run_fine_alignment
from circle_visualization import CircleRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"logs"/"circle_demo")
    args = parser.parse_args()
    run = CircleRun(args.output, source="simulation")
    pose = np.array([100., 200., 150., 90., -88., 90.])
    origin = pose.copy()
    matrix = np.array([[-20., 2.], [1., 18.]])
    target = np.array([360., 240.])
    count = 0
    run("alignment_start", target_uv=target.tolist(), jacobian=matrix.tolist())
    run("coarse_measurement", iteration=1, center=[380., 220.], target_uv=target.tolist(),
        confidence=.95, error_px=float(np.hypot(20, 20)))
    run("coarse_measurement", iteration=2, center=target.tolist(), target_uv=target.tolist(), confidence=.95, error_px=0.)

    def measure(hint):
        nonlocal count
        count += 1
        shift = matrix @ (pose[:2]-origin[:2])
        detection = target+shift
        center = detection+[7., -5.]
        angles = np.linspace(0, 2*np.pi, 181)
        points = center+60*np.c_[np.cos(angles), np.sin(angles)]
        quality = {"valid": count != 3, "flags": [] if count != 3 else ["bbox_sensitive"],
                   "stability_checked": True, "rmse_px": .5, "angular_coverage": .95,
                   "inlier_fraction": .9, "bbox_jitter_max_center_shift_px": .12,
                   "scale_center_shift_px": .1, "candidate_center_spread_px": .2}
        hole = dict(detection_index=0, status="review" if count == 3 else "ok", confidence=.95,
                    center=dict(zip(("x", "y"), center)), detection_center=dict(zip(("x", "y"), detection)),
                    detection_bbox=dict(zip(("x1", "y1", "x2", "y2"), [*(detection-65), *(detection+65)])),
                    circle=dict(center_x=center[0], center_y=center[1], radius=60., diameter=120.),
                    fit_quality=quality, visualization=dict(outline_points_px=points.tolist(), edge_points_px=points[::2].tolist()))
        image = np.full((480, 720, 3), 180, dtype=np.uint8)
        cv2.circle(image, tuple(np.rint(center).astype(int)), 60, (55, 55, 55), -1)
        path = run.directory/f"frame_{count:03d}.png"
        cv2.imwrite(str(path), image)
        run("circle_response", image=str(path.resolve()), expected_detection=hint.tolist(),
            target_uv=target.tolist(), response={"data": {"image_size": {"width": 720, "height": 480}, "holes": [hole]}})
        if count == 3:
            raise CircleMeasurementError("模拟框扰动质量拒绝 / simulated quality rejection")
        return hole

    def move(destination):
        pose[:] = destination

    try:
        result = run_fine_alignment(target, measure=measure, get_pose=lambda: pose.copy(), move=move,
            jacobian=matrix, target_uv=target, settings=CircleServoSettings(gain=.4), record=run)
        run("alignment_complete", tcp=result)
    except BaseException as error:
        run("alignment_failed", reason=str(error))
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    from run_logging import enable_run_logging
    enable_run_logging(__file__)
    main()
