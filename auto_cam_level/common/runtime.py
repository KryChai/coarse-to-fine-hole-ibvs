"""Run camera-levelling experiments against a synthetic projection model."""

import argparse
import json
from pathlib import Path

import numpy as np

from auto_cam_level import parameters as P
from .control import ControlSettings, RotationServo
from .geometry import CameraModel, finite
from .simulation import SyntheticPlant


def main(scheme, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional synthetic camera and controller parameters")
    parser.add_argument("--seed", type=int, default=P.SIMULATION_SEED)
    parser.add_argument("--tilt", type=float, nargs=2, default=P.SIMULATION_TILT_DEG)
    args = parser.parse_args(argv)
    _, camera, feature, tolerance, settings = load_config(args.config, scheme)
    plant = SyntheticPlant(camera, feature, args.seed, args.tilt)
    servo = RotationServo(plant.measure, lambda: plant.pose.copy(), plant.move,
                          tolerance, settings, clock=lambda: plant.now)
    servo.run()
    print(json.dumps({"scheme": scheme, "source": "simulation", "moves": len(plant.moves),
                      "remaining_tilt_deg": plant.truth_tilt_deg()}))


def load_config(path, scheme):
    if type(P.MEASUREMENT_SAMPLES) is not int or P.MEASUREMENT_SAMPLES < 3:
        raise ValueError("MEASUREMENT_SAMPLES must be an integer >=3")
    data = P.default_config(scheme)
    if path:
        override = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(override, dict) or set(override)-set(data):
            raise ValueError("unknown config sections")
        for group, value in override.items():
            data[group].update(value)
    if not isinstance(data, dict) or set(data)-{"camera", "array", "feature", "control"}:
        raise ValueError("expected camera/array/feature/control config")
    camera_data = data["camera"]
    camera = CameraModel(**{k: camera_data[k] for k in ("K", "D", "image_size")})
    array = data["array"]
    if set(array)-{"count", "radius_mm"}:
        raise ValueError("unknown array configuration")
    options = dict(data.get("feature", {}))
    tolerance = options.pop("tolerance", None)
    if scheme == "pose":
        from auto_cam_level.scheme1_pose.features import PoseFeatures
        feature = PoseFeatures(camera, **array, **options)
        tolerance = tolerance if tolerance is not None else [float(np.deg2rad(.5))]*2
    else:
        from auto_cam_level.scheme2_shape.features import ShapeFeatures
        feature = ShapeFeatures(camera, count=array["count"], **options)
        tolerance = tolerance if tolerance is not None else ([.002, .002, .0008, .0008] if feature.enhanced else [.002]*2)
        if not feature.enhanced and (not path or "tolerance" not in override.get("feature", {})):
            tolerance = tolerance[:2]
    dimensions = 2 if scheme == "pose" or not feature.enhanced else 4
    tolerance = finite(tolerance, (dimensions,))
    if np.any(tolerance <= 0):
        raise ValueError("feature tolerances must be positive")
    cfg = ControlSettings(**data.get("control", {})).validate()
    return data, camera, feature, tolerance, cfg
