"""Offline chessboard calibration; emits a camera section for either scheme."""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from auto_cam_level.parameters import CAMERA_CALIBRATION as C, CAMERA


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=C["images"], help="quoted image glob")
    parser.add_argument("--columns", type=int, default=C["columns"], help="inner corners, NOT squares")
    parser.add_argument("--rows", type=int, default=C["rows"])
    parser.add_argument("--square-mm", type=float, default=C["square_mm"])
    parser.add_argument("--camera-id", default=CAMERA["camera_id"])
    parser.add_argument("--output", default=C["output"])
    parser.add_argument("--max-rms-px", type=float, default=C["max_rms_px"])
    args = parser.parse_args(argv)
    if (min(args.columns, args.rows) < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0
            or not args.camera_id.strip() or not np.isfinite(args.max_rms_px) or args.max_rms_px <= 0):
        parser.error("invalid calibration parameters")
    if Path(args.output).exists():
        parser.error("output exists; choose a new file")
    obj = np.zeros((args.columns*args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.columns, 0:args.rows].T.reshape(-1, 2)*args.square_mm
    objects, images, paths, size = [], [], [], None
    for path in sorted(glob.glob(args.images)):
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise ValueError(f"cannot read calibration image: {path}")
        current = (frame.shape[1], frame.shape[0])
        if size is not None and current != size:
            raise ValueError("mixed calibration image dimensions")
        size = current
        ok, corners = cv2.findChessboardCornersSB(frame, (args.columns, args.rows))
        if ok:
            objects.append(obj.copy())
            images.append(corners)
            paths.append(str(Path(path).resolve()))
    if len(images) < 10:
        raise ValueError("need >=10 valid, varied chessboard views covering the image")
    rms, K, D, rv, tv = cv2.calibrateCamera(objects, images, size, None, None)
    per_view = [float(np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, D)[0]-p)**2, axis=2))))
                for o, p, r, t in zip(objects, images, rv, tv)]
    if not np.isfinite(rms) or rms > args.max_rms_px or max(per_view) > 2*args.max_rms_px:
        raise ValueError(f"camera RMS gate failed: global={rms}, views={per_view}")
    camera = dict(source="hardware", camera_id=args.camera_id, K=K.tolist(), D=D.reshape(-1).tolist(),
                  image_size=size, rms_px=rms, per_view_rms_px=per_view, images=paths)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(dict(camera=camera), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Camera RMS={rms:.4f}px; {len(paths)} views; {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
