"""Lightweight OpenCV preview overlay; never edits the raw camera frame."""
import cv2
import numpy as np


def draw_circle_preview(view, data, image_size):
    try:
        _draw(view, data, image_size)
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError, cv2.error):
        # Malformed diagnostics must never interrupt the camera capture thread.
        return


def _draw(view, data, image_size):
    if data.get("image_size") != dict(zip(("width", "height"), image_size)):
        return
    scale = np.array([view.shape[1]/image_size[0], view.shape[0]/image_size[1]])
    for hole in data.get("holes", []):
        circle = hole.get("circle")
        if not circle:
            continue
        color = (0, 255, 0) if hole.get("status") == "ok" else (0, 165, 255)
        center = np.array([circle["center_x"], circle["center_y"]])
        points = (hole.get("visualization") or {}).get("outline_points_px")
        if points:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                continue
            points = np.rint(points*scale).astype(np.int32)
            cv2.polylines(view, [points], True, color, 1, cv2.LINE_AA)
        point = tuple(np.rint(center*scale).astype(int))
        cv2.drawMarker(view, point, (0, 0, 255), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(view, f"{hole.get('status')} ({center[0]:.2f}, {center[1]:.2f})",
                    point, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
