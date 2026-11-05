"""Circle API protocol and closed loop tests; no robot/camera/model weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import requests

from circle_center_servo import (
    CircleMeasurementError, CircleProtocolError, CircleRefinementClient,
    CircleServoSettings, refinement_url, run_fine_alignment, select_circle,
)


def hole(center=(1280., 720.), detection=(1280., 720.), status="ok"):
    return {"detection_index": 0, "confidence": .95,
            "detection_center": dict(zip(("x", "y"), detection)),
            "detection_bbox": dict(zip(("x1", "y1", "x2", "y2"),
                                       [detection[0]-100, detection[1]-100,
                                        detection[0]+100, detection[1]+100])),
            "status": status, "center": dict(zip(("x", "y"), center)),
            "circle": {"center_x": center[0], "center_y": center[1], "radius": 90., "diameter": 180.},
            "fit_quality": {"valid": status == "ok", "stability_checked": True,
                            "flags": [] if status == "ok" else ["ambiguous_concentric_edge"]}}


def payload(*holes):
    return {"code": 200, "data": {"algorithm": "polar_dp_soft_l1_circle_v1",
            "center_definition": "fitted_outer_circle_center", "coordinate_frame": "original_image_pixels",
            "units": "px", "image_size": {"width": 2560, "height": 1440},
            "check_stability": True, "bbox_expansion_fraction": .10, "holes": list(holes)}}


class CircleClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = CircleServoSettings()

    def select(self, response, expected=(1280., 720.)):
        return select_circle(response, (2560, 1440), expected, self.settings, .5)

    def test_routes_select_same_hole_model(self):
        for model in (7, 11):
            self.assertEqual(refinement_url(f"http://example.test:8001/api/usb_hole{model}_detect"),
                             f"http://example.test:8001/api/usb_hole{model}_circle_refine")
        with self.assertRaises(ValueError):
            refinement_url("http://example.test/api/usb_hole11_circle_fit")

    def test_match_detection_center_instead_of_fit_or_highest_confidence(self):
        selected = hole(center=(1286., 715.))
        selected["confidence"] = .8
        other = hole(detection=(1350., 720.))
        self.assertIs(self.select(payload(other, selected)), selected)

    def test_bad_selected_target_does_not_fall_back_to_good_neighbour(self):
        for status in ("review", "failed"):
            selected = hole(status=status)
            selected["center"] = None
            with self.subTest(status=status), self.assertRaises(CircleMeasurementError):
                self.select(payload(hole(detection=(1290., 720.)), selected))

    def test_ambiguous_missing_and_far_targets_are_rejected(self):
        for response in (payload(), payload(hole(detection=(1400., 720.))),
                         payload(hole(detection=(1279., 720.)), hole(detection=(1281., 720.)))):
            with self.assertRaises(CircleMeasurementError):
                self.select(response)

    def test_contract_and_quality_mismatches_never_supply_control_point(self):
        cases = []
        for key, value in (("algorithm", "old_ellipse"), ("coordinate_frame", "roi_pixels"),
                           ("center_definition", "bbox_center"), ("check_stability", False),
                           ("image_size", {"width": 1280, "height": 720}),
                           ("bbox_expansion_fraction", .05)):
            response = payload(hole())
            response["data"][key] = value
            cases.append(response)
        bad = payload(hole())
        bad["data"]["holes"][0]["circle"]["radius"] = float("nan")
        cases.append(bad)
        for response in cases:
            with self.subTest(response=response), self.assertRaises(CircleProtocolError):
                self.select(response)
        for key, value in (("valid", False), ("stability_checked", False), ("flags", ["bbox_sensitive"])):
            response = payload(hole())
            response["data"]["holes"][0]["fit_quality"][key] = value
            with self.assertRaises(CircleMeasurementError):
                self.select(response)

    @staticmethod
    def response(code=200, body=None):
        response = requests.Response()
        response.status_code = code
        response._content = json.dumps(body if body is not None else payload(hole())).encode()
        return response

    def test_upload_keeps_original_and_saves_raw_response_and_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"frame.png"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), np.uint8))
            original = path.read_bytes()
            client = CircleRefinementClient("http://example.test/api/usb_hole11_circle_refine", (2560, 1440))
            def post(url, **kwargs):
                self.assertEqual(kwargs["params"], {"conf": .5, "check_stability": "true"})
                name, stream, mime = kwargs["files"]["file"]
                self.assertEqual((name, mime), ("frame.png", "image/png"))
                self.assertEqual(stream.read(), original)
                self.assertEqual(kwargs["timeout"], 60.)
                return self.response()
            with patch("circle_center_servo.requests.post", side_effect=post):
                self.assertEqual(client.measure(path, (1280, 720))["status"], "ok")
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(path.with_suffix(".circle.json").is_file())
            self.assertTrue((path.parent/"frame_circle.jpg").is_file())

    def test_protocol_errors_stop_but_network_and_review_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"frame.jpg"
            cv2.imwrite(str(path), np.zeros((1440, 2560, 3), np.uint8))
            client = CircleRefinementClient("http://example.test/api/usb_hole11_circle_refine", (2560, 1440))
            for code, error in ((404, CircleProtocolError), (422, CircleProtocolError),
                                (500, CircleMeasurementError), (429, CircleMeasurementError)):
                with patch("circle_center_servo.requests.post", return_value=self.response(code)), \
                        self.assertRaises(error):
                    client.measure(path, (1280, 720))
            with patch("circle_center_servo.requests.post", side_effect=requests.Timeout), \
                    self.assertRaises(CircleMeasurementError):
                client.measure(path, (1280, 720))
            with patch("circle_center_servo.requests.post", return_value=self.response(body=payload(hole(status="review")))), \
                    self.assertRaises(CircleMeasurementError):
                client.measure(path, (1280, 720))
            self.assertEqual(json.loads(path.with_suffix(".circle.json").read_text(encoding="utf8"))
                             ["response"]["data"]["holes"][0]["status"], "review")


class FineLoopTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.array([100., 200., 150., 90., -88., 90.])
        self.moves = []
        self.records = []
        self.matrix = np.array([[-20., 2.], [1., 18.]])

    def move(self, target):
        target = np.asarray(target)
        self.assertLessEqual(np.linalg.norm(target[:2]-self.pose[:2]), .150001)
        np.testing.assert_array_equal(target[2:], self.pose[2:])
        self.moves.append(target.copy())
        self.pose[:] = target

    def run_loop(self, measure, **kwargs):
        return run_fine_alignment((1280., 720.), measure=measure, get_pose=lambda: self.pose.copy(),
            move=kwargs.pop("move", self.move), jacobian=kwargs.pop("jacobian", self.matrix),
            target_uv=(1280., 720.), record=lambda event, **data: self.records.append((event, data)), **kwargs)

    def test_coupled_plant_converges_and_uses_fitted_center(self):
        origin = self.pose.copy()
        offset = np.array([7., -5.])
        def measure(hint):
            shift = self.matrix @ (self.pose[:2]-origin[:2])
            detection = np.array([1280., 720.])+shift
            np.testing.assert_allclose(hint, detection)
            return hole(center=detection+offset, detection=detection)
        result = self.run_loop(measure)
        self.assertGreater(len(self.moves), 1)
        error = offset+self.matrix @ (np.array(result[:2])-origin[:2])
        self.assertLess(np.linalg.norm(error), 1.)
        events = [event for event, _ in self.records]
        self.assertEqual(events[-1], "fine_aligned")
        self.assertEqual(events[-4:-1], ["fine_measurement"]*3)

    def test_subthreshold_requires_stationary_consecutive_stable_frames(self):
        measure = Mock(side_effect=[hole(center=(1280.8, 720.)), hole(center=(1279.2, 720.)),
                                   hole(), hole(), hole()])
        self.run_loop(measure)
        self.assertEqual(measure.call_count, 5)
        self.assertEqual(self.moves, [])

    def test_quality_failure_resets_stable_window_and_never_moves(self):
        measure = Mock(side_effect=[hole(), hole(), CircleMeasurementError("review"), hole(), hole(), hole()])
        self.run_loop(measure)
        self.assertEqual(measure.call_count, 6)
        self.assertEqual(self.moves, [])

    def test_repeated_failure_protocol_error_and_motion_error_abort(self):
        for exception in (CircleMeasurementError("review"), CircleProtocolError("old API")):
            measure = Mock(side_effect=exception)
            with self.assertRaises(RuntimeError):
                self.run_loop(measure)
            self.assertEqual(measure.call_count, 3 if isinstance(exception, CircleMeasurementError) else 1)
            self.assertEqual(self.moves, [])
        move = Mock(side_effect=RuntimeError("robot failure"))
        with self.assertRaisesRegex(RuntimeError, "robot failure"):
            self.run_loop(lambda hint: hole(center=(1290., 720.)), move=move)
        move.assert_called_once()

    def test_no_actual_motion_does_not_assume_command_was_executed(self):
        with self.assertRaisesRegex(RuntimeError, "未到达"):
            self.run_loop(lambda hint: hole(center=(1290., 720.)), move=lambda target: None,
                          settings=CircleServoSettings(gain=.5))

    def test_non_improvement_and_unstable_frames_have_bounded_exit(self):
        with self.assertRaisesRegex(RuntimeError, "未改善"):
            self.run_loop(lambda hint: hole(center=(1290., 720.)))
        self.moves.clear()
        count = 0
        def oscillating(hint):
            nonlocal count
            count += 1
            return hole(center=(1280+(.9 if count % 2 else -.9), 720.))
        with self.assertRaisesRegex(RuntimeError, "最大轮数"):
            self.run_loop(oscillating, settings=CircleServoSettings(max_iterations=8))
        self.assertEqual(count, 8)
        self.assertEqual(self.moves, [])

    def test_invalid_jacobian_fails_before_measurement(self):
        measure = Mock()
        with self.assertRaises(ValueError):
            self.run_loop(measure, jacobian=np.zeros((2, 2)))
        measure.assert_not_called()


if __name__ == "__main__":
    unittest.main()
