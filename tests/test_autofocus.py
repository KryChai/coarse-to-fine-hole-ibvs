import unittest
from unittest.mock import Mock, patch

import camera_autofocus as focus

SETTINGS = dict(timeout_s=1., min_wait_s=.2, stable_frames=3, stable_ratio=.05)


class FocusQualityTests(unittest.TestCase):
    def run_focus(self, scores, setter=None):
        clock = [0.]
        values = iter(scores)
        last = [scores[-1]]
        camera = Mock()
        camera.read.return_value = (True, object())
        camera.set.side_effect = setter or (lambda *args: True)
        def score(frame):
            last[0] = next(values, last[0])
            return last[0]
        with patch.object(focus.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(focus.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0]+.05)), \
                patch.object(focus, "sharpness", side_effect=score):
            result = focus.focus_capture(camera, SETTINGS)
        self.assertEqual(camera.set.call_args.args, (focus.cv2.CAP_PROP_AUTOFOCUS, 0))
        return result

    def test_blurred_plateau_is_not_success(self):
        result = self.run_focus([100.,60.,60.,60.])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["peak"],100.)

    def test_sweep_recovers_then_locks(self):
        result = self.run_focus([100.,40.,80.,120.,120.,120.,120.,120.])
        self.assertEqual(result["status"],"stable")
        self.assertEqual(result["score"],120.)

    def test_already_sharp_without_visible_lens_motion_is_usable(self):
        self.assertEqual(self.run_focus([100.]*20)["status"],"stable")

    def test_disabling_af_that_resets_focus_is_detected(self):
        result = self.run_focus([100.]*6+[40.]*20)
        self.assertEqual(result["status"],"failed")

    def test_unsupported_property_is_not_claimed_as_refocus_success(self):
        result = self.run_focus([100.]*20, setter=lambda *args: False)
        self.assertEqual(result["status"],"unsupported")

    def test_enable_ack_without_disable_ack_fails(self):
        result = self.run_focus([100.]*20, setter=lambda prop,value: value==1)
        self.assertEqual(result["status"],"failed")

    def test_capture_failure_and_cancellation_still_request_af_off(self):
        camera=Mock();camera.read.return_value=(False,None)
        for cancelled in (lambda:False,lambda:True):
            result=focus.focus_capture(camera,SETTINGS,cancelled=cancelled)
            self.assertEqual(result["status"],"failed")
            self.assertEqual(camera.set.call_args.args,(focus.cv2.CAP_PROP_AUTOFOCUS,0))
