import unittest

from coarse_hole_tracking import CoarseHoleTracker, TrackingProtocolError, predict_target_uv


def detection(hole_id, x, confidence=.9, missing=False):
    return dict(hole_id=hole_id, center_x=x, center_y=240, conf=confidence, missing=missing)


class CoarseHoleTrackingTests(unittest.TestCase):
    def test_locks_nearest_id_through_neighbor_dropout_and_outlier(self):
        tracker = CoarseHoleTracker((320, 240))
        target = detection(2, 310, .7)
        neighbor = detection(1, 290)
        self.assertEqual(tracker.select([neighbor, target]), (310, 240, .7))
        self.assertEqual(tracker.hole_id, 2)
        self.assertEqual(tracker.select([detection(1, 320), detection(2, 315)]), (315, 240, .9))
        prediction = predict_target_uv((315, 240), ((-10, 0), (0, 10)), (-.4, 0))
        self.assertEqual(tuple(prediction), (319, 240))
        self.assertIsNone(tracker.select([neighbor, detection(2, 315, missing=True)], prediction))
        self.assertIsNone(tracker.select([neighbor], prediction))
        self.assertIsNone(tracker.select([], prediction))
        self.assertIsNone(tracker.select([detection(2, 400)], prediction))
        self.assertEqual(tracker.select([neighbor, detection(2, 319)], prediction), (319, 240, .9))

    def test_invalid_protocol_does_not_establish_a_track(self):
        tracker = CoarseHoleTracker((320, 240))
        with self.assertRaises(TrackingProtocolError):
            tracker.select([detection(1, 320), detection(1, 325)])
        with self.assertRaises(TrackingProtocolError):
            tracker.select([detection(1, float("nan"))])
        self.assertIsNone(tracker.hole_id)
