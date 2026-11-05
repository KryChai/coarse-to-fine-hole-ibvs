"""Offline screening must reject missing edges and insufficient temporal evidence."""
import unittest

import cv2
import numpy as np

from tools.test_hole_boundaries import analyze_box, summarize


class BoundaryScreeningTests(unittest.TestCase):
    def test_two_clear_boundaries_with_offset_centers(self):
        gray = np.full((512, 512), 150, dtype=np.uint8)
        cv2.circle(gray, (256, 256), 90, 65, -1, cv2.LINE_AA)
        cv2.circle(gray, (260, 253), 45, 5, -1, cv2.LINE_AA)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.3)
        result = analyze_box(gray, [156, 156, 356, 356])
        self.assertTrue(result['candidate_pair_pass'], result['reason'])
        np.testing.assert_allclose(result['boundaries']['outer']['center_px'], [256, 256], atol=.5)
        np.testing.assert_allclose(result['boundaries']['inner']['center_px'], [260, 253], atol=.5)

    def test_blank_and_single_boundary_cannot_pass_as_pair(self):
        for single in (False, True):
            gray = np.full((512, 512), 150, dtype=np.uint8)
            if single: cv2.circle(gray, (256, 256), 90, 5, -1, cv2.LINE_AA)
            gray = cv2.GaussianBlur(gray, (0, 0), 1.3)
            with self.subTest(single=single):
                self.assertFalse(analyze_box(gray, [156, 156, 356, 356])['candidate_pair_pass'])

    @staticmethod
    def row(index, passed=True, variant='nominal'):
        return {'source': str(index), 'sha256': str(index), 'variant': variant,
                'candidate_pair_pass': passed,
                'boundaries': {name: {'center_px': [100., 100.]} for name in ('outer', 'inner')} if passed else {}}

    def test_one_photo_or_repeated_copy_is_not_temporal_validation(self):
        self.assertFalse(summarize([self.row(0)], True)['static_evidence_sufficient'])
        rows = [self.row(i) for i in range(100)]
        for row in rows: row['sha256'] = 'same photograph'
        self.assertFalse(summarize(rows, True)['static_evidence_sufficient'])
        self.assertFalse(summarize([self.row(i) for i in range(100)], False)['static_evidence_sufficient'])

    def test_failures_remain_in_rate_and_box_perturbation_counts(self):
        rows = [self.row(i, passed=i >= 10) for i in range(100)]
        rows.extend([self.row(10, True, 'shift'), self.row(11, False, 'shift')])
        summary = summarize(rows, True)
        self.assertEqual(summary['candidate_pair_rate'], .9)
        self.assertFalse(summary['temporal_numeric_screen_pass'])
        self.assertEqual(summary['box_perturbations']['inner']['trials_on_passing_nominal'], 2)
        self.assertEqual(summary['box_perturbations']['inner']['passing_trials'], 1)

    def test_center_jumps_fail_numeric_screen_despite_good_fit_rate(self):
        rows = [self.row(i) for i in range(100)]
        for row in rows[50:]:
            row['boundaries']['outer']['center_px'][0] += 4
        summary = summarize(rows, True)
        self.assertEqual(summary['candidate_pair_rate'], 1)
        self.assertFalse(summary['temporal_numeric_screen_pass'])
        self.assertFalse(summary['approved_for_robot_control'])


if __name__ == '__main__': unittest.main()
