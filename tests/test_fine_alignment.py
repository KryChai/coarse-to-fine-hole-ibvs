"""Fine-loop/controller and protocol tests without any hardware SDK."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import requests

from fine_alignment import ALGORITHM, REFERENCE, CENTER_DEFINITION, BOUNDARY_METHOD, BOUNDARY_REVISION, FineSettings, RefinementClient, RefinementError, run_fine_alignment


def response(data, status=200):
    res = requests.Response()
    res.status_code = status
    res._content = json.dumps(data).encode()
    return res


def payload(center=(100, 100), coarse=(101, 100), valid=True):
    points = [[90,90],[110,90],[110,110],[90,110],[90,90]]
    return {"code":200,"task":"usb_hole11_refine_center","data":{"algorithm":ALGORITHM,"boundary_method":BOUNDARY_METHOD,"boundary_revision":BOUNDARY_REVISION,"calibration_id":None,"calibration_required":False,"reference":REFERENCE,"center_definition":CENTER_DEFINITION,
        "image_size":[200,200],"coordinate_frame":"original_image_pixels",
        "target_hole":{"conf":.9,"x1":80,"y1":80,"x2":120,"y2":120,
            "detection_center":dict(zip(("center_x","center_y"),coarse)),
            "refined_center":{"center_x":center[0],"center_y":center[1],"method":ALGORITHM,"reference":REFERENCE,"center_definition":CENTER_DEFINITION},
            "fit_quality":{"valid":valid,"residual_rms_px":.3,"angular_coverage_deg":320,"caliper_support_fraction":.8},
            "geometry":{"outlines_px":[points],"edge_points_px":[points]}}}}


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.client = RefinementClient('http://localhost:8001/api/usb_hole11_circle_fit',(200,200))
        self.client.preflight_done = True
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'image.jpg'
        self.path.write_bytes(b'original image')

    def test_upload_uses_target_gate_and_preserves_raw_response(self):
        with patch('fine_alignment.requests.post',return_value=response(payload())) as post:
            result=self.client.measure(self.path,(101,100))
        np.testing.assert_array_equal(result['center'],[100,100])
        self.assertEqual(post.call_args.kwargs['params']['max_distance'],20)
        self.assertIn('usb_hole11_refine_center',post.call_args.args[0])
        self.assertIsNotNone(self.client.last_data)

    def test_wrong_calibration_or_coordinates_never_reach_control(self):
        for key,value in (('boundary_revision',1),('boundary_method','old_radial'),('calibration_id','changed'),('reference','outer_front'),('coordinate_frame','roi_pixels'),('image_size',[100,100])):
            data=payload(); data['data'][key]=value
            with self.subTest(key=key),patch('fine_alignment.requests.post',return_value=response(data)):
                with self.assertRaises(RefinementError): self.client.measure(self.path,(101,100))

    def test_invalid_fit_is_missing_not_box_center_fallback(self):
        with patch('fine_alignment.requests.post',return_value=response(payload(valid=False))):
            self.assertIsNone(self.client.measure(self.path,(101,100)))

    def test_low_or_nonfinite_caliper_support_is_rejected(self):
        for support in (.64,1.01,float('nan')):
            data=payload();data['data']['target_hole']['fit_quality']['caliper_support_fraction']=support
            with self.subTest(support=support),patch('fine_alignment.requests.post',return_value=response(data)):
                with self.assertRaises(RefinementError): self.client.measure(self.path,(101,100))

    def test_wrong_target_or_nan_is_protocol_failure(self):
        for center,coarse in (((float('nan'),100),(101,100)),((100,100),(160,100))):
            with patch('fine_alignment.requests.post',return_value=response(payload(center,coarse))):
                with self.assertRaises(RefinementError): self.client.measure(self.path,(101,100))

    def test_preflight_requires_outer_rim_contract(self):
        with patch('fine_alignment.requests.get',return_value=response({'ready':True,'image_size':[200,200]})):
            with self.assertRaises(RefinementError): self.client.preflight()

    def test_outer_rim_preflight_and_upload_work_without_calibration(self):
        data={'code':200,'ready':True,'algorithm':ALGORITHM,'reference':REFERENCE,
              'center_definition':CENTER_DEFINITION,'boundary_method':BOUNDARY_METHOD,
              'boundary_revision':BOUNDARY_REVISION,'calibration_required':False,
              'calibration_id':None,'image_size':None,'image_size_policy':'from_upload'}
        self.client.preflight_done=False
        with patch('fine_alignment.requests.get',return_value=response(data)) as get, \
                patch('fine_alignment.requests.post',return_value=response(payload())) as post:
            self.assertIsNotNone(self.client.measure(self.path,(101,100)))
            self.assertIsNotNone(self.client.measure(self.path,(101,100)))
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs['params']['mode'],'outer_rim')
        self.assertEqual(post.call_args.kwargs['params']['mode'],'outer_rim')


class FineLoopTests(unittest.TestCase):
    def run_loop(self, measure, settings=None, move=None):
        self.record=Mock();self.phase=Mock();self.move=move or Mock()
        return run_fine_alignment(capture=lambda:'frame.jpg',measure=measure,move=self.move,
            get_tcp=lambda:[0,0,10,0,0,0],record=self.record,set_phase=self.phase,
            jacobian=np.eye(2)*10,initial_target=(100,100),target_pixel=(100,100),
            settings=settings,iteration_offset=7)

    @staticmethod
    def sample(point):
        return {'center':np.array(point,dtype=float),'coarse':np.array(point)+[2,0],
                'confidence':.9,'rms':.2,'coverage':320,'data':{'synthetic':True}}

    def test_biased_coarse_center_is_corrected_below_legacy_two_pixel_deadband(self):
        error=np.array([1.6,0.0])
        moves=[]
        def measure(path,expected): return self.sample(np.array([100,100])+error)
        def move(tcp):
            moves.append(tcp)
            error[:] += 10*np.array(tcp[:2])
        self.run_loop(measure,move=move)
        self.assertTrue(moves)
        self.assertLess(np.linalg.norm(error),1)
        self.assertTrue(all(np.linalg.norm(t[:2])<=.15+1e-12 for t in moves))
        self.assertTrue(any(c.args[0]=='fine_aligned' for c in self.record.call_args_list))
        self.assertGreater(self.record.call_args_list[0].kwargs['iteration'],7)

    def test_three_stationary_frames_required_before_success(self):
        measure=Mock(return_value=self.sample([100.1,100.1]))
        self.run_loop(measure)
        self.assertEqual(measure.call_count,3)
        self.move.assert_not_called()

    def test_fit_loss_aborts_without_any_motion(self):
        with self.assertRaisesRegex(RuntimeError,'连续失败'):
            self.run_loop(lambda *args:None)
        self.move.assert_not_called()

    def test_unstable_in_tolerance_center_is_not_success(self):
        values=iter([[100.9,100],[99.1,100]]*5)
        with self.assertRaisesRegex(RuntimeError,'未稳定'):
            self.run_loop(lambda *args:self.sample(next(values)),replace(FineSettings(),max_iterations=6))
        self.move.assert_not_called()

    def test_non_improving_error_stops(self):
        with self.assertRaisesRegex(RuntimeError,'未改善'):
            self.run_loop(lambda *args:self.sample([104,100]))

    def test_no_measurement_history_is_shared_with_coarse_controller(self):
        settings=FineSettings()
        self.assertEqual(settings.controller(np.eye(2)).settings.align_tol_px,1)
        with self.assertRaises(ValueError): replace(settings,stable_frames=1).controller(np.eye(2))

    def test_plots_keep_coarse_and_fine_features_separate_and_show_fine_threshold(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from servo_telemetry import RunTelemetry
        from plot_servo_run import load_run,summarize
        from plot_panels import draw_panel
        with tempfile.TemporaryDirectory() as directory:
            recorder=RunTelemetry(directory,config={'fine_control':{'tolerance_px':1}})
            recorder.set_phase('servo_align')
            recorder.record('vision',iteration=1,error_norm_px=0)
            recorder.set_phase('servo_fine')
            recorder.record('vision',iteration=2,error_norm_px=3)
            recorder.record('vision',iteration=3,error_norm_px=.5)
            recorder.record('fine_aligned',iteration=3)
            recorder.close('reported_success')
            run=load_run(recorder.directory)
            fig,ax=plt.subplots()
            try:
                draw_panel(ax,'servo_convergence',run)
                self.assertTrue(np.isnan(ax.lines[0].get_ydata()[1]))
                self.assertIn('Fine threshold 1 px',[line.get_label() for line in ax.lines])
                self.assertTrue(summarize(run)['fine_alignment_confirmed'])
            finally: plt.close(fig)


if __name__=='__main__': unittest.main()
