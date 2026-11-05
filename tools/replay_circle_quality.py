"""Replay saved image/response pairs through the client; no HTTP or robot SDK."""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circle_center_servo import (CircleMeasurementError, CircleServoSettings,
                                 run_fine_alignment, select_circle)


class FirstCommand(Exception):
    pass


def replay(run_directory, images):
    config = json.loads((Path(run_directory) / 'metadata.json').read_text(encoding='utf-8'))['config']
    allowed = {field.name for field in fields(CircleServoSettings)}
    base = CircleServoSettings(**{key: value for key, value in config['fine_settings'].items()
                                  if key in allowed})
    settings = replace(base, allow_smoothing_sensitive=True, smoothing_max_shift_px=2.0)
    strict = replace(settings, allow_smoothing_sensitive=False)
    frames = []
    for image_path in map(Path, images):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot read {image_path}')
        saved = json.loads(image_path.with_suffix('.circle.json').read_text(encoding='utf-8'))
        frames.append((image_path, (image.shape[1], image.shape[0]), saved))
    checks = []
    for path, size, saved in frames:
        item = {'image': str(path.resolve())}
        for label, candidate in [('strict', strict), ('fallback', settings)]:
            try:
                selected = select_circle(saved['response'], size, saved['expected_detection'], candidate, .5)
                item[label] = {'eligible': True, 'server_status': selected['status'],
                               'server_flags': selected['fit_quality']['flags']}
            except CircleMeasurementError as error:
                item[label] = {'eligible': False, 'reason': str(error)}
        checks.append(item)
    # Synthetic constant TCP is deliberate: this checks policy and command
    # timing only. It cannot establish actual stationarity or assembly accuracy.
    pose = np.array([0., 0., config.get('fine_approach_z_mm', 141.24), 0., 0., 0.])
    used = 0
    events = []
    commands = []
    def measure(hint):
        nonlocal used
        if used >= len(frames):
            raise StopIteration
        _, size, saved = frames[used]
        used += 1
        return select_circle(saved['response'], size, hint, settings, .5)
    def record(event, **data):
        if event in ('fine_fallback_check', 'fine_aligned', 'fine_rejected'):
            events.append({'event': event, **data})
    def move(target):
        commands.append({'after_frame': used,
                         'planned_xy_mm': (np.asarray(target[:2]) - pose[:2]).tolist()})
        raise FirstCommand
    try:
        run_fine_alignment(frames[0][2]['expected_detection'], measure=measure,
            get_pose=lambda: pose.copy(), move=move,
            jacobian=config['fine_jacobian_px_per_mm'], target_uv=config.get('target_uv', [1280., 720.]),
            settings=settings, record=record)
        outcome = 'aligned_in_replay'
    except FirstCommand:
        outcome = 'stopped_before_first_command_execution'
    except StopIteration:
        outcome = 'saved_frames_exhausted'
    except RuntimeError as error:
        outcome = str(error)
    return {'source_run': str(run_directory), 'no_hardware_used': True,
            'note': 'Historical response replay with synthetic TCP, not a new fit or precision validation.',
            'frame_checks': checks, 'outcome': outcome, 'planned_commands': commands,
            'confirmation_events': events}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    result = replay(args.run, args.images)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)
