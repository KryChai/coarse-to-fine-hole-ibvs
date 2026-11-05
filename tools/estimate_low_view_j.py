"""Offline low-view J estimate from hardware telemetry; never moves hardware."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


PRIOR_J = np.array([[-18.47, 1.52], [0.80, 21.19]])
RIDGE = 0.0036  # mm^2: stabilizes the weakly excited direction of this run
HUBER_PX = 1.5


def extract_pairs(rows, min_travel_mm=0.02):
    """Associate each image with its first subsequent, pre-control TCP feedback.

    The pose is sampled shortly after the image (not synchronized exposure).
    Reject hole transitions, lost detections, tiny travel and Z changes.
    """
    indices = [i for i, row in enumerate(rows)
               if row['event'] == 'vision' and row['phase'] == 'servo_align']
    samples = []
    for k, i in enumerate(indices):
        end = indices[k + 1] if k + 1 < len(indices) else len(rows)
        pose = None
        for row in rows[i + 1:end]:
            if row['event'] == 'control':
                break
            if row['event'] == 'feedback' and row['phase'] == 'servo_align' and row['tcp_x_mm']:
                pose = np.array([float(row[f'tcp_{axis}_mm']) for axis in 'xyz'])
                break
        error = np.array([float(rows[i][f'error_{axis}_px']) for axis in 'uv'])
        samples.append((i, pose, error))
    travel, pixel = [], []
    for (i, p, e), (j, q, f) in zip(samples, samples[1:]):
        if p is None or q is None or rows[i]['hole_index'] != rows[j]['hole_index']:
            continue
        between = rows[i + 1:j]
        controls = [r for r in between if r['event'] == 'control' and r['phase'] == 'servo_align']
        if len(controls) != 1 or any(r['event'] == 'detection_failed' for r in between):
            continue
        delta = q - p
        if (not np.all(np.isfinite(np.r_[delta, e, f])) or
                abs(delta[2]) > 0.05 or np.linalg.norm(delta[:2]) < min_travel_mm):
            continue
        travel.append(delta[:2])
        pixel.append(f - e)
    return np.asarray(travel).reshape(-1, 2), np.asarray(pixel).reshape(-1, 2)


def fit_j(travel, pixel, prior=PRIOR_J):
    if len(travel) < 3 or np.linalg.matrix_rank(travel) < 2:
        raise ValueError('Need at least three finite motion pairs spanning two XY directions')
    prior = np.asarray(prior, dtype=float).T
    weights = np.ones(len(travel))
    for _ in range(20):
        coefficients = np.linalg.solve(
            travel.T @ (weights[:, None] * travel) + RIDGE * np.eye(2),
            travel.T @ (weights[:, None] * pixel) + RIDGE * prior)
        residuals = np.linalg.norm(pixel - travel @ coefficients, axis=1)
        weights = np.minimum(1.0, HUBER_PX / np.maximum(residuals, 1e-9))
    return coefficients.T


def analyze(directory):
    root = Path(directory)
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('source') != 'hardware':
        raise ValueError('Only hardware telemetry may supply this estimate')
    with (root / 'samples.csv').open(encoding='utf-8', newline='') as stream:
        travel, pixel = extract_pairs(list(csv.DictReader(stream)))
    estimate = fit_j(travel, pixel)
    def rms(jacobian):
        return float(np.sqrt(np.mean(np.sum((pixel - travel @ jacobian.T) ** 2, axis=1))))
    return {
        'source_run': root.name, 'phase': 'servo_align', 'pairs': len(travel),
        'motion_singular_values_mm': np.linalg.svd(travel, compute_uv=False).tolist(),
        'motion_condition': float(np.linalg.cond(travel)),
        'prior_j_px_per_mm': PRIOR_J.tolist(), 'ridge_mm_squared': RIDGE,
        'huber_px': HUBER_PX, 'estimated_j_px_per_mm': estimate.tolist(),
        'prior_residual_rms_px': rms(PRIOR_J), 'fit_residual_rms_px': rms(estimate),
        'limitations': 'Prior-dependent local estimate from frame centers, unsynchronized TCP; '
                       'not an independent circle-center or full 2D calibration.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), ensure_ascii=False, indent=2))
