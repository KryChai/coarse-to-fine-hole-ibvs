"""Camera-only capture and offline radial-boundary screening. No robot imports.

This is an EXPERIMENT, not the production refinement service. It measures raw
image ellipse repeatability, not physical circle-center accuracy or coaxial pose.
One manual box per static sequence; normalized search bands stay fixed throughout.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def outline(ellipse, count=180):
    center, axes, degrees = ellipse
    a = np.deg2rad(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = np.linspace(0, 2*np.pi, count, endpoint=False)
    return (np.c_[np.cos(t), np.sin(t)]*(np.array(axes)/2)) @ rotation.T + center


def residuals(points, ellipse):
    center, axes, degrees = ellipse
    a = np.deg2rad(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = (points-center) @ rotation
    radii = np.array(axes)/2
    values = np.sum((local/radii)**2, axis=1)-1
    gradients = 2*local/radii**2
    return np.abs(values)/np.maximum(np.linalg.norm(gradients, axis=1), 1e-9)


def fit_boundary(gray, bbox, band, polarity):
    x1, y1, x2, y2 = bbox
    scale = min(x2-x1, y2-y1)
    seed = np.array([(x1+x2)/2, (y1+y2)/2])
    radius = np.arange(band[0]*scale, band[1]*scale, .5)
    if len(radius) < 7:
        raise ValueError('search_band_too_small')
    theta = np.linspace(0, 2*np.pi, 360, endpoint=False)
    direction = np.c_[np.cos(theta), np.sin(theta)]
    xy = seed+direction[:, None, :]*radius[None, :, None]
    intensity = cv2.remap(gray, xy[:, :, 0].astype('float32'), xy[:, :, 1].astype('float32'),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101).astype(float)
    gradient = (intensity[:, 2:]-intensity[:, :-2])  # difference across 1 pixel
    score = gradient if polarity == 'positive' else np.abs(gradient)
    peak = score.argmax(axis=1)
    inside = ((xy[:, :, 0] >= 1) & (xy[:, :, 0] < gray.shape[1]-1)
              & (xy[:, :, 1] >= 1) & (xy[:, :, 1] < gray.shape[0]-1)).all(axis=1)
    good = inside & (peak > 0) & (peak < score.shape[1]-1) & (score.max(axis=1) >= 2)
    indices = np.where(good)[0]
    if len(indices) < 30:
        raise ValueError('insufficient_gradient_samples')
    p = peak[indices]
    left, mid, right = score[indices, p-1], score[indices, p], score[indices, p+1]
    delta = np.clip(.5*(left-right)/np.minimum(left-2*mid+right, -1e-6), -.5, .5)
    sampled = seed+direction[indices]*(radius[1:-1][p]+.5*delta)[:, None]
    points = sampled.copy()
    for _ in range(6):
        ellipse = cv2.fitEllipseDirect(points.astype('float32'))
        if not np.isfinite([*ellipse[0], *ellipse[1], ellipse[2]]).all() or min(ellipse[1]) < 8:
            raise ValueError('degenerate_ellipse')
        distances = residuals(sampled, ellipse)
        keep = distances <= max(2., 2.5*float(np.median(distances)))
        if keep.sum() < 30:
            raise ValueError('insufficient_inliers')
        points = sampled[keep]
    ellipse = cv2.fitEllipseDirect(points.astype('float32'))
    angle = np.deg2rad(ellipse[2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    unit = (points-ellipse[0]) @ rotation/(np.array(ellipse[1])/2)
    bins = np.floor((np.arctan2(unit[:, 1], unit[:, 0])+np.pi)/(2*np.pi)*36).astype(int).clip(0, 35)
    rms = float(np.sqrt(np.mean(residuals(points, ellipse)**2)))
    coverage = int(len(np.unique(bins))*10)
    ratio = len(points)/len(sampled)
    reasons = []
    if rms > 1.2: reasons.append('rms_above_1.2px')
    if coverage < 270: reasons.append('coverage_below_270deg')
    if ratio < .65: reasons.append('inlier_fraction_below_0.65')
    if not (x1 <= ellipse[0][0] <= x2 and y1 <= ellipse[0][1] <= y2): reasons.append('center_outside_box')
    if min(ellipse[1])/max(ellipse[1]) < .25: reasons.append('ellipse_too_flat')
    return {'ellipse': ellipse, 'center_px': list(ellipse[0]), 'rms_px': rms,
            'coverage_deg': coverage, 'inlier_fraction': ratio, 'reason': ','.join(reasons),
            'quality_pass': not reasons, 'sampled_points_px': sampled.tolist(), 'inliers_px': points.tolist()}


def analyze_box(gray, bbox, inner_band=(.16, .34), outer_band=(.36, .54)):
    result = {'bbox': list(bbox), 'candidate_pair_pass': False, 'reason': '', 'boundaries': {}}
    if (not np.isfinite(bbox).all() or bbox[0] < 0 or bbox[1] < 0
            or bbox[2] > gray.shape[1] or bbox[3] > gray.shape[0]
            or min(bbox[2]-bbox[0], bbox[3]-bbox[1]) < 24):
        result['reason'] = 'invalid_box'; return result
    for name, band, polarity in [('outer', outer_band, 'absolute'), ('inner', inner_band, 'positive')]:
        try:
            fit = fit_boundary(gray, bbox, band, polarity)
            result['boundaries'][name] = fit
        except (ValueError, cv2.error, FloatingPointError) as exc:
            result['reason'] += f'{name}:{exc};'
    if len(result['boundaries']) != 2:
        return result
    outer, inner = [result['boundaries'][name] for name in ('outer', 'inner')]
    if not all(f['quality_pass'] for f in (outer, inner)):
        result['reason'] = ';'.join(f'{name}:{f["reason"]}' for name, f in result['boundaries'].items() if f['reason'])
        return result
    if max(inner['ellipse'][1]) >= .9*max(outer['ellipse'][1]):
        result['reason'] = 'boundary_sizes_not_separated'; return result
    center, axes, angle = outer['ellipse']
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    unit = (outline(inner['ellipse'])-center) @ rot/(np.array(axes)/2)
    if np.mean(np.sum(unit**2, axis=1) < 1.05) < .95:
        result['reason'] = 'inner_not_nested'; return result
    result['candidate_pair_pass'] = True
    return result


def box_variants(box):
    box = np.array(box, dtype=float)
    yield 'nominal', box
    for axis in (0, 1):
        for shift in (-10, -5, 5, 10):
            moved = box.copy(); moved[[axis, axis+2]] += shift
            yield f'axis{axis}_{shift:+d}px', moved
    center = (box[:2]+box[2:])/2
    for factor in (.95, 1.05):
        half = (box[2:]-box[:2])*factor/2
        yield f'scale_{factor}', np.r_[center-half, center+half]


def summarize(rows, static_sequence=False):
    nominal = [r for r in rows if r['variant'] == 'nominal']
    valid = [r for r in nominal if r['candidate_pair_pass']]
    summary = {'frames': len(nominal), 'passing_frames': len(valid),
               'candidate_pair_rate': len(valid)/len(nominal) if nominal else 0,
               'static_sequence_declared': static_sequence, 'static_evidence_sufficient': False,
               'physical_boundary_identity_verified': False, 'approved_for_robot_control': False,
               'temporal_centers': {}, 'box_perturbations': {}}
    by_image = {r['source']: r for r in nominal}
    for name in ('outer', 'inner'):
        centers = np.array([r['boundaries'][name]['center_px'] for r in valid])
        if len(centers):
            deviation = np.linalg.norm(centers-centers.mean(axis=0), axis=1)
            summary['temporal_centers'][name] = {'mean_px': centers.mean(axis=0).tolist(),
                'std_xy_px': centers.std(axis=0).tolist(), 'p95_deviation_px': float(np.percentile(deviation, 95)),
                'max_deviation_px': float(deviation.max())}
        shifted = [r for r in rows if r['variant'] != 'nominal' and by_image[r['source']]['candidate_pair_pass']]
        differences = [np.linalg.norm(np.array(r['boundaries'][name]['center_px'])
                       -by_image[r['source']]['boundaries'][name]['center_px']) for r in shifted if r['candidate_pair_pass']]
        summary['box_perturbations'][name] = {'trials_on_passing_nominal': len(shifted),
            'passing_trials': len(differences),
            'p95_center_change_px': float(np.percentile(differences, 95)) if differences else None,
            'max_center_change_px': float(max(differences)) if differences else None}
    # Repeated copies of one still do not constitute temporal evidence.
    summary['static_evidence_sufficient'] = bool(static_sequence and len(nominal) >= 30
        and len({r['sha256'] for r in nominal}) >= 2 and len(valid) >= 2)
    summary['temporal_numeric_screen_pass'] = bool(summary['static_evidence_sufficient']
        and summary['candidate_pair_rate'] >= .95
        and all(v['p95_deviation_px'] <= .35 and v['max_deviation_px'] <= 1
                for v in summary['temporal_centers'].values()))
    return summary


def save_panels(image, result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = result['bbox']
    margin = .15*max(x2-x1, y2-y1)
    l, t = max(0, int(x1-margin)), max(0, int(y1-margin))
    r, b = min(image.shape[1], int(x2+margin)), min(image.shape[0], int(y2+margin))
    if r <= l or b <= t: return
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if not font_path.exists():
        from matplotlib import font_manager
        font_path = Path(font_manager.findfont('DejaVu Sans'))
    font = ImageFont.truetype(str(font_path), 20)
    for lang in ('zh', 'en'):
        panels = []
        for names in ((), ('outer',), ('inner',), ('outer', 'inner')):
            view = image.copy()
            for name in names:
                fit = result['boundaries'].get(name)
                if not fit: continue
                color = (0, 255, 255) if name == 'outer' else (255, 255, 0)
                cv2.polylines(view, [np.rint(outline(fit['ellipse'])).astype('int32')], True, color, 2, cv2.LINE_AA)
                for p in fit['inliers_px']: cv2.circle(view, tuple(np.rint(p).astype(int)), 1, color, -1)
                cv2.drawMarker(view, tuple(np.rint(fit['center_px']).astype(int)), color, cv2.MARKER_CROSS, 15, 2)
            crop = Image.fromarray(cv2.cvtColor(view[t:b, l:r], cv2.COLOR_BGR2RGB))
            crop.thumbnail((520, 520))
            panel = Image.new('RGB', (560, 650), 'white')
            panel.paste(crop, ((560-crop.width)//2, 115))
            draw = ImageDraw.Draw(panel)
            labels = {'outer': '外圈候选', 'inner': '内圈候选'} if lang == 'zh' else {'outer': 'Outer candidate', 'inner': 'Inner candidate'}
            title = ' + '.join(labels[n] for n in names) if names else ('原始图像' if lang == 'zh' else 'Original image')
            draw.text((12, 6), title, font=font, fill='black')
            for i, name in enumerate(names):
                f = result['boundaries'].get(name)
                text = f'{name}: RMS={f["rms_px"]:.2f}px, {f["coverage_deg"]}deg' if f else f'{name}: extraction failed'
                draw.text((12, 35+i*26), text, font=font, fill='black')
            note = '仅椭圆拟合；未恢复真实圆心' if lang == 'zh' else 'Ellipse fits only; physical centers unknown'
            draw.text((12, 85), note, font=font, fill='black')
            key = '_'.join(names) or 'original'
            panel.save(folder/f'{key}_{lang}.png'); panels.append(panel)
        composite = Image.new('RGB', (1120, 1300), 'white')
        for i, panel in enumerate(panels): composite.paste(panel, ((i%2)*560, (i//2)*650))
        composite.save(folder/f'combined_{lang}.png')


def capture(args):
    camera = cv2.VideoCapture(args.camera)
    manifest = {'requested_size': [args.width, args.height], 'frames': [], 'robot_connected': False}
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not camera.isOpened(): raise RuntimeError('camera_not_open')
        for _ in range(30): camera.read()
        for i in range(args.frames):
            ok, frame = camera.read()
            if not ok: raise RuntimeError('camera_read_failed')
            if frame.shape[1::-1] != (args.width, args.height): raise RuntimeError(f'wrong_camera_resolution:{frame.shape[1::-1]}')
            path = args.output/f'frame_{i:04d}.png'
            if not cv2.imwrite(str(path), frame): raise OSError(f'Cannot save {path}')
            manifest['frames'].append({'file': str(path), 'captured_at': datetime.now().isoformat()})
            print(f'{i+1}/{args.frames}: {path}')
            time.sleep(args.interval)
    except Exception as exc:
        manifest['error'] = str(exc)
        raise
    finally:
        camera.release(); write_json(args.output/'capture.json', manifest)


def analyze(args):
    if not args.input.exists():
        raise ValueError(f'Input does not exist: {args.input}. Save original photos outside temporary clipboard storage.')
    files = [args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'))
    if not files: raise ValueError('No images found')
    if not (0 < args.inner_band[0] < args.inner_band[1] < args.outer_band[0] < args.outer_band[1] < 1):
        raise ValueError('Search bands must satisfy 0 < inner_low < inner_high < outer_low < outer_high < 1')
    bbox = args.bbox
    if bbox is None:
        frame = cv2.imread(str(files[0]))
        if frame is None: raise ValueError('Cannot decode first frame')
        # Select on a resized view, then map back to original pixels.
        scale = min(1., 1280/frame.shape[1])
        selection = cv2.selectROI('Select ONE outer hole, Enter to confirm', cv2.resize(frame, None, fx=scale, fy=scale), False)
        cv2.destroyAllWindows()
        x, y, w, h = np.array(selection)/scale
        if min(w, h) <= 0: raise ValueError('ROI selection cancelled')
        bbox = [x, y, x+w, y+h]
    write_json(args.output/'config.json', {'bbox': bbox, 'inner_band': args.inner_band, 'outer_band': args.outer_band,
        'algorithm': 'experimental_radial_v1', 'calibration_used': False, 'manual_box': True,
        'static_sequence_declared': args.static_sequence, 'image_paths': [str(p.resolve()) for p in files]})
    rows = []
    expected_size = None
    with (args.output/'measurements.jsonl').open('w', encoding='utf-8') as audit:
        for index, path in enumerate(files):
            image = cv2.imread(str(path)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if image is not None and expected_size is None: expected_size = image.shape[:2]
            gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.3) if image is not None else None
            for variant, box in box_variants(bbox):
                if gray is None or image.shape[:2] != expected_size:
                    result = {'bbox': box.tolist(), 'candidate_pair_pass': False, 'boundaries': {}, 'reason': 'decode_or_resolution_error'}
                else:
                    result = analyze_box(gray, box, args.inner_band, args.outer_band)
                result.update(source=str(path.resolve()), sha256=digest, variant=variant)
                audit.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n'); audit.flush(); rows.append(result)
                if variant == 'nominal':
                    if image is not None: save_panels(image, result, args.output/f'{index:04d}_{path.stem}')
                    print(f'{path.name}: candidate_pair={result["candidate_pair_pass"]} {result["reason"]}')
    summary = summarize(rows, args.static_sequence); write_json(args.output/'summary.json', summary)
    fields = ['source', 'variant', 'candidate_pair_pass', 'reason']
    fields += [f'{name}_{field}' for name in ('outer', 'inner') for field in ('u', 'v', 'rms_px', 'coverage_deg', 'inlier_fraction')]
    with (args.output/'measurements.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields[:4]}
            for name, fit in row['boundaries'].items():
                flat.update({f'{name}_u': fit['center_px'][0], f'{name}_v': fit['center_px'][1]})
                flat.update({f'{name}_{k}': fit[k] for k in ('rms_px', 'coverage_deg', 'inlier_fraction')})
            writer.writerow(flat)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture', help='Explicitly open USB camera; never connects robot')
    cap.add_argument('--camera', type=int, default=0)
    cap.add_argument('--width', type=int, default=2560); cap.add_argument('--height', type=int, default=1440)
    cap.add_argument('--frames', type=int, default=100); cap.add_argument('--interval', type=float, default=.1)
    cap.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('analyze', help='Offline only; do not mix poses for temporal statistics')
    check.add_argument('--input', type=Path, required=True); check.add_argument('--output', type=Path, required=True)
    check.add_argument('--bbox', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
    check.add_argument('--inner-band', type=float, nargs=2, default=(.16, .34))
    check.add_argument('--outer-band', type=float, nargs=2, default=(.36, .54))
    check.add_argument('--static-sequence', action='store_true')
    args = parser.parse_args()
    if args.command == 'capture' and (args.frames <= 0 or args.interval < 0 or min(args.width, args.height) <= 0):
        parser.error('Invalid capture settings')
    # One fresh output directory per run prevents stale plots/data mixing with new data.
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == 'capture': capture(args)
    else: analyze(args)


if __name__ == '__main__':
    main()
