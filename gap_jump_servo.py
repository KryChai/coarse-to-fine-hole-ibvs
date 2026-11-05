"""Callback-driven XY servo. No camera, HTTP, SDK or plotting imports.

Observations are stopped, host-timestamped samples, not exposure-synchronised data.
The feature epoch is an estimator boundary, not a claim of physical target identity.
"""
from dataclasses import asdict, dataclass, replace
import math
import time

import numpy as np

from ibvs_jacobian_estimator import (
    JacobianSettings, OnlineJacobianEstimator, calculate_dls_step, vector,
)
from servo_control import ServoSettings, ServoStepController


PROFILES = {
    "P0": dict(adaptive=False, online=False, fine=False, guard=True),
    "P1": dict(adaptive=True, online=False, fine=False, guard=True),
    "P2": dict(adaptive=True, online=True, fine=False, guard=True),
    "P3": dict(adaptive=True, online=True, fine=True, guard=False),
    "P4": dict(adaptive=True, online=True, fine=True, guard=True),
}


@dataclass(frozen=True)
class GapSettings:
    profile: str = "P4"
    target_uv: tuple = (1280., 720.)
    image_size: tuple = (2560, 1440)
    coarse_J: tuple = ((-12., 1.), (.5, 10.))
    fine_J: tuple = ((-12., 1.), (.5, 10.))
    coarse_gain: float = .03
    coarse_max_gain: float = .15
    fine_gain: float = .1
    fine_max_gain: float = .1
    coarse_step_mm: float = 1.
    fine_step_mm: float = .15
    coarse_tol_px: float = 2.
    fine_tol_px: float = 1.
    fine_tol_mm: float = .05
    stable_frames: int = 3
    stable_spread_px: float = .35
    handoff_frames: int = 3
    handoff_spread_px: float = .35
    association_px: float = 40.
    ambiguity_px: float = 3.
    max_failures: int = 3
    max_non_improving: int = 8
    coarse_iterations: int = 150
    fine_iterations: int = 100
    min_motion_mm: float = .05
    min_pixel_motion_px: float = 2.
    beta: float = .3
    damping_ratio: float = .03
    pose_tolerance_mm: float = .1
    pose_tolerance_deg: float = .1
    stationary_mm: float = .02
    max_sample_age_s: float = 30.
    max_pair_interval_s: float = 60.
    max_rejected_updates: int = 8
    fine_learning: bool = True

    def validate(self, hardware=False):
        if self.profile not in PROFILES:
            raise ValueError("profile must be P0..P4")
        if hardware and not PROFILES[self.profile]["guard"]:
            raise ValueError("P3 is a deliberately unsafe simulation-only ablation")
        exempt = {"profile", "target_uv", "image_size", "coarse_J", "fine_J", "fine_learning"}
        for key, value in asdict(self).items():
            if key not in exempt and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("stable_frames", "handoff_frames", "max_failures", "max_non_improving",
                    "coarse_iterations", "fine_iterations", "max_rejected_updates"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if type(self.fine_learning) is not bool:
            raise ValueError("fine_learning must be boolean")
        size, uv = vector(self.image_size, 2), vector(self.target_uv, 2)
        if np.any(size <= 0) or np.any(size != np.floor(size)) or np.any(uv < 0) or np.any(uv >= size):
            raise ValueError("invalid raw image size/target")
        if (self.stable_frames < 3 or self.handoff_frames < 3 or self.beta > 1
                or self.stable_spread_px >= self.fine_tol_px
                or self.coarse_step_mm > 1 or self.fine_step_mm > .15):
            raise ValueError("invalid stability, beta or physical step bounds")
        for stage in ("coarse", "fine"):
            self.estimator(stage)  # validates matrix scale and condition
            self.controller(stage, getattr(self, stage + "_J"))
        return self

    def estimator(self, stage):
        cfg = JacobianSettings(mode="fixed", max_step_mm=getattr(self, stage+"_step_mm"),
            image_size=self.image_size, target_uv=self.target_uv,
            beta=self.beta, min_motion_mm=self.min_motion_mm,
            min_pixel_motion_px=self.min_pixel_motion_px, damping_ratio=self.damping_ratio,
            arrival_tolerance_mm=self.pose_tolerance_mm,
            max_orientation_change_deg=self.pose_tolerance_deg,
            max_sample_age_s=self.max_sample_age_s, max_pair_interval_s=self.max_pair_interval_s)
        return OnlineJacobianEstimator(getattr(self, stage+"_J"), cfg)

    def controller(self, stage, matrix):
        fine = stage == "fine"
        return ServoStepController(matrix, ServoSettings(
            mode="adaptive" if PROFILES[self.profile]["adaptive"] else "fixed",
            gain=getattr(self, stage+"_gain"), max_gain=getattr(self, stage+"_max_gain"),
            max_step_mm=getattr(self, stage+"_step_mm"),
            align_tol_px=getattr(self, stage+"_tol_px"),
            transition_error_px=10. if fine else 40., slowdown_error_px=5. if fine else 20.))


@dataclass
class Observation:
    center: object
    anchor: object  # box centre, used for association in BOTH feature stages
    tcp: object
    frame_id: int
    timestamp: float
    feature: str
    confidence: float = .9
    noise_px: float = 0.
    image_size: tuple = (2560, 1440)
    target_id: str = "locked-hole"  # local track token unless service supplies a stable ID
    valid: bool = True
    quality: object = None

    def sample(self):
        return dict(tcp=list(self.tcp), center=list(self.center), frame_id=self.frame_id,
                    timestamp=self.timestamp, conf=self.confidence, noise_px=self.noise_px,
                    image_size=list(self.image_size), valid=self.valid, settled=True)


def associate(candidates, predicted, radius, ambiguity):
    """Choose proximity BEFORE quality. Never replace a rejected target by another hole."""
    if not candidates:
        return None
    if predicted is None:
        return max(candidates, key=lambda c: c["confidence"])
    ranked = sorted((float(np.linalg.norm(vector(c["anchor"], 2)-predicted)), i, c)
                    for i, c in enumerate(candidates))
    if ranked[0][0] > radius:
        return None
    if len(ranked) > 1 and ranked[1][0] <= radius and ranked[1][0]-ranked[0][0] < ambiguity:
        return None
    return ranked[0][2]


def run_alignment(settings, *, measure, move, get_pose, record=lambda *a, **k: None,
                  clock=time.monotonic, hardware=False):
    """measure(stage, predicted_box_center) -> Observation | None.

    move(tcp) must block and validate stop/arrival. Returned aligned pose alone
    authorises the caller's OFFSET; exceptions must block subsequent assembly.
    Each invocation starts new local target, feature epoch and model banks.
    """
    cfg = settings.validate(hardware)
    flags = PROFILES[cfg.profile]
    stage = "coarse"
    models = {s: cfg.estimator(s) for s in ("coarse", "fine")}
    controller = cfg.controller(stage, models[stage].J)
    reference = vector(get_pose(), 6)
    pending = None
    hint = None
    target_id = None
    last_frame, last_time = 0, -math.inf
    failures = stalled = rejected = 0
    best = math.inf
    stable, overlap = [], []
    handoff_pose = None
    stage_iteration = 0
    naive_boundary = False
    record("gap_start", settings=asdict(cfg), hardware=hardware)

    def discard(reason):
        nonlocal pending
        if pending is not None:
            record("pair_discarded", reason=reason, feature=pending[0].feature,
                   frame_id=pending[0].frame_id)
        pending = None

    try:
        while True:
            stage_iteration += 1
            if stage_iteration > getattr(cfg, stage+"_iterations"):
                raise RuntimeError(stage+" iteration limit")
            obs = measure(stage, None if hint is None else hint.copy())
            if obs is None:
                failures += 1
                stable.clear()
                overlap.clear()
                handoff_pose = None
                discard("detection_gap")
                controller.detection_lost()
                record("measurement_rejected", stage=stage, reason="missing_or_quality_or_association",
                       consecutive=failures)
                if failures >= cfg.max_failures:
                    raise RuntimeError("consecutive detection/quality/association failures")
                continue
            failures = 0
            center, anchor, pose = vector(obs.center, 2), vector(obs.anchor, 2), vector(obs.tcp, 6)
            if (not obs.valid or obs.feature != stage or type(obs.frame_id) is not int
                    or obs.frame_id <= last_frame or not np.isfinite(obs.timestamp)
                    or obs.timestamp <= last_time or not 0 <= clock()-obs.timestamp <= cfg.max_sample_age_s
                    or tuple(obs.image_size) != tuple(cfg.image_size)
                    or not np.isfinite(obs.confidence) or not .5 <= obs.confidence <= 1
                    or not np.isfinite(obs.noise_px) or not 0 <= obs.noise_px <= getattr(cfg, stage+"_tol_px")
                    or np.any(center < 0) or np.any(center >= cfg.image_size)
                    or np.any(anchor < 0) or np.any(anchor >= cfg.image_size)):
                raise RuntimeError("invalid, stale, noisy or cross-feature observation")
            if target_id is not None and obs.target_id != target_id:
                raise RuntimeError("target identity changed; restart alignment explicitly")
            target_id = obs.target_id
            if hint is not None and np.linalg.norm(anchor-hint) > cfg.association_px:
                raise RuntimeError("target association outside predicted gate")
            live = vector(get_pose(), 6)
            angles = lambda a, b: np.max(np.abs((a[3:]-b[3:]+180) % 360-180))
            if (np.linalg.norm(live[:3]-pose[:3]) > cfg.stationary_mm
                    or angles(live, pose) > cfg.pose_tolerance_deg
                    or abs(pose[2]-reference[2]) > cfg.pose_tolerance_mm
                    or angles(pose, reference) > cfg.pose_tolerance_deg):
                raise RuntimeError("measurement pose/Z/orientation mismatch")
            last_frame, last_time = obs.frame_id, obs.timestamp
            model = models[stage]
            if pending is not None and flags["online"] and (stage != "fine" or cfg.fine_learning):
                before, commanded = pending
                # P3 intentionally retains the last BOX sample across the first
                # fine motion, reproducing a specific defective concatenation.
                if naive_boundary and np.linalg.norm(pose[:2]-np.asarray(before.tcp)[:2]) < cfg.min_motion_mm:
                    pass
                else:
                    if flags["guard"] and (before.feature != obs.feature or before.target_id != obs.target_id):
                        discard("epoch_or_identity")
                    else:
                        sample = obs.sample()
                        sample["commanded_tcp"] = commanded
                        update = model.update(before.sample(), sample, robot_settled=True, now=clock())
                        record("jacobian_update", stage=stage, cross_feature=before.feature != obs.feature,
                               before_frame=before.frame_id, after_frame=obs.frame_id, **update)
                        if update["accepted"]:
                            rejected = 0
                        elif update["reason"] not in ("motion_too_small", "visual_change_too_small"):
                            rejected += 1
                        if rejected >= cfg.max_rejected_updates:
                            raise RuntimeError("consecutive rejected Jacobian updates")
                        pending = None
                        naive_boundary = False
            error = center-vector(cfg.target_uv, 2)
            norm = float(np.linalg.norm(error))
            mm_error = float(np.linalg.norm(np.linalg.solve(model.J, error)))
            record("measurement", stage=stage, iteration=stage_iteration, observation=asdict(obs),
                   error_px=norm, estimated_mm=mm_error, J=model.J.tolist())
            hint = anchor.copy()

            if stage == "fine" and flags["guard"] and len(overlap) < cfg.handoff_frames:
                if handoff_pose is None:
                    handoff_pose = pose.copy()
                if (np.linalg.norm(pose[:3]-handoff_pose[:3]) > cfg.stationary_mm
                        or angles(pose, handoff_pose) > cfg.pose_tolerance_deg):
                    raise RuntimeError("robot pose changed during feature handoff")
                overlap.append(center-anchor)
                record("handoff_sample", offset_px=(center-anchor).tolist(), tcp=pose.tolist())
                if len(overlap) == cfg.handoff_frames:
                    spread = float(np.max(np.linalg.norm(overlap-np.mean(overlap, axis=0), axis=1)))
                    if spread > cfg.handoff_spread_px:
                        raise RuntimeError("stationary box/circle offset is unstable")
                    record("handoff_validated", offset_px=np.mean(overlap, axis=0).tolist(),
                           spread_px=spread, fine_J=model.J.tolist(), policy="independent_initial_J")
                continue  # no motion, no learning during paired-feature validation

            if norm < getattr(cfg, stage+"_tol_px") and (stage == "coarse" or mm_error < cfg.fine_tol_mm):
                if stage == "coarse" and flags["fine"]:
                    record("feature_switch", from_feature=stage, to_feature="fine",
                           tcp=pose.tolist(), coarse_J=model.J.tolist(), guarded=flags["guard"])
                    if flags["guard"]:
                        discard("feature_switch")
                    else:
                        models["fine"] = model
                        pending = (obs, pose.tolist())
                        naive_boundary = True
                    stage = "fine"
                    controller = cfg.controller(stage, models[stage].J)
                    stable.clear()
                    overlap.clear()
                    handoff_pose = None
                    stage_iteration = stalled = rejected = 0
                    best = math.inf
                    continue
                stable.append(center)
                required = cfg.stable_frames if stage == "fine" else 1
                stable = stable[-required:]
                if len(stable) == required and (stage != "fine" or float(np.max(
                        np.linalg.norm(stable-np.mean(stable, axis=0), axis=1))) <= cfg.stable_spread_px):
                    discard("aligned")
                    record("aligned", stage=stage, tcp=pose.tolist(), error_px=norm,
                           estimated_mm=mm_error, J=model.J.tolist())
                    return pose.tolist()
                continue
            stable.clear()
            if norm < best-.001:
                best, stalled = norm, 0
            else:
                stalled += 1
            if stalled >= cfg.max_non_improving:
                raise RuntimeError(stage+" error not improving")
            controller.inverse = np.linalg.inv(model.J)
            step = controller.step(error)
            dls_cfg = replace(model.settings, gain=step.effective_gain, max_step_mm=step.step_limit_mm)
            delta, diagnostics = calculate_dls_step(model.J, error, dls_cfg, return_diagnostics=True)
            destination = pose.copy()
            destination[:2] += delta
            record("control", stage=stage, target_tcp=destination.tolist(),
                   **step.telemetry(), dls=diagnostics)
            move(destination.tolist())
            actual = vector(get_pose(), 6)
            if (np.linalg.norm(actual[:3]-destination[:3]) > cfg.pose_tolerance_mm
                    or angles(actual, destination) > cfg.pose_tolerance_deg):
                raise RuntimeError("actual arrival mismatch")
            record("motion_complete", stage=stage, tcp=actual.tolist(),
                   actual_delta=(actual[:2]-pose[:2]).tolist())
            # Preserve the obsolete coarse sample only in the labelled P3 defect model.
            if naive_boundary:
                pending = (pending[0], destination.tolist())
            else:
                pending = (obs, destination.tolist())
            hint = anchor+model.J@(actual[:2]-pose[:2])
    except BaseException as error:
        discard("aborted")
        record("alignment_aborted", stage=stage, reason=str(error))
        raise
