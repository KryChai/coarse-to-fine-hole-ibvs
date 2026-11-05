"""m x 2 rotation Jacobian, measured finite differences, DLS and gated Broyden.

The SDK's RX/RY fields are in degrees; visual pose features are in radians.
No camera, network, SDK or implicit recovery movement belongs here.
"""
from dataclasses import dataclass
import time

import numpy as np

from .geometry import finite, MeasurementError
from auto_cam_level.parameters import CONTROL, CONTROL_GATES as G


def angle_delta(a, b):
    return (np.asarray(a)-np.asarray(b)+180.) % 360.-180.


@dataclass
class ControlSettings:
    gain: float = CONTROL["gain"]
    damping_ratio: float = CONTROL["damping_ratio"]
    max_step_deg: float = CONTROL["max_step_deg"]
    probe_deg: float = CONTROL["probe_deg"]
    max_excursion_deg: float = CONTROL["max_excursion_deg"]
    max_travel_deg: float = CONTROL["max_travel_deg"]
    arrival_deg: float = CONTROL["arrival_deg"]
    position_mm: float = CONTROL["position_mm"]
    max_age_s: float = CONTROL["max_age_s"]
    min_singular: float = CONTROL["min_singular"] # whitened feature / degree
    max_condition: float = CONTROL["max_condition"]
    beta: float = CONTROL["beta"]
    stable_frames: int = CONTROL["stable_frames"]
    max_iterations: int = CONTROL["max_iterations"]
    max_non_improving: int = CONTROL["max_non_improving"]
    online: bool = CONTROL["online"]

    def validate(self):
        if not all(np.isfinite(v) and v > 0 for v in G.values()) or G["improvement_ratio"] > 1:
            raise ValueError("invalid CONTROL_GATES in parameters.py")
        for key, value in vars(self).items():
            if key != "online" and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("stable_frames", "max_iterations", "max_non_improving"):
            if type(getattr(self, key)) is not int:
                raise ValueError(f"{key} must be an integer")
        if type(self.online) is not bool or self.stable_frames < 3:
            raise ValueError("online must be bool; stable_frames must be >=3")
        if (self.gain > 1 or self.beta > 1 or self.damping_ratio > 1
                or not 3*self.arrival_deg <= self.probe_deg <= self.max_step_deg <= 1
                or self.max_step_deg > self.max_excursion_deg or self.max_condition < 1):
            raise ValueError("invalid rotation gain, probe, step or conditioning bounds")
        return self


@dataclass
class Observation:
    feature: object
    tcp: object
    frame_id: int
    timestamp: float
    noise: object
    metadata: object = None


class RotationServo:
    def __init__(self, measure, get_pose, move, tolerance, settings=None, record=None, clock=time.monotonic):
        self.measure, self.get_pose, self.move = measure, get_pose, move
        self.record = record or (lambda event, **data: None)
        def tracked_pose():
            pose = get_pose()
            self.record("pose_read", tcp=list(pose))
            return pose
        self.get_pose = tracked_pose
        self.cfg = (settings or ControlSettings()).validate()
        self.tolerance = np.asarray(tolerance, dtype=float)
        if self.tolerance.ndim != 1 or len(self.tolerance) not in (2, 4) or not np.isfinite(self.tolerance).all() or np.any(self.tolerance <= 0):
            raise ValueError("expected 2 or 4 positive feature tolerances")
        self.origin = finite(self.get_pose(), (6,)).copy()
        self.clock, self.last_frame, self.last_time = clock, -1, -np.inf
        self.travel = 0.
        self.phase = "initial"

    def check_pose(self, pose):
        p = finite(pose, (6,))
        if (np.linalg.norm(p[:3]-self.origin[:3]) > self.cfg.position_mm
                or abs(angle_delta(p[5], self.origin[5])) > self.cfg.arrival_deg
                or np.linalg.norm(angle_delta(p[3:5], self.origin[3:5])) > self.cfg.max_excursion_deg):
            raise RuntimeError("XYZ/RZ changed or RX/RY excursion exceeded")
        return p

    def sample(self):
        obs = self.measure()
        now = self.clock()
        from dataclasses import asdict
        self.record("measurement_input", now=now, observation=asdict(obs) if isinstance(obs, Observation) else None)
        if not isinstance(obs, Observation):
            raise MeasurementError("rotation measurement missing")
        obs.feature = finite(obs.feature, self.tolerance.shape)
        obs.noise = finite(obs.noise, self.tolerance.shape)
        obs.tcp = self.check_pose(obs.tcp)
        if (type(obs.frame_id) is not int or obs.frame_id <= self.last_frame
                or not np.isfinite(obs.timestamp) or obs.timestamp <= self.last_time
                or not 0 <= now-obs.timestamp <= self.cfg.max_age_s):
            raise MeasurementError("stale/duplicate rotation observation")
        actual = self.check_pose(self.get_pose())
        if (np.linalg.norm(actual[:3]-obs.tcp[:3]) > self.cfg.position_mm
                or np.max(np.abs(angle_delta(actual[3:], obs.tcp[3:]))) > self.cfg.arrival_deg):
            raise RuntimeError("robot moved after observation")
        if np.any(obs.noise < 0) or np.linalg.norm(obs.noise/self.tolerance) > G["noise_norm"]:
            raise MeasurementError("feature noise exceeds convergence tolerance")
        self.last_frame, self.last_time = obs.frame_id, obs.timestamp
        self.record("observation", phase=self.phase, frame_id=obs.frame_id, timestamp=obs.timestamp,
                    tcp=obs.tcp.tolist(), feature=obs.feature.tolist(), noise=obs.noise.tolist(), metadata=obs.metadata)
        return obs

    def move_checked(self, target):
        before = self.check_pose(self.get_pose())
        target = self.check_pose(target)
        delta = angle_delta(target[3:5], before[3:5])
        norm = float(np.linalg.norm(delta))
        if norm > self.cfg.max_step_deg+1e-8 or self.travel+norm > self.cfg.max_travel_deg:
            raise RuntimeError("rotation step or total travel limit exceeded")
        self.record("motion_requested", target_tcp=target.tolist(), delta_deg=delta.tolist())
        self.move(target.tolist())
        after = self.check_pose(self.get_pose())
        if (np.linalg.norm(after[:3]-target[:3]) > self.cfg.position_mm
                or np.max(np.abs(angle_delta(after[3:], target[3:]))) > self.cfg.arrival_deg):
            raise RuntimeError("rotation target not reached")
        self.travel += norm
        self.record("motion_reached", actual_tcp=after.tolist(), travel_deg=self.travel,
                    actual_delta_deg=angle_delta(after[3:5], before[3:5]).tolist())

    def validate_jacobian(self, matrix):
        J = finite(matrix, (len(self.tolerance), 2))
        singular = np.linalg.svd(J/self.tolerance[:, None], compute_uv=False)
        if singular[-1] < self.cfg.min_singular or singular[0]/singular[-1] > self.cfg.max_condition:
            raise MeasurementError("rotation Jacobian unobservable/ill-conditioned; recalibrate or use enhanced features")
        return J.copy()

    def calibrate(self, initial=None):
        self.phase = "calibration"
        initial = initial or self.sample()
        origin = initial.tcp.copy()
        samples = [initial]
        for axis in (0, 1):
            for sign in (1, -1):
                target = origin.copy()
                target[3+axis] += sign*self.cfg.probe_deg
                self.move_checked(target)
                samples.append(self.sample())
                # Return only after a valid measurement. Failure never triggers
                # an unobserved rollback; the caller must inspect the last pose.
                self.move_checked(origin)
                returned = self.sample()
                if np.linalg.norm((returned.feature-initial.feature)/self.tolerance) > G["return_error"]:
                    raise MeasurementError("rotation probe return inconsistent")
        Q = np.array([angle_delta(s.tcp[3:5], origin[3:5]) for s in samples])
        S = np.array([s.feature for s in samples])
        Qc, Sc = Q-Q.mean(axis=0), S-S.mean(axis=0)
        if np.linalg.svd(Qc, compute_uv=False)[-1] < self.cfg.probe_deg:
            raise MeasurementError("insufficient actual RX/RY probe excitation")
        J = self.validate_jacobian(np.linalg.lstsq(Qc, Sc, rcond=None)[0].T)
        noise = np.max([s.noise for s in samples], axis=0)/self.tolerance
        # The observation matrix has rank TWO even for four visual features.
        signal = np.linalg.svd(Qc@(J/self.tolerance[:, None]).T, compute_uv=False)[1]
        if signal < max(G["min_probe_signal"], G["noise_sigmas"]*np.linalg.norm(noise)):
            raise MeasurementError("probe feature displacement below noise floor")
        residual = np.max(np.linalg.norm((Sc-Qc@J.T)/self.tolerance, axis=1))
        if residual > max(G["calibration_residual"], G["noise_sigmas"]*np.linalg.norm(noise)):
            raise MeasurementError("probe model residual too large")
        self.record("rotation_calibration", J=J.tolist(), units="feature/degree",
                    origin_tcp=origin.tolist(), residual=residual, signal=signal)
        return J

    def converged(self, obs, window):
        if np.linalg.norm(obs.feature/self.tolerance) > 1:
            window.clear()
            return False
        window.append(obs.feature.copy())
        del window[:-self.cfg.stable_frames]
        if len(window) < self.cfg.stable_frames:
            return False
        spread = np.max(np.linalg.norm((np.asarray(window)-np.mean(window, axis=0))/self.tolerance, axis=1))
        return spread <= G["stable_spread"]

    def run(self, initial_J=None):
        try:
            return self._run(initial_J)
        except BaseException as exc:
            self.record("levelling_failed", reason=str(exc), exception=type(exc).__name__)
            raise

    def _run(self, initial_J):
        from dataclasses import asdict
        self.record("controller_start", initial_J=None if initial_J is None else np.asarray(initial_J).tolist(),
                    settings=asdict(self.cfg), guards=dict(G), tolerance=self.tolerance.tolist(), origin_tcp=self.origin.tolist())
        obs, stable = self.sample(), []
        # Avoid probing a level plane, especially the quadratic shape-only zero.
        confirmations = 0
        while np.linalg.norm(obs.feature/self.tolerance) <= 1:
            confirmations += 1
            if confirmations > self.cfg.max_iterations:
                raise MeasurementError("level features failed stationary stability confirmation")
            if self.converged(obs, stable):
                self.record("levelled", tcp=obs.tcp.tolist(), iterations=0)
                return obs.tcp.tolist()
            obs = self.sample()
        stable.clear()  # Never carry confirmations across probe/control motion.
        J = self.calibrate(obs) if initial_J is None else self.validate_jacobian(initial_J)
        reference = J.copy()
        self.phase = "control"
        obs, previous, stagnant = self.sample(), None, 0
        for iteration in range(self.cfg.max_iterations):
            error = float(np.linalg.norm(obs.feature/self.tolerance))
            if previous is not None:
                before, old_error = previous
                dq = angle_delta(obs.tcp[3:5], before.tcp[3:5])
                ds = obs.feature-before.feature
                reason = "fixed"
                if self.cfg.online:
                    reason = "noise_or_small_motion"
                    noise = np.linalg.norm((obs.noise+before.noise)/self.tolerance)
                    if np.linalg.norm(dq) > G["min_update_motion_deg"] and np.linalg.norm(ds/self.tolerance) > max(G["noise_sigmas"]*noise, G["min_update_signal"]):
                        candidate = J+self.cfg.beta*np.outer(ds-J@dq, dq)/(dq@dq)
                        try:
                            candidate = self.validate_jacobian(candidate)
                            if (np.linalg.norm((candidate-J)/self.tolerance[:, None]) <= G["max_relative_update"]*np.linalg.norm(J/self.tolerance[:, None])
                                    and np.linalg.norm((candidate-reference)/self.tolerance[:, None]) <= G["max_reference_change"]*np.linalg.norm(reference/self.tolerance[:, None])):
                                J, reason = candidate, "accepted"
                            else:
                                reason = "update_limit"
                        except MeasurementError:
                            reason = "conditioning"
                self.record("broyden", reason=reason, actual_delta_deg=dq.tolist(), J=J.tolist())
                stagnant = stagnant+1 if error >= old_error*G["improvement_ratio"] else 0
                if stagnant >= self.cfg.max_non_improving:
                    raise RuntimeError("rotation error is not improving")
            if self.converged(obs, stable):
                self.record("levelled", tcp=obs.tcp.tolist(), iterations=iteration, J=J.tolist())
                return obs.tcp.tolist()
            if error <= 1:
                previous = None
                obs = self.sample()
                continue
            A = self.validate_jacobian(J)/self.tolerance[:, None]
            U, singular, Vt = np.linalg.svd(A, full_matrices=False)
            mu = self.cfg.damping_ratio*singular[0]
            delta = -self.cfg.gain*Vt.T@((singular/(singular**2+mu**2))*(U.T@(obs.feature/self.tolerance)))
            raw_delta = delta.copy()
            delta *= min(1., self.cfg.max_step_deg/max(np.linalg.norm(delta), 1e-12))
            target = obs.tcp.copy()
            target[3:5] += delta
            self.record("control", iteration=iteration, error_norm=error, delta_deg=delta.tolist(),
                        raw_delta_deg=raw_delta.tolist(), mu=float(mu), singular_values=singular.tolist(), J=J.tolist())
            self.move_checked(target)
            previous = obs, error
            obs = self.sample()
        raise RuntimeError("rotation iteration limit exceeded")
