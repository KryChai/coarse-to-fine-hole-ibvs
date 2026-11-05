"""Bounded fixed/adaptive XY IBVS steps; no camera, network or robot imports."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ServoSettings:
    mode: str = "adaptive"
    gain: float = 0.03
    max_gain: float = 0.15
    align_tol_px: float = 2.0
    transition_error_px: float = 40.0
    slowdown_error_px: float = 20.0
    max_step_mm: float = 1.0
    near_step_ratio: float = 0.2
    backoff_factor: float = 0.5
    min_gain_scale: float = 0.125
    recovery_factor: float = 1.1
    increase_tolerance: float = 0.02

    def validate(self):
        if self.mode not in ("fixed", "adaptive"):
            raise ValueError("servo mode must be fixed or adaptive")
        numeric = {key: value for key, value in asdict(self).items() if key != "mode"}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError("servo settings must be finite")
        if not 0 < self.gain <= self.max_gain < 1:
            raise ValueError("require 0 < gain <= max_gain < 1")
        if not 0 < self.align_tol_px < self.slowdown_error_px <= self.transition_error_px:
            raise ValueError("require 0 < alignment tolerance < slowdown <= transition error")
        if self.max_step_mm <= 0 or not 0 < self.near_step_ratio <= 1:
            raise ValueError("step limit must be positive and near_step_ratio in (0, 1]")
        if not 0 < self.backoff_factor < 1 or not 0 < self.min_gain_scale <= 1:
            raise ValueError("invalid adaptive backoff bounds")
        if self.recovery_factor <= 1 or not 0 <= self.increase_tolerance < 1:
            raise ValueError("invalid gain recovery / increase tolerance")


@dataclass(frozen=True)
class ServoStep:
    requested: np.ndarray
    selected: np.ndarray
    nominal_gain: float
    effective_gain: float
    gain_scale: float
    step_limit_mm: float
    backoff_applied: bool

    def telemetry(self):
        requested_norm = math.hypot(*self.requested)
        selected_norm = math.hypot(*self.selected)
        return {
            "requested_dx_mm": float(self.requested[0]), "requested_dy_mm": float(self.requested[1]),
            "dx_mm": float(self.selected[0]), "dy_mm": float(self.selected[1]),
            "step_limited": int(requested_norm > self.step_limit_mm),
            "nominal_gain": self.nominal_gain, "effective_gain": self.effective_gain,
            "gain_scale": self.gain_scale, "step_limit_mm": self.step_limit_mm,
            "requested_step_mm": requested_norm, "selected_step_mm": selected_norm,
            "backoff_applied": int(self.backoff_applied),
        }


class ServoStepController:
    """Error-based scheduling plus response backoff; a new instance per alignment.

    Nominal gain rises linearly to max_gain as error grows. The physical step
    ceiling shrinks near alignment. Divergence or a direction reversal halves
    both gain and ceiling; improvement gradually restores the scale.
    """

    def __init__(self, jacobian, settings=None):
        self.settings = settings or ServoSettings()
        self.settings.validate()
        jacobian = np.asarray(jacobian, dtype=float)
        if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
            raise ValueError("IBVS J must be a finite 2x2 matrix")
        condition = float(np.linalg.cond(jacobian))
        if not math.isfinite(condition) or condition > 1e6:
            raise ValueError("IBVS J is singular or ill-conditioned")
        self.inverse = np.linalg.inv(jacobian)
        if not np.all(np.isfinite(self.inverse)):
            raise ValueError("IBVS J inverse is not finite")
        self.previous_error = None
        self.scale = 1.0
        self._pending_backoff = False

    def detection_lost(self):
        if self.settings.mode == "adaptive":
            self.scale = max(self.settings.min_gain_scale, self.scale * self.settings.backoff_factor)
            self._pending_backoff = True
        self.previous_error = None

    def step(self, error):
        error = np.asarray(error, dtype=float)
        if error.shape != (2,) or not np.all(np.isfinite(error)):
            raise ValueError("image error must contain two finite pixel values")
        size = math.hypot(*error)
        if not math.isfinite(size):
            raise ValueError("image error norm is invalid")
        cfg = self.settings
        nominal = cfg.gain
        limit = cfg.max_step_mm
        backed_off = self._pending_backoff
        self._pending_backoff = False
        if cfg.mode == "adaptive":
            if self.previous_error is not None:
                previous_size = math.hypot(*self.previous_error)
                reversed_direction = float(error @ self.previous_error) < 0
                if size > previous_size * (1 + cfg.increase_tolerance) or reversed_direction:
                    self.scale = max(cfg.min_gain_scale, self.scale * cfg.backoff_factor)
                    backed_off = True
                elif size < previous_size * (1 - cfg.increase_tolerance):
                    self.scale = min(1.0, self.scale * cfg.recovery_factor)
            fraction = float(np.clip((size - cfg.align_tol_px) /
                                    (cfg.transition_error_px - cfg.align_tol_px), 0, 1))
            nominal += (cfg.max_gain - cfg.gain) * fraction
            near_fraction = float(np.clip((size - cfg.align_tol_px) /
                                         (cfg.slowdown_error_px - cfg.align_tol_px), 0, 1))
            limit *= (cfg.near_step_ratio + (1 - cfg.near_step_ratio) * near_fraction) * self.scale
        gain = nominal * self.scale
        requested = -gain * (self.inverse @ error)
        if size < cfg.align_tol_px:
            requested = np.zeros(2)
        if not np.all(np.isfinite(requested)):
            raise ValueError("computed servo displacement is invalid")
        length = math.hypot(*requested)
        if not math.isfinite(length):
            raise ValueError("computed servo displacement norm is invalid")
        selected = requested * (limit / length) if length > limit else requested.copy()
        self.previous_error = error.copy()
        return ServoStep(requested, selected, nominal, gain, self.scale, limit, backed_off)
