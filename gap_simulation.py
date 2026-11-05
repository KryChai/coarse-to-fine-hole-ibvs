"""Explicit synthetic linear plant; no YOLO, images or physical accuracy claims."""
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import numpy as np

from gap_jump_servo import GapSettings, Observation, PROFILES, associate, run_alignment
from gap_runtime import GapLog


SCENARIOS = ("nominal", "feature_bias", "dropout", "ambiguous", "unstable_handoff", "stale")


class SyntheticPlant:
    def __init__(self, settings, seed=0, scenario="feature_bias"):
        self.cfg, self.scenario = settings, scenario
        self.rng = np.random.default_rng(seed)
        self.J = np.array([[-11., 1.5], [.8, 9.]])
        self.pose = np.zeros(6)
        self.origin_error = np.array([40., -25.])
        self.bias = np.array([3., -2.]) if scenario != "nominal" else np.zeros(2)
        self.frame = self.fine_frames = self.moves = 0
        self.now = 1.
        self.distance = 0.

    def truth(self):
        return np.asarray(self.cfg.target_uv)+self.origin_error+self.J@self.pose[:2]

    def measure(self, stage, hint):
        self.frame += 1
        self.now += .1
        if stage == "fine":
            self.fine_frames += 1
        if self.scenario == "dropout" and self.frame in (10, 11):
            return None
        circle = self.truth()+self.rng.normal(0., .02, 2)
        anchor = circle+self.bias
        if self.scenario == "ambiguous" and self.frame >= 5:
            selected = associate([dict(anchor=anchor, confidence=.9),
                                  dict(anchor=anchor+np.array([1., 0.]), confidence=.95)],
                                 hint, self.cfg.association_px, self.cfg.ambiguity_px)
            if selected is None:
                return None
        if self.scenario == "unstable_handoff" and stage == "fine":
            circle = circle+np.array([(-1.)**self.fine_frames*2., 0.])
        frame = self.frame-1 if self.scenario == "stale" and self.frame == 6 else self.frame
        return Observation((anchor if stage == "coarse" else circle).tolist(), anchor.tolist(),
            self.pose.tolist(), frame, self.now, stage, noise_px=.02, image_size=tuple(self.cfg.image_size))

    def move(self, target):
        target = np.asarray(target)
        self.distance += float(np.linalg.norm(target[:2]-self.pose[:2]))
        self.pose = target.copy()
        self.moves += 1


def simulate(profile="P4", scenario="feature_bias", seed=0, root="runs/gap", settings=None):
    cfg = replace(settings, profile=profile) if settings is not None else GapSettings(profile=profile)
    plant = SyntheticPlant(cfg, seed, scenario)
    log = GapLog(root, "simulation", dict(settings=asdict(cfg), seed=seed,
                 scenario=scenario, true_J=plant.J.tolist(), box_bias=plant.bias.tolist()))
    events = []
    def record(event, **data):
        events.append(dict(event=event, **data))
        log(event, **data)
    reason = None
    try:
        run_alignment(cfg, measure=plant.measure, move=plant.move,
                      get_pose=lambda: plant.pose.tolist(), clock=lambda: plant.now, record=record)
        aligned = True
    except RuntimeError as exc:
        aligned, reason = False, str(exc)
    updates = [e for e in events if e["event"] == "jacobian_update"]
    controls = [e for e in events if e["event"] == "control"]
    result = dict(source="simulation", profile=profile, scenario=scenario, seed=seed,
        aligned=aligned, reason=reason, observations=plant.frame, moves=plant.moves,
        path_mm=plant.distance, true_circle_error_px=float(np.linalg.norm(plant.truth()-cfg.target_uv)),
        updates_accepted=sum(e["accepted"] for e in updates),
        cross_feature_pairs_attempted=sum(e["cross_feature"] for e in updates),
        cross_feature_updates=sum(e["accepted"] and e["cross_feature"] for e in updates),
        max_step_mm=max((np.linalg.norm(e["dls"]["limited_delta_xy"]) for e in controls), default=0.),
        aborted_before_assembly=not aligned, log=str(log.path.resolve()))
    log("simulation_result", **{k: v for k, v in result.items() if k != "source"})
    (log.directory/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def suite(root, profiles, scenarios, seeds, settings=None):
    from datetime import datetime
    output = Path(root)/("suite_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    rows = [simulate(p, s, seed, output, settings) for s in scenarios for seed in range(seeds) for p in profiles]
    with (output/"summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(source="simulation", scenarios=list(scenarios), profiles=list(profiles),
        seeds=list(range(seeds)), count=len(rows), purpose="paired synthetic ablation; not field validation",
        groups={p: PROFILES[p] for p in profiles})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output, rows
