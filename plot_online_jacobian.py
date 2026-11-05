"""Export bilingual, single-axis figures from complete Jacobian JSONL audit logs.

No robot SDK imports. Data are not smoothed, interpolated or downsampled.
Candidate-fit residuals use the same sample and are not validation errors.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


REASONS = {
    "updated": ("接受更新", "Accepted"), "motion_too_small": ("位移过小", "Motion too small"),
    "visual_change_too_small": ("像素变化不足", "Pixel change too small"),
    "residual_outlier": ("残差异常", "Residual outlier"), "reference_drift": ("累计漂移超限", "Reference drift"),
    "update_too_large": ("单次更新超限", "Update too large"),
    "invalid_sample_or_matrix": ("样本或矩阵无效", "Invalid sample or matrix"),
    "robot_not_settled": ("机器人未停稳", "Robot moving"), "height_changed": ("高度变化", "Height changed"),
    "orientation_changed": ("姿态变化", "Orientation changed"), "invalid_detection": ("检测无效", "Invalid detection"),
    "resolution_changed": ("分辨率变化", "Resolution changed"), "stale_image": ("图像过期", "Stale image"),
    "unpaired_frames": ("帧配对无效", "Unpaired frames"), "invalid_noise": ("噪声无效", "Invalid noise"),
    "unexpected_motion": ("实际位移异常", "Unexpected motion"),
    "command_feedback_mismatch": ("指令与反馈不符", "Command-feedback mismatch"),
    "jacobian_sign_flip": ("矩阵符号翻转", "Determinant sign flip"),
    "axis_direction_flip": ("坐标轴方向翻转", "Axis direction flip"),
}


def read_events(path):
    raw = Path(path).read_bytes()
    lines = raw.splitlines(keepends=True)
    events, warnings = [], []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                warnings.append("Incomplete final line skipped; rerender after the writer flushes.")
                continue
            raise ValueError(f"Invalid JSONL record at line {i + 1}")
        if not isinstance(event, dict) or "event" not in event:
            raise ValueError(f"Invalid event at line {i + 1}")
        events.append(event)
    return events, raw, warnings


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def component(value, index):
    try:
        for i in index:
            value = value[i]
        return number(value)
    except (TypeError, KeyError, IndexError):
        return np.nan


def make_specs(events):
    """Each specification is one independent figure, never a subplot panel."""
    obs = [e for e in events if e["event"] == "observation"]
    updates = [e for e in events if e["event"] == "jacobian_update"]
    feedback = [e for e in events if e["event"] == "motion_feedback"]
    initial = next((e.get("initial_J") for e in events if e["event"] == "servo_start"), None)
    if initial is None:
        initial = next((e.get("J") for e in events if e["event"] == "calibration_saved"), None)
    specs = []
    def steps(rows):
        return [e.get("iteration", i) + 1 for i, e in enumerate(rows)]
    def add(name, title, ylabel, series, kind="line", **extra):
        specs.append(dict(name=name, title=title, ylabel=ylabel, series=series, kind=kind, **extra))
    def series(rows, key, label):
        return (steps(rows), [number(e.get(key)) for e in rows], label)
    for row in range(2):
        for col in range(2):
            name = f"J{row+1}{col+1}"
            x = ([0] if initial is not None else []) + steps(obs)
            y = ([component(initial, [row, col])] if initial is not None else []) + [component(e.get("J"), [row, col]) for e in obs]
            candidate = [component(e.get("J_candidate"), [row, col]) for e in updates]
            add(name, (f"雅可比元素 {name} 的变化", f"Evolution of {name}"), ("雅可比 (px/mm)", "Jacobian (px/mm)"),
                [(x, y, ("实际采用值", "Applied")), (steps(updates), candidate, ("候选值", "Candidate"))])
    add("error_px", ("图像误差收敛过程", "Image error convergence"), ("误差范数 (px)", "Error norm (px)"),
        [(steps(obs), [float(np.linalg.norm(e["error_px"])) for e in obs], ("图像误差", "Image error"))])
    add("error_mm", ("模型估计的剩余修正量", "Model-estimated remaining correction"), ("估计修正量 (mm)", "Estimated correction (mm)"),
        [series(obs, "error_mm", ("模型估计值", "Model estimate"))])
    add("residual", ("同一样本上的更新前后残差", "Before/after residual on the same sample"), ("残差范数 (px)", "Residual norm (px)"),
        [series(updates, "residual_norm", ("更新前", "Before")), series(updates, "residual_after_norm", ("实际采用矩阵之后", "After applied matrix"))])
    for axis, label in enumerate(("u", "v")):
        add(f"prediction_{label}", (f"{label} 方向像素变化预测", f"Pixel-change prediction in {label}"), ("像素变化 (px)", "Pixel change (px)"),
            [(steps(updates), [component(e.get(key), [axis]) for e in updates], names)
             for key, names in (("ds", ("实际变化", "Observed")), ("predicted_ds_before", ("更新前预测", "Before prediction")),
                                ("predicted_ds_after", ("实际采用矩阵预测", "Applied prediction")))])
    for axis, label in enumerate(("x", "y")):
        actual, commanded = [], []
        for e in feedback:
            before = component(e.get("before", {}).get("tcp"), [axis])
            actual.append(component(e.get("after", {}).get("tcp"), [axis]) - before)
            commanded.append(component(e.get("commanded_tcp"), [axis]) - before)
        add(f"motion_{label}", (f"{label.upper()} 方向指令与实际位移", f"Commanded and actual {label.upper()} displacement"), ("位移 (mm)", "Displacement (mm)"),
            [(steps(feedback), commanded, ("指令位移", "Commanded")), (steps(feedback), actual, ("实际位移", "Actual"))])
    add("condition", ("矩阵条件数", "Matrix condition number"), ("条件数", "Condition number"),
        [series(updates, "condition", ("实际采用矩阵", "Applied matrix")), series(updates, "condition_candidate", ("候选矩阵", "Candidate matrix"))])
    add("singular_values", ("实际采用矩阵的奇异值", "Singular values of the applied matrix"), ("奇异值 (px/mm)", "Singular value (px/mm)"),
        [(steps(updates), [component(e.get("singular_values"), [i]) for e in updates], (f"奇异值 {i+1}", f"Singular value {i+1}")) for i in range(2)])
    add("weight", ("残差鲁棒权重", "Robust residual weight"), ("权重", "Weight"), [series(updates, "weight", ("Huber 权重", "Huber weight"))])
    add("relative_update", ("单次候选与实际更新幅度", "Candidate and applied update magnitude"), ("相对矩阵范数", "Relative matrix norm"),
        [series(updates, "candidate_relative_update", ("候选更新", "Candidate")), series(updates, "relative_update", ("实际更新", "Applied"))])
    add("reference_drift", ("相对初始标定的累计漂移", "Drift from the initial calibration"), ("相对矩阵范数", "Relative matrix norm"),
        [series(updates, "reference_drift", ("实际采用矩阵", "Applied matrix"))])
    add("update_gate", ("逐次更新判定", "Update decisions"), ("拒绝 / 接受", "Rejected / Accepted"),
        [(steps(updates), [1 if e.get("accepted") else 0 for e in updates], ("更新判定", "Decision"))], kind="gate")
    reasons = list(dict.fromkeys(e.get("reason", "unknown") for e in updates))
    add("update_reasons", ("更新接受与拒绝原因", "Update acceptance and rejection reasons"), ("次数", "Count"), [], kind="reasons",
        reasons=reasons, counts=[sum(e.get("reason") == r for e in updates) for r in reasons])
    target = next((e.get("settings", {}).get("target_uv") for e in events if e["event"] in ("servo_start", "run_start")), None)
    add("uv_trajectory", ("孔中心在图像中的轨迹", "Hole-center trajectory in the image"), ("v (px)", "v (px)"),
        [([component(e.get("sample", {}).get("center"), [0]) for e in obs],
          [component(e.get("sample", {}).get("center"), [1]) for e in obs], ("孔中心", "Hole center"))], kind="trajectory", target=target)
    return specs


def export_figures(log_path, output_dir=None, formats=("png", "pdf", "svg"), dpi=300):
    events, raw, warnings = read_events(log_path)
    output = Path(output_dir) if output_dir else Path(log_path).with_suffix("") / "figures"
    output.mkdir(parents=True, exist_ok=True)
    # Keep font cache writable in the run folder on managed Windows workstations.
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts = {font.name for font in font_manager.fontManager.ttflist}
    chinese = next((f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei") if f in fonts), None)
    if chinese is None:
        raise RuntimeError("中文字体缺失，请安装 Microsoft YaHei / Noto Sans CJK SC 后重绘")
    if not set(formats) <= {"png", "pdf", "svg"} or not formats:
        raise ValueError("formats must contain png, pdf or svg")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive")
    specs = make_specs(events)
    source = next((e.get("source") for e in events if e.get("source")), "unspecified")
    aborted = any(e["event"] == "aborted" for e in events)
    complete = any(e["event"] in ("aligned_frozen", "calibration_saved") for e in events)
    status = "aborted" if aborted else "complete" if complete else "in_progress"
    manifest = {"source_log": str(Path(log_path).resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source": source, "status": status, "event_count": len(events), "warnings": warnings,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": [],
                "notes": ["Each file is an independent single-axis figure; no composites.",
                          "Residuals before/after use the same learning sample, not held-out validation.",
                          "Missing diagnostics remain gaps; all valid points retained without smoothing."]}
    # Exact audit rows, including rejected attempts and per-frame measurements.
    snapshot = output / "events_snapshot.jsonl"
    snapshot.write_bytes(b"".join((json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for e in events))
    manifest["files"].append(str(snapshot.relative_to(output)))
    updates = [e for e in events if e["event"] == "jacobian_update"]
    with (output / "updates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted(set().union(*(e.keys() for e in updates))) if updates else ["iteration", "accepted", "reason"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for e in updates:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in e.items()})
    manifest["files"].append("updates.csv")
    for language, index in (("zh", 0), ("en", 1)):
        dest = output / language
        dest.mkdir(exist_ok=True)
        with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": [chinese if index == 0 else "DejaVu Sans"],
                             "font.size": 10, "axes.titlesize": 12, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.unicode_minus": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False}):
            for spec in specs:
                fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
                try:
                    have_data = False
                    for n, (xs, ys, labels) in enumerate(spec["series"]):
                        if any(np.isfinite(y) for y in ys):
                            have_data = True
                        ax.plot(xs, ys, marker="o" if n == 0 else "x", markersize=3.5,
                                linewidth=1.5, color=("#176B9B", "#C56C39", "#348B78")[n % 3],
                                linestyle="-" if n == 0 else "--", label=labels[index])
                    if spec["kind"] == "reasons" and spec["reasons"]:
                        have_data = True
                        labels = [REASONS.get(r, (r, r))[index] for r in spec["reasons"]]
                        ax.barh(labels, spec["counts"], color="#176B9B")
                        ax.set_xlabel(spec["ylabel"][index])
                        ax.set_ylabel( ("判定原因", "Decision reason")[index])
                    else:
                        ax.set_xlabel(("伺服迭代次数", "Servo iteration")[index])
                        ax.set_ylabel(spec["ylabel"][index])
                    if spec["kind"] == "trajectory":
                        ax.set_xlabel("u (px)")
                        ax.invert_yaxis()
                        ax.set_aspect("equal", adjustable="datalim")
                        if spec.get("target") is not None:
                            ax.scatter(*spec["target"], marker="*", s=100, color="#C56C39", label=("目标点", "Target")[index])
                    if spec["kind"] == "gate":
                        ax.set_yticks([0, 1], [("拒绝", "接受"), ("Rejected", "Accepted")][index])
                        ax.set_ylim(-.15, 1.15)
                    if spec["name"] == "weight":
                        ax.set_ylim(0, 1.05)
                    ax.set_title(spec["title"][index])
                    ax.grid(axis="both", alpha=.16)
                    if have_data and spec["series"]:
                        ax.legend(loc="best", fontsize=9)
                    if not have_data:
                        ax.text(.5, .5, ("暂无有效数据", "No valid data available")[index], transform=ax.transAxes, ha="center")
                    badge = (("硬件记录" if source == "hardware" else "模拟数据" if source == "simulation" else "来源未指定"),
                             ("Hardware record" if source == "hardware" else "SIMULATED DATA" if source == "simulation" else "Unspecified source"))[index]
                    badge += " · " + {"complete": ("已完成", "Completed"), "aborted": ("已中止", "Aborted"), "in_progress": ("进行中", "In progress")}[status][index]
                    fig.text(.01, .005, badge, fontsize=8, color="#555555")
                    for fmt in formats:
                        path = dest / (spec["name"] + "." + fmt)
                        if fmt == "png":
                            fig.savefig(dest / (spec["name"] + ".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
                        elif fmt == "pdf":
                            fig.savefig(dest / (spec["name"] + ".pdf"), bbox_inches="tight", facecolor="white")
                        else:
                            fig.savefig(dest / (spec["name"] + ".svg"), bbox_inches="tight", facecolor="white")
                        manifest["files"].append(str(path.relative_to(output)))
                finally:
                    plt.close(fig)
    result = output / "manifest.json"
    result.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true", help="Refresh exported independent images while the JSONL log grows")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval <= 0 or not np.isfinite(args.interval):
        parser.error("interval must be positive")
    last_size = None
    try:
        while True:
            size = args.log.stat().st_size
            if size != last_size:
                print(export_figures(args.log, args.output))
                last_size = size
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
