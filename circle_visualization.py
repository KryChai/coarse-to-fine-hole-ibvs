"""Durable alignment telemetry and bilingual, one-axis-per-file exports.

Plotting is deferred until control has returned. Offline export needs no SDK.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class CircleRun:
    def __init__(self, root, source="hardware"):
        self.directory = Path(root)/("circle_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_{os.getpid()}")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory/"events.jsonl"
        self.started = time.monotonic()
        self.sequence = 0
        self.source = source

    def __call__(self, event, **data):
        self.sequence += 1
        row = clean(dict(event=event, sequence=self.sequence, elapsed_s=time.monotonic()-self.started,
                         source=self.source, **data))
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line+"\n")
            stream.flush()
        print("[CircleData] "+line)

    def finish(self, plots=True):
        if plots:
            try:
                export_run(self.path)
                print(f"[CircleData] 中英文独立图: {self.directory/'figures'}")
            except Exception as error:
                self("export_failed", reason=str(error))
                print(f"[CircleData] 导出失败，可使用原始日志离线重绘: {error}")


def load_events(path):
    # Reject corrupt logs rather than silently dropping failed measurements.
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_rows(events):
    rows = []
    for e in events:
        if e["event"] not in ("coarse_measurement", "fine_measurement"):
            continue
        center = e["center"]
        target = e.get("target_uv", [1280., 720.])
        quality = e.get("quality", {})
        detection = e.get("detection_center", center)
        rows.append(dict(sample=len(rows)+1, sequence=e["sequence"], elapsed_s=e["elapsed_s"],
            phase="fine" if e["event"] == "fine_measurement" else "coarse", iteration=e["iteration"],
            u=center[0], v=center[1], error_u=center[0]-target[0], error_v=center[1]-target[1],
            error_px=e["error_px"], confidence=e.get("confidence"), radius_px=e.get("radius_px"),
            center_shift_px=float(np.linalg.norm(np.array(center)-detection)),
            **{key: quality.get(key) for key in ("rmse_px", "angular_coverage", "inlier_fraction",
                "bbox_jitter_max_center_shift_px", "scale_center_shift_px", "candidate_center_spread_px")}))
    return rows


def export_run(log_path, output=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import Circle, Rectangle
    from matplotlib.lines import Line2D
    import cv2

    log_path = Path(log_path).resolve()
    raw_log = log_path.read_bytes()
    events = [json.loads(line) for line in raw_log.decode("utf-8").splitlines() if line.strip()]
    rows = sample_rows(events)
    out = Path(output) if output else log_path.parent/"figures"
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sample", "phase", "iteration", "error_px"]
    with (out/"samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Exact input snapshot supports auditing which events each export used.
    (out/"events_snapshot.jsonl").write_bytes(raw_log)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chinese = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if name in fonts), None)
    if not chinese:
        raise RuntimeError("缺少中文字体，请安装 Microsoft YaHei、SimHei 或 Noto Sans CJK SC 后离线重绘")
    manifest = {"log": str(log_path), "sha256": hashlib.sha256(raw_log).hexdigest(),
                "source": sorted({e.get("source", "unspecified") for e in events}),
                "status": next((e["event"] for e in reversed(events)
                                if e["event"] in ("alignment_complete", "alignment_failed")), "incomplete"),
                "files": [], "skipped_overlays": [], "chinese_font": chinese}
    measurements = [("error_px", "圆心误差", "Centre error", "px"),
        ("error_u", "水平像素误差", "Horizontal error", "px"),
        ("error_v", "垂直像素误差", "Vertical error", "px"),
        ("confidence", "检测置信度", "Detection confidence", ""),
        ("center_shift_px", "拟合圆心与框中心偏移", "Fitted vs box centre offset", "px"),
        ("radius_px", "外圈拟合半径", "Outer circle radius", "px"),
        ("rmse_px", "拟合残差", "Fit residual RMS", "px"),
        ("angular_coverage", "角度覆盖比例", "Angular coverage", ""),
        ("inlier_fraction", "边缘内点比例", "Edge inlier fraction", ""),
        ("bbox_jitter_max_center_shift_px", "框扰动圆心变化", "Box perturbation centre shift", "px"),
        ("scale_center_shift_px", "平滑尺度圆心变化", "Smoothing scale centre shift", "px"),
        ("candidate_center_spread_px", "候选圆心分歧", "Candidate centre spread", "px")]
    simulation = any(e.get("source") == "simulation" for e in events)
    for lang in ("zh", "en"):
        folder = out/lang
        folder.mkdir(exist_ok=True)
        with plt.rc_context({"font.family": chinese if lang == "zh" else "DejaVu Sans", "axes.unicode_minus": False}):
            def save(fig, name, title, image=False):
                fig.axes[0].set_title(title+("（模拟数据）" if lang == "zh" else " (SIMULATED DATA)") if simulation else title)
                fig.tight_layout()
                for ext in (("png",) if image else ("png", "pdf", "svg")):
                    path = folder/(name+"."+ext)
                    fig.savefig(path, dpi=150 if image else 300)
                    manifest["files"].append(str(path.relative_to(out)))
                plt.close(fig)

            for key, zh, en, unit in measurements:
                fig, ax = plt.subplots(figsize=(7, 4))
                for phase in ("coarse", "fine"):
                    samples = [r for r in rows if r["phase"] == phase]
                    label = ({"coarse": "粗对准", "fine": "圆心精调"}[phase] if lang == "zh" else phase)
                    ax.plot([r["sample"] for r in samples], [np.nan if r[key] is None else r[key] for r in samples],
                            ".-", label=label)
                if key == "error_px":
                    config = next((e["settings"] for e in events if e["event"] == "fine_start"), {})
                    ax.axhline(config.get("tolerance_px", 1.), color="red", linestyle="--",
                               label="精调阈值" if lang == "zh" else "Fine threshold")
                ax.set_xlabel("观测序号" if lang == "zh" else "Observation index")
                ax.set_ylabel((zh if lang == "zh" else en)+(f" ({unit})" if unit else ""))
                ax.grid(alpha=.25)
                ax.legend()
                save(fig, key, zh if lang == "zh" else en)

            for kind in ("step", "tcp", "uv", "outcomes"):
                fig, ax = plt.subplots(figsize=(7, 4))
                motions = [e for e in events if e["event"] == "fine_motion_complete"]
                if kind == "step":
                    for field, zh, en in (("commanded", "指令位移", "Commanded travel"), ("actual", "实际位移", "Actual travel")):
                        ax.plot(range(1, len(motions)+1), [np.linalg.norm(e[field]) for e in motions], ".-", label=zh if lang == "zh" else en)
                    ax.set(xlabel="运动序号" if lang == "zh" else "Motion index", ylabel="mm")
                    ax.legend()
                elif kind == "tcp":
                    for i, axis in enumerate(("X", "Y")):
                        ax.plot(range(1, len(motions)+1), [e["tcp"][i] for e in motions], ".-", label=axis)
                    ax.set(xlabel="运动序号" if lang == "zh" else "Motion index", ylabel="TCP (mm)")
                    ax.legend()
                elif kind == "uv":
                    for phase in ("coarse", "fine"):
                        data = [r for r in rows if r["phase"] == phase]
                        label = {"coarse": "粗对准", "fine": "圆心精调"}[phase] if lang == "zh" else phase
                        ax.plot([r["u"] for r in data], [r["v"] for r in data], ".-", label=label)
                    ax.invert_yaxis()
                    ax.set(xlabel="u (px)", ylabel="v (px)")
                    ax.legend()
                else:
                    counts = {name: sum(e["event"] == name for e in events) for name in
                              ("fine_measurement", "fine_rejected", "alignment_failed", "alignment_complete")}
                    labels = ["接受", "测量拒绝", "任务失败", "任务完成"] if lang == "zh" else ["Accepted", "Rejected", "Failed", "Complete"]
                    ax.bar(labels, list(counts.values()))
                    ax.set_ylabel("次数" if lang == "zh" else "Count")
                ax.grid(alpha=.25)
                titles = {"step": ("精调指令与实际位移", "Fine commanded and actual travel"),
                          "tcp": ("精调实际 TCP", "Fine actual TCP"), "uv": ("图像圆心轨迹", "Image centre trajectory"),
                          "outcomes": ("测量与任务状态", "Measurement and run outcomes")}
                save(fig, kind, titles[kind][0 if lang == "zh" else 1])

            for e in events:
                if e["event"] != "circle_response":
                    continue
                path = Path(e["image"])
                image = cv2.imread(str(path))
                if image is None:
                    manifest["skipped_overlays"].append(str(path))
                    continue
                payload = e.get("response")
                data = payload.get("data") if isinstance(payload, dict) else None
                holes = data.get("holes") if isinstance(data, dict) else None
                if not isinstance(holes, list) or any(not isinstance(h, dict) for h in holes):
                    manifest["skipped_overlays"].append(str(path)+": malformed response")
                    continue
                for crop in (False, True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    for h in holes:
                        color = "lime" if h.get("status") == "ok" else "orange"
                        b = h.get("detection_bbox")
                        if not b:
                            continue
                        ax.add_patch(Rectangle((b["x1"], b["y1"]), b["x2"]-b["x1"], b["y2"]-b["y1"], fill=False, edgecolor="magenta"))
                        detection = h.get("detection_center")
                        if detection:
                            ax.plot(detection["x"], detection["y"], "+", color="magenta")
                        c = h.get("circle")
                        if c:
                            ax.add_patch(Circle((c["center_x"], c["center_y"]), c["radius"], fill=False, edgecolor=color))
                            ax.plot(c["center_x"], c["center_y"], "+", color="red")
                        edges = h.get("visualization") or {}
                        pts = np.asarray(edges.get("edge_points_px", []))
                        if pts.ndim == 2 and pts.shape[1] == 2:
                            ax.scatter(pts[:, 0], pts[:, 1], s=2, color="cyan")
                        status = {"ok": "通过", "review": "需复核", "failed": "失败"}.get(h.get("status"), "未知") if lang == "zh" else h.get("status")
                        ax.text(b["x1"], b["y1"], f"#{h.get('detection_index')} {status}", color=color)
                    target = e.get("target_uv", [image.shape[1]/2, image.shape[0]/2])
                    ax.plot(*target, "x", color="blue")
                    if crop and holes:
                        hint = np.array(e["expected_detection"])
                        nearest = min(holes, key=lambda h: np.linalg.norm(np.array([h["detection_center"]["x"], h["detection_center"]["y"]])-hint))
                        b = nearest["detection_bbox"]
                        pad = max(b["x2"]-b["x1"], b["y2"]-b["y1"])*.15
                        ax.set_xlim(max(0, b["x1"]-pad), min(image.shape[1], b["x2"]+pad))
                        ax.set_ylim(min(image.shape[0], b["y2"]+pad), max(0, b["y1"]-pad))
                    ax.set(xlabel="u (px)", ylabel="v (px)")
                    names = (["拟合圆（通过/复核）", "拟合圆心", "检测框及框中心", "拟合边缘点", "图像目标"] if lang == "zh"
                             else ["Circle (accepted/review)", "Fitted centre", "Box and box centre", "Fitted edge points", "Image target"])
                    handles = [Line2D([], [], color="lime", label=names[0]),
                               Line2D([], [], marker="+", linestyle="", color="red", label=names[1]),
                               Line2D([], [], marker="+", color="magenta", label=names[2]),
                               Line2D([], [], marker=".", linestyle="", color="cyan", label=names[3]),
                               Line2D([], [], marker="x", linestyle="", color="blue", label=names[4])]
                    ax.legend(handles=handles, fontsize=8, loc="upper right")
                    title = "外圈拟合：圆、边缘点与检测框" if lang == "zh" else "Outer circle, fitted edge points and detection boxes"
                    save(fig, f"fit_{e['sequence']:04d}"+("_roi" if crop else ""), title, image=True)
    (out/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out/"manifest.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从圆心对准日志导出中英文独立图，无需硬件")
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(export_run(args.log, args.output))
