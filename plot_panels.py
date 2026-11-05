"""Shared bilingual panel renderers for live windows, composites and single plots."""

import math
import textwrap

from servo_telemetry import JOINT_FIELDS, TCP_FIELDS, TARGET_TCP_FIELDS, TARGET_JOINT_FIELDS


COLORS = ("#0072B2", "#CC79A7", "#009E73", "#D55E00", "#56B4E9", "#6C5B7B")
SERVO_PANELS = ("servo_error", "servo_convergence", "servo_correction", "servo_confidence",
                "servo_gain", "servo_step")
ROBOT_PANELS = ("tcp_x", "tcp_y", "tcp_z", "tcp_path", "tcp_orientation", "joint_positions")
PANEL_TITLES = dict(zip(SERVO_PANELS + ROBOT_PANELS, (
    "Image feature error", "Convergence (gaps = missing detection)",
    "XY command before / after limiting", "Detection quality", "Adaptive gain", "Step norm and ceiling",
    "TCP X", "TCP Y", "TCP Z",
    "Sampled TCP path (equal mm scale)", "TCP orientation", "Joint positions (x = command target)",
)))
ZH = {
    "Adaptive gain": "自适应增益", "Step norm and ceiling": "步长范数与上限",
    "Nominal gain": "名义增益", "Effective gain": "实际增益", "Gain (dimensionless)": "增益（无量纲）",
    "Requested norm": "限幅前范数", "Selected norm": "选定步长范数", "Step ceiling": "本轮步长上限",
    "Step norm (mm)": "步长范数（mm）", "No gain / step telemetry in this run": "本次运行未记录增益／步长数据",
    "Visual servo diagnostics": "视觉伺服诊断", "Robot coordinates in the base frame": "机械臂基坐标系位置",
    "Visual servo and robot overview": "视觉伺服与机械臂坐标总览",
    "SIMULATED DEMO": "模拟演示", "[DEMO]": "[模拟]",
    "Image feature error": "图像特征误差", "Convergence (gaps = missing detection)": "误差收敛（断点表示检测缺失）",
    "XY command before / after limiting": "限幅前后的 XY 修正指令", "Detection quality": "孔洞检测质量",
    "TCP X": "TCP 的 X 坐标", "TCP Y": "TCP 的 Y 坐标", "TCP Z": "TCP 的 Z 坐标",
    "Sampled TCP path (equal mm scale)": "TCP 三维采样轨迹（等毫米比例）", "TCP orientation": "TCP 姿态",
    "Joint positions (x = command target)": "关节位置（叉号为指令目标）",
    "Detection / control step": "检测／控制步", "Elapsed time (s)": "运行时间（s）",
    "Signed error (px)": "有符号误差（px）", "Error norm (px)": "误差范数（px）",
    "TCP increment (mm)": "TCP 修正量（mm）", "Confidence (0-1)": "置信度（0—1）",
    "u - cx": "横向误差 u-cx", "v - cy": "纵向误差 v-cy", "Error norm": "误差范数",
    "Threshold {value:g} px": "阈值 {value:g} px", "Threshold {value:g}": "阈值 {value:g}",
    "Detection failed": "检测失败", "Requested d{axis}": "限幅前 d{axis}", "Selected d{axis}": "选定 d{axis}",
    "Confidence": "置信度", "Actual {axis}": "实际 {axis}", "Command target": "指令目标",
    "Waiting for visual measurements": "等待视觉测量数据", "Waiting for robot feedback": "等待机械臂反馈",
    "Feedback samples": "反馈采样点", "Start": "起点", "Latest": "最新位置",
    "SDK Euler angle (deg)": "SDK 欧拉角（度）", "Joint angle (deg)": "关节角（度）",
    "Convergence": "误差收敛", "Selected correction": "选定修正量", "Feature error (px)": "特征误差（px）",
    "Increment (mm)": "修正量（mm）", "IBVS - convergence": "IBVS－视觉误差", "IBVS - robot coordinates": "IBVS－机械臂坐标",
    "Run comparison | observed single-run traces": "运行对比｜单次运行的观测曲线",
    "Single hole; no smoothing. Selected commands are not measured motion. Force / torque unavailable.":
        "单孔原始数据，未平滑；选定指令不代表实际位移；当前无力／力矩反馈。",
    "Event / feedback sampling; connecting lines do not reconstruct motion. Joint multi-turn values are preserved.":
        "事件／反馈采样；连线不重建连续运动轨迹；保留关节多圈角度。",
    "Single-hole visual measurements and sampled robot feedback; commands and actual motion are distinct.":
        "单孔视觉测量与机械臂离散反馈；修正指令与实际运动分别记录。",
    "Compare matched starting conditions / calibrations. These traces do not establish statistical superiority.":
        "应在相同初始条件和标定下比较；单次曲线不能证明统计意义上的性能优势。",
    "running": "运行中", "reported success": "任务报告完成", "failed": "失败", "interrupted": "已中断",
    "error": "异常", "unknown": "未知", "startup": "启动", "outbound": "去程", "pickup": "拾取螺丝",
    "pickup_spin": "拾取后旋转", "return_route": "回程", "screw_hole_above": "螺孔上方",
    "servo_align": "粗对准", "photo_coarse": "拍照高度粗调", "fine_approach": "精调接近", "retract": "抬升回程", "circle_fine": "圆心精调", "Phase threshold": "阶段阈值",
    "assembly_offset": "装配偏移", "assembly_z": "装配高度调整",
    "tightening": "拧紧", "release": "电批收回", "finish": "完成位置",
}


def text(key, language="en", **values):
    if language not in ("en", "zh"):
        raise ValueError(f"Unsupported plot language: {language}")
    return (ZH.get(key, key) if language == "zh" else key).format(**values)


def configure_style(language="en"):
    import matplotlib as mpl
    from matplotlib import font_manager
    families = ["DejaVu Sans", "Arial"]
    candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "WenQuanYi Micro Hei", "Arial Unicode MS")
    for family in candidates:
        try:
            font_manager.findfont(font_manager.FontProperties(family=[family]), fallback_to_default=False)
            families = [family, *families] if language == "zh" else [*families, family]
            break
        except ValueError:
            continue
    else:
        if language == "zh":
            raise RuntimeError("Chinese font unavailable. Install Microsoft YaHei or Noto Sans CJK SC, "
                               "then refresh the matplotlib font cache; English exports are retained.")
    families.append("sans-serif")
    mpl.rcParams.update({
        # Explicit families enable per-glyph fallback (a generic sans-serif
        # family alone selects one font and can lose Chinese run identifiers).
        "font.family": families, "font.sans-serif": families,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5, "axes.unicode_minus": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
    })


def columns(rows, field):
    result = []
    previous = None
    for row in rows:
        hole = row.get("hole_index", 0)
        if not math.isfinite(hole):
            hole = 0
        if previous is not None and hole != previous:
            result.append(float("nan"))
        result.append(row[field])
        previous = hole
    return result


def draw_threshold(ax, run, rows, field, language):
    config = run.metadata.get("config", {})
    coarse = config.get("align_tol_px", 2.0)
    if rows and (config.get("fine_enabled") or any(r["phase"] == "photo_coarse" for r in rows)):
        fine = config.get("fine_align_tol_px", .5)
        ax.plot(columns(rows, field),
                columns([{**r, "threshold": 5. if r["phase"] == "photo_coarse" else fine if r["phase"] == "circle_fine" else coarse} for r in rows], "threshold"),
                color="#848B94", linestyle="--", drawstyle="steps-post",
                label=text("Phase threshold", language))
    else:
        ax.axhline(coarse, color="#848B94", linestyle="--",
                   label=text("Threshold {value:g} px", language, value=coarse))


def _empty(ax, key, language):
    draw_text = ax.text2D if hasattr(ax, "text2D") else ax.text
    draw_text(0.5, 0.5, text(key, language), ha="center", va="center", transform=ax.transAxes,
              color="#606975", fontsize=10)


def finish(ax, title, xlabel, ylabel):
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, color="#E6E9ED", linewidth=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", ncol=2)
    ax.margins(x=0.04)


def header(fig, run, title, language):
    prefix = text("SIMULATED DEMO", language) + " | " if run.metadata.get("source") == "simulation" else ""
    state = text(run.metadata.get("outcome", "unknown").replace("_", " "), language)
    phase = text(run.rows[-1]["phase"] if run.rows else "startup", language)
    # Preserve the supplied run label as an identifier in both language versions.
    label = textwrap.fill(str(run.label), 70)
    fig.suptitle(f"{prefix}{text(title, language)}\n{label} | {state} | {phase}",
                 fontsize=13, fontweight="bold")


def layout(fig, footer, language):
    fig.text(0.02, 0.008, text(footer, language), fontsize=9, color="#606975")
    fig.set_layout_engine("constrained")
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.94))


def draw_panel(ax, panel, run, x_axis="step", language="en", letter="", view=None):
    """One renderer per metric: standalone and combined plots use identical data."""
    t = lambda key, **values: text(key, language, **values)
    title = (letter + "  " if letter else "") + t(PANEL_TITLES[panel])
    field = "iteration" if x_axis == "step" else "elapsed_s"
    xlabel = t("Detection / control step" if x_axis == "step" else "Elapsed time (s)")
    observations = [r for r in run.rows if r["event"] in ("vision", "detection_failed")]
    controls = [r for r in run.rows if r["event"] in ("control", "detection_failed")]
    x = columns(observations, field)
    if panel == "servo_error":
        for key, label, color in zip(("error_u_px", "error_v_px"), ("u - cx", "v - cy"), COLORS):
            ax.plot(x, columns(observations, key), color=color, label=t(label))
        ax.axhline(0, color="#848B94", linewidth=0.8, linestyle=":")
        finish(ax, title, xlabel, t("Signed error (px)"))
    elif panel == "servo_convergence":
        ax.plot(x, columns(observations, "error_norm_px"), color=COLORS[0], label=t("Error norm"))
        draw_threshold(ax, run, observations, field, language)
        failures = [r[field] for r in observations if r["event"] == "detection_failed"]
        if failures:
            ax.plot(failures, [0.03] * len(failures), "x", color=COLORS[3],
                    transform=ax.get_xaxis_transform(), label=t("Detection failed"))
        ax.set_ylim(bottom=0)
        finish(ax, title, xlabel, t("Error norm (px)"))
    elif panel == "servo_correction":
        for axis, color in zip(("x", "y"), COLORS):
            ax.plot(columns(controls, field), columns(controls, f"requested_d{axis}_mm"),
                    linestyle="--", alpha=0.65, color=color, label=t("Requested d{axis}", axis=axis))
            ax.plot(columns(controls, field), columns(controls, f"d{axis}_mm"),
                    color=color, label=t("Selected d{axis}", axis=axis))
        ax.axhline(0, color="#848B94", linewidth=0.8, linestyle=":")
        finish(ax, title, xlabel, t("TCP increment (mm)"))
    elif panel == "servo_confidence":
        ax.plot(x, columns(observations, "confidence"), color=COLORS[2], label=t("Confidence"))
        threshold = run.metadata.get("config", {}).get("conf_thresh", 0.5)
        ax.axhline(threshold, color="#848B94", linestyle="--", label=t("Threshold {value:g}", value=threshold))
        ax.set_ylim(0, 1.05)
        finish(ax, title, xlabel, t("Confidence (0-1)"))
    elif panel in ("servo_gain", "servo_step"):
        if panel == "servo_gain":
            metrics = (("nominal_gain", "Nominal gain"), ("effective_gain", "Effective gain"))
            ylabel = t("Gain (dimensionless)")
        else:
            metrics = (("requested_step_mm", "Requested norm"), ("selected_step_mm", "Selected norm"),
                       ("step_limit_mm", "Step ceiling"))
            ylabel = t("Step norm (mm)")
        for index, ((key, label), color) in enumerate(zip(metrics, COLORS)):
            ax.plot(columns(controls, field), columns(controls, key), color=color,
                    linestyle="--" if index == 0 else "-", label=t(label))
        ax.set_ylim(bottom=0)
        finish(ax, title, xlabel, ylabel)
        if not any(math.isfinite(row[metrics[0][0]]) for row in controls):
            _empty(ax, "No gain / step telemetry in this run", language)
    else:
        feedback = [r for r in run.rows if r["event"] == "feedback"]
        times = columns(feedback, "elapsed_s")
        if panel in ("tcp_x", "tcp_y", "tcp_z"):
            index = ("tcp_x", "tcp_y", "tcp_z").index(panel)
            axis = "XYZ"[index]
            targets = [r for r in run.rows if r["event"] == "tcp_target"]
            ax.plot(times, columns(feedback, TCP_FIELDS[index]), color=COLORS[index],
                    marker=".", markersize=2.5, label=t("Actual {axis}", axis=axis))
            if targets:
                ax.plot(columns(targets, "elapsed_s"), columns(targets, TARGET_TCP_FIELDS[index]),
                        "x", color="#606975", label=t("Command target"))
            finish(ax, title, t("Elapsed time (s)"), f"{axis} (mm)")
        elif panel == "tcp_orientation":
            for key, label, color in zip(TCP_FIELDS[3:], ("Rx", "Ry", "Rz"), COLORS):
                ax.plot(times, columns(feedback, key), color=color, label=label)
            finish(ax, title, t("Elapsed time (s)"), t("SDK Euler angle (deg)"))
        elif panel == "joint_positions":
            targets = [r for r in run.rows if r["event"] == "joint_target"]
            for i, (key, color) in enumerate(zip(JOINT_FIELDS, COLORS)):
                ax.plot(times, columns(feedback, key), color=color, label=f"J{i + 1}")
                if targets:
                    ax.plot(columns(targets, "elapsed_s"), columns(targets, TARGET_JOINT_FIELDS[i]),
                            "x", color=color, markersize=3, alpha=0.65)
            finish(ax, title, t("Elapsed time (s)"), t("Joint angle (deg)"))
        elif panel == "tcp_path":
            trajectory = [r for r in feedback if all(math.isfinite(r[k]) for k in TCP_FIELDS[:3])]
            if trajectory:
                x, y, z = [columns(trajectory, key) for key in TCP_FIELDS[:3]]
                ax.plot(x, y, z, ".:", color=COLORS[0], markersize=3, label=t("Feedback samples"))
                ax.scatter(x[0], y[0], z[0], color=COLORS[2], marker="o", s=35, label=t("Start"))
                ax.scatter(x[-1], y[-1], z[-1], color=COLORS[3], marker="^", s=40, label=t("Latest"))
                span = max(max(max(v) - min(v) for v in (x, y, z)), 1.0) * 1.12
                for values, setter in zip((x, y, z), (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                    mid = (max(values) + min(values)) / 2
                    setter(mid - span / 2, mid + span / 2)
                ax.set_box_aspect((1, 1, 1))
                ax.legend(loc="best", fontsize=8)
            ax.set(title=title, xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
            ax.view_init(*(view or (24, -55)))
        else:
            raise ValueError(f"Unknown plot panel: {panel}")
        if not feedback:
            _empty(ax, "Waiting for robot feedback", language)
    if panel in SERVO_PANELS[:4] and not observations:
        _empty(ax, "Waiting for visual measurements", language)


def _grid(fig, run, panels, shape, title, footer, x_axis, language, view=None):
    fig.clear()
    header(fig, run, title, language)
    grid = fig.add_gridspec(*shape)
    for index, panel in enumerate(panels):
        ax = fig.add_subplot(grid[index // shape[1], index % shape[1]],
                             projection="3d" if panel == "tcp_path" else None)
        draw_panel(ax, panel, run, x_axis, language, chr(ord("a") + index), view)
    layout(fig, footer, language)


SERVO_FOOTER = "Single hole; no smoothing. Selected commands are not measured motion. Force / torque unavailable."
ROBOT_FOOTER = "Event / feedback sampling; connecting lines do not reconstruct motion. Joint multi-turn values are preserved."
OVERVIEW_FOOTER = "Single-hole visual measurements and sampled robot feedback; commands and actual motion are distinct."
COMPARE_FOOTER = "Compare matched starting conditions / calibrations. These traces do not establish statistical superiority."


def draw_servo(fig, run, x_axis="step", language="en"):
    _grid(fig, run, SERVO_PANELS, (3, 2), "Visual servo diagnostics", SERVO_FOOTER, x_axis, language)


def draw_robot(fig, run, view=None, language="en"):
    _grid(fig, run, ROBOT_PANELS, (2, 3), "Robot coordinates in the base frame", ROBOT_FOOTER,
          "time", language, view)


def draw_overview(fig, run, x_axis="step", language="en"):
    _grid(fig, run, SERVO_PANELS + ROBOT_PANELS, (6, 2), "Visual servo and robot overview",
          OVERVIEW_FOOTER, x_axis, language)


def draw_single(fig, run, panel, x_axis="step", language="en"):
    fig.clear()
    header(fig, run, PANEL_TITLES[panel], language)
    ax = fig.add_subplot(111, projection="3d" if panel == "tcp_path" else None)
    draw_panel(ax, panel, run, x_axis, language)
    layout(fig, "SIMULATED DEMO" if run.metadata.get("source") == "simulation" else "", language)


def draw_comparison_panel(ax, run, column, x_axis="step", language="en"):
    t = lambda key, **values: text(key, language, **values)
    field = "iteration" if x_axis == "step" else "elapsed_s"
    xlabel = t("Detection / control step" if x_axis == "step" else "Elapsed time (s)")
    rows = [r for r in run.rows if r["event"] in ("vision", "detection_failed")]
    controls = [r for r in run.rows if r["event"] in ("control", "detection_failed")]
    if column == 0:
        for key, label, color in zip(("error_u_px", "error_v_px"), ("u - cx", "v - cy"), COLORS):
            ax.plot(columns(rows, field), columns(rows, key), color=color, label=t(label))
        source = t("[DEMO]") + " " if run.metadata.get("source") == "simulation" else ""
        state = t(run.metadata.get("outcome", "unknown").replace("_", " "))
        finish(ax, textwrap.fill(f"{source}{run.label} | {state}", 42), xlabel, t("Feature error (px)"))
    elif column == 1:
        ax.plot(columns(rows, field), columns(rows, "error_norm_px"), color=COLORS[0], label=t("Error norm"))
        draw_threshold(ax, run, rows, field, language)
        ax.set_ylim(bottom=0)
        finish(ax, t("Convergence"), xlabel, t("Error norm (px)"))
    else:
        for axis, color in zip(("x", "y"), COLORS):
            ax.plot(columns(controls, field), columns(controls, f"d{axis}_mm"), color=color,
                    label=t("Selected d{axis}", axis=axis))
        finish(ax, t("Selected correction"), xlabel, t("Increment (mm)"))


def draw_comparison(fig, runs, x_axis="step", language="en"):
    fig.clear()
    axes = fig.subplots(len(runs), 3, squeeze=False)
    demo = any(r.metadata.get("source") == "simulation" for r in runs)
    prefix = text("SIMULATED DEMO", language) + " | " if demo else ""
    fig.suptitle(prefix + text("Run comparison | observed single-run traces", language),
                 fontsize=13, fontweight="bold")
    for row, run in zip(axes, runs):
        for column, ax in enumerate(row):
            draw_comparison_panel(ax, run, column, x_axis, language)
    for column in range(3):
        lower = min(row[column].get_ylim()[0] for row in axes)
        upper = max(row[column].get_ylim()[1] for row in axes)
        xmax = max(row[column].get_xlim()[1] for row in axes)
        for row in axes:
            row[column].set_ylim(lower, upper)
            row[column].set_xlim(0, xmax)
    layout(fig, COMPARE_FOOTER, language)
    return axes


def save_figure(fig, prefix):
    """Keep every label inside each raster or editable vector export."""
    paths = []
    fig.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(f"{prefix}.svg", bbox_inches="tight", pad_inches=0.15)
    for extension in ("png", "pdf", "svg"):
        paths.append(f"{prefix}.{extension}")
    return paths
