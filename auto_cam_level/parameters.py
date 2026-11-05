"""自动找平统一调参入口。修改后重新启动；角度为度，位置为 mm。

不传 --config 时两个入口直接读取这里；传入 JSON 时 JSON 覆盖对应分组。
相机/工具/特征或扩框协议变化后，重新进行旋转标定。不要把示例内参用于硬件。
"""
from copy import deepcopy

# 服务端检测框总宽高扩大比例：0.10 表示宽高各乘 1.10，每侧增加 5%。
BBOX_EXPANSION_FRACTION = .10
# 感知版本同时写入日志及标定签名；旧 5% 标定文件不允许直接使用。
PERCEPTION_VERSION = "backend-ellipse-area-v2-roi10"

CAMERA = {
    "source": "example",  # 实测标定后改 hardware；示例模式禁止打开设备。
    "image_size": [2560, 1440],  # 原图 [宽, 高]，不能填预览缩放尺寸。
    "K": [[1800, 0, 1280], [0, 1780, 720], [0, 0, 1]],  # 实测内参 fx/fy/cx/cy。
    "D": [0, 0, 0, 0, 0],  # 实测 OpenCV 畸变参数。
}
ARRAY_COUNT = 8  # 完整单圈孔数，至少 6；缺孔时不继续控制。
POSE_RADIUS_MM = 40.  # 方案一孔心分布圆半径，不是单孔半径。
POSE_FEATURE = {
    "method": "pnp",  # pnp 或 homography；更换后重标旋转 J。
    "max_reprojection_px": 2.,  # 单孔最大重投影误差，px。
    "ambiguity_px": .1,  # 两解 RMS 差小于此值时检查法向歧义。
    "ambiguity_deg": 2.,  # 低残差候选法向分歧超过此角度则拒绝。
    "tolerance": [.00872664626]*2,  # 两倾斜分量容差，rad，约 0.5°。
}
SHAPE_FEATURE = {
    "enhanced": True,  # True: 4D 形状+尺寸梯度；False: 2D 协方差对照。
    "gamma": 1.,  # 尺寸梯度权重；改变后重标旋转 J。
    "max_size_residual": .05,  # 对数尺寸回归最大残差，约 5% 相对误差。
    "tolerance": [.002, .002, .0008, .0008],  # 无量纲；关闭增强时只使用前两项。
}
CONTROL = {
    "gain": .5,  # DLS 控制增益，(0,1]。
    "damping_ratio": .03,  # 阻尼相对最大奇异值的比例。
    "max_step_deg": .5,  # RX/RY 单步向量范数上限。
    "probe_deg": .3,  # 初始 J 标定时正负探测角。
    "max_excursion_deg": 12.,  # 相对起始 RX/RY 的最大偏移范数。
    "max_travel_deg": 60.,  # 标定及控制累计指令转角上限。
    "arrival_deg": .03,  # 姿态到位及固定 RZ 的允许误差。
    "position_mm": .1,  # 保持 XYZ 的位置误差上限。
    "max_age_s": 90.,  # 一次三照片观测从开始采图起的最大年龄。
    "min_singular": .01,  # 按特征容差白化后的 J 最小奇异值。
    "max_condition": 200.,  # 白化 J 最大条件数。
    "beta": .3,  # Broyden 更新比例，(0,1]。
    "stable_frames": 3,  # 连续稳定观测次数，每次观测含三张照片。
    "max_iterations": 100,  # 闭环最多观测轮数。
    "max_non_improving": 12,  # 连续未改善次数上限。
    "online": True,  # 是否在线更新 J；False 为固定 J 的 DLS 对照。
}
# 高级质量门控：建议先保留默认值，结合静态噪声和探测数据再修改。
CONTROL_GATES = {
    "noise_norm": .4,  # 白化观测噪声范数上限。
    "stable_spread": .35,  # 稳定窗口中最大白化散布。
    "return_error": .5,  # 标定回原点的白化特征差上限。
    "calibration_residual": .5,  # 标定白化拟合残差下限门限。
    "noise_sigmas": 3.,  # 特征信号与噪声区分的倍数。
    "min_probe_signal": .05,  # 白化探测响应的最小第二奇异值。
    "min_update_motion_deg": .0001,  # Broyden 最小实际运动。
    "min_update_signal": .01,  # Broyden 最小白化特征变化。
    "max_relative_update": .3,  # 每次 J 改变量相对当前 J 的上限。
    "max_reference_change": 1.,  # J 偏离标定 J 的相对上限。
    "improvement_ratio": .999,  # 新误差需小于旧误差的此比例才算改善。
    "reuse_pose_deg": 3.,  # 复用保存的旋转 J 时允许偏离参考姿态的范数。
}
VISUALIZATION = {
    "save_plots": True,  # 完成/失败后保存中英文独立数据图及拟合图。
    "live": False,  # True 打开实时误差窗口；关闭窗口不停止控制。
    "live_language": "zh",  # 实时窗口语言 zh/en，文件始终保存两版。
    "dpi": 300,  # 曲线图 PNG 分辨率。
    "image_dpi": 150,  # 拟合全图/逐孔局部图 PNG 分辨率。
    "formats": ["png", "svg", "pdf"],  # 每个数据图独立保存，可编辑矢量图。
    "figure_size": [7, 4],  # 单张数据图大小，英寸。
    "image_figure_size": [9, 6],  # 单张拟合图大小，英寸。
    "font_size": 11,  # 图标题、轴标签的基础字号，pt。
    "live_refresh_ms": 750,  # 独立实时窗口刷新间隔，毫秒。
    "chinese_fonts": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"],  # 按顺序选字体。
}
SIMULATION_SEED = 1  # 投影噪声随机种子，记录到日志便于复现。
SIMULATION_TILT_DEG = [6., -4.]  # 虚拟平面初始两个倾斜分量。
SIMULATION = {
    "array_radius_mm": 40.,  # 方案二模拟环半径；方案一使用 POSE_RADIUS_MM。
    "distance_mm": 450.,  # 虚拟相机到平面距离。
    "offset_xy_mm": [15., -10.],  # 虚拟平面在视野中的偏心量。
    "hole_radius_mm": 2.,  # 模拟孔口半径，影响表观尺寸。
    "coupling": [[.95, .18], [-.12, 1.05]],  # 模拟 RX/RY 到相机倾斜的耦合。
    "noise_px": .01,  # 合成孔心高斯噪声标准差，像素。
}
MEASUREMENT_SAMPLES = 3  # 每次停稳观测的照片数（建议至少 3）。
CAMERA_CALIBRATION = {
    "images": "photo/chessboard/*.jpg",  # 离线棋盘照片通配路径。
    "columns": 9, "rows": 6,  # 棋盘格内角点列数、行数。
    "square_mm": 20.,  # 实际每格边长 mm。
    "output": "configs/camera.local.json",  # 不覆盖已有文件。
    "max_rms_px": 1.,  # 标定全局重投影 RMS 门限，单视图允许其两倍。
}


def default_config(scheme):
    feature = deepcopy(POSE_FEATURE if scheme == "pose" else SHAPE_FEATURE)
    if scheme == "shape" and not feature["enhanced"]:
        feature["tolerance"] = feature["tolerance"][:2]
    return dict(camera=deepcopy(CAMERA), array=dict(count=ARRAY_COUNT, **(
        dict(radius_mm=POSE_RADIUS_MM) if scheme == "pose" else {})),
        feature=feature, control=deepcopy(CONTROL))
