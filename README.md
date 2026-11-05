# Coarse-to-Fine Hole IBVS

Image-based visual servoing for aligning a robot with a selected hole in an assembly scene. Nearby holes can trade detector confidence from frame to frame, and a detection-box center can differ from the center of the physical rim. Those two errors can make a robot follow the wrong hole or settle at a biased position.

## Method

1. **Keep the selected hole.** Choose the visible hole nearest the image target at the start of coarse alignment. Reuse its track ID, and gate later measurements against a pixel prediction computed from the measured XY motion and the image Jacobian. A dropout or an outlier leaves the ID locked.
2. **Make bounded coarse moves.** Schedule the gain and step limit from the current pixel error. Reduce the step after divergence, direction reversal or detection loss, and recover the gain as the response improves.
3. **Refine the center and verify convergence.** Associate the fine measurement with the coarse target, use the fitted rim center for the fine error, and require several fresh, stable, in-tolerance observations before declaring alignment.

## Try it with synthetic observations

Use Python 3.10 or later. The tests and examples use simulated plants, mock frames or synthetic images.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python demo_servo_visualization.py
python demo_circle_visualization.py
```

The controllers take observations and measured poses as inputs. A hardware integration supplies its own camera, motion callbacks, Jacobian and validated operating limits. The [outer-rim-hole-localization project](https://github.com/KryChai/outer-rim-hole-localization) contains the image geometry behind the outer-rim measurement.

## Branches

This snapshot is based on `master`. Integrated coarse and fine controllers, focus monitoring and diagnostics.

- [`track_hole`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/track_hole) — Selected-hole tracking, adaptive coarse motion, and circle-center fine alignment.
- [`master`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/master) — Integrated coarse and fine controllers, focus monitoring and diagnostics.
- [`dev`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/dev) — Integration of adaptive step control, telemetry and circle-center fine alignment.
- [`dev-lyd`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/dev-lyd) — Focus-quality monitoring, gain scheduling, and fine alignment experiments.
- [`circle_center_fit`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/circle_center_fit) — Circle quality checks, bounded fine moves and multi-frame confirmation.
- [`codex/coarse-fine-servo`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/codex/coarse-fine-servo) — Fine alignment using the outer-rim center and image-boundary diagnostics.
- [`stepOpt_pltView`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/stepOpt_pltView) — Adaptive gain, error-dependent step limits and convergence plots.
- [`online_update_J`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/online_update_J) — Initial Jacobian estimation, gated Broyden updates and damped least squares.
- [`zhuanli_fixGapJump`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/zhuanli_fixGapJump) — Guarded feature switching and a synthetic comparison of update policies.
- [`auto_cam_level`](https://github.com/KryChai/coarse-to-fine-hole-ibvs/tree/auto_cam_level) — Monocular levelling from a complete hole ring with two feature formulations.

## 中文说明

项目针对机器人对孔时的两个问题。相邻孔的检测置信度可能交替变化，导致粗调切换目标。检测框中心也可能偏离实际孔口圆心，使精调停在有偏差的位置。

方法是首帧选孔并锁定 ID，用实际 XY 位移与图像雅可比预测下一帧位置。粗调根据像素误差调节步长和增益，精调切换到孔口拟合圆心，并用连续稳定帧确认收敛。分支中还提供特征切换、在线雅可比和相机找平实验。示例使用合成数据，便于复现控制逻辑。
