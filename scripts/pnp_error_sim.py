#!/usr/bin/env python3
"""量化角点误差 -> PnP 解算误差的蒙特卡洛仿真。

用途：回答"INT8 量化后，角点误差经过 PnP 解算会造成多大的距离/角度偏差"。

链路：
    模型输出角点(letterbox 576x768 空间)
      -> INT8/INT16 量化误差（步长/2 的均匀分布）
      -> 映射回原图（除以 letterbox 缩放系数）
      -> cv2.solvePnP 解算位姿
      -> 统计距离 / yaw / pitch 误差

用法：
    python3 scripts/pnp_error_sim.py                 # 默认：小装甲板 INT8
    python3 scripts/pnp_error_sim.py --armor large
    python3 scripts/pnp_error_sim.py --step 0.0257   # INT16 步长
"""

import argparse

import cv2
import numpy as np

# 相机内参：来自 Galaxy_camera_node/rm_bringup/config/camera_info.yaml
# 该文件标注分辨率 1280x1024，fx=1807.12。本仿真按实际使用分辨率 1440x1080
# 等比缩放 fx/fy（视场角保持一致），误差结论在角度意义上与分辨率无关。
FX_1280, FY_1024 = 1807.12121, 1806.46896
IMG_W, IMG_H = 1440, 1080
FX = FX_1280 * IMG_W / 1280.0
FY = FY_1024 * IMG_H / 1024.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# 模型输入 576x768；1440x1080 等比缩放，无 padding
LETTERBOX_SCALE = min(768 / IMG_W, 576 / IMG_H)

# 装甲板尺寸（米）。RM 常用值，若与实际不符，误差近似按 1/尺寸 缩放
ARMORS = {
    "small": (0.135, 0.055),
    "large": (0.230, 0.127),
}


def object_points(w, h):
    """装甲板局部坐标系下 4 个角点，顺序：左上、右上、右下、左下。"""
    return np.array([[-w / 2, -h / 2, 0.0],
                     [w / 2, -h / 2, 0.0],
                     [w / 2, h / 2, 0.0],
                     [-w / 2, h / 2, 0.0]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def project(pts_cam):
    """相机坐标系点 -> 像素坐标。"""
    u = FX * pts_cam[:, 0] / pts_cam[:, 2] + CX
    v = FY * pts_cam[:, 1] / pts_cam[:, 2] + CY
    return np.stack([u, v], axis=1)


def run(armor, step_letterbox, trials=400, seed=0):
    rng = np.random.default_rng(seed)
    w, h = ARMORS[armor]
    obj = object_points(w, h)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)

    # letterbox 空间步长 -> 原图像素步长；均匀分布 U(-step/2, step/2)
    step_orig = step_letterbox / LETTERBOX_SCALE
    sigma = step_orig / np.sqrt(12.0)  # 等效标准差

    print(f"装甲板: {armor}  {w*1000:.0f} x {h*1000:.0f} mm")
    print(f"letterbox 缩放系数: {LETTERBOX_SCALE:.4f}（576x768 <- {IMG_W}x{IMG_H}）")
    print(f"量化步长: letterbox {step_letterbox:.4f} px -> 原图 {step_orig:.3f} px"
          f"（±{step_orig/2:.3f} px，等效 σ={sigma:.3f} px）")
    print()
    header = (f"{'距离':>6}{'偏航角':>8}{'瞄准角误差':>11}{'距离误差':>12}{'相对误差':>10}"
              f"{'yaw误差':>10}{'pitch误差':>11}{'脱靶量':>11}{'命中率':>8}")
    print(header)
    print("-" * len(header))

    for z in (1.0, 2.0, 3.0, 5.0, 7.0):
        for yaw_deg in (0.0, 15.0, 30.0, 45.0):
            yaw = np.radians(yaw_deg)
            R = rot_y(yaw)
            t = np.array([0.0, 0.0, z])
            pts_cam = (R @ obj.T).T + t
            ideal = project(pts_cam)

            # 真实装甲板中心在原图上的投影（云台应指向的方向）
            c_true = project(t.reshape(1, 3))[0]

            dz, dyaw, dpitch, miss_x, miss_y = [], [], [], [], []
            for _ in range(trials):
                noise = rng.uniform(-step_orig / 2, step_orig / 2, size=ideal.shape)
                noisy = ideal + noise
                ok, rvec, tvec = cv2.solvePnP(
                    obj, noisy.astype(np.float64), K, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    continue
                Rm, _ = cv2.Rodrigues(rvec)
                n = Rm @ np.array([0.0, 0.0, 1.0])      # 装甲板法向
                yaw_hat = np.arctan2(n[0], n[2])
                pitch_hat = np.arctan2(n[1], np.hypot(n[0], n[2]))
                dz.append(np.linalg.norm(tvec) - z)
                dyaw.append(np.degrees(yaw_hat - yaw))
                dpitch.append(np.degrees(pitch_hat))

                # 瞄准点误差：解算出的中心 相对 真实中心 的角度偏差
                c_est = project(tvec.reshape(1, 3))[0]
                ang_x = (c_est[0] - c_true[0]) / FX
                ang_y = (c_est[1] - c_true[1]) / FY
                miss_x.append(ang_x * z)                # 目标距离处的横向脱靶（米）
                miss_y.append(ang_y * z)                # 纵向脱靶（米）

            dz = np.array(dz)
            dyaw = np.array(dyaw)
            dpitch = np.array(dpitch)
            miss_x = np.array(miss_x)
            miss_y = np.array(miss_y)
            # 命中判据：脱靶量落在装甲板范围内
            hit = float(np.mean((np.abs(miss_x) < w / 2) & (np.abs(miss_y) < h / 2)))
            aim_ang = np.degrees(np.hypot(miss_x, miss_y).mean() / z)
            print(f"{z:>5.1f}m{yaw_deg:>7.0f}°"
                  f"{aim_ang:>11.4f}°"
                  f"{np.sqrt((dz**2).mean()):>10.3f} m"
                  f"{np.sqrt((dz**2).mean())/z*100:>9.2f}%"
                  f"{np.sqrt((dyaw**2).mean()):>9.3f}°"
                  f"{np.sqrt((dpitch**2).mean()):>10.3f}°"
                  f"{np.hypot(miss_x, miss_y).mean()*1000:>9.1f} mm"
                  f"{hit*100:>7.1f}%")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--armor", choices=list(ARMORS), default="small")
    ap.add_argument("--step", type=float, default=6.601,
                    help="letterbox 空间的量化步长（默认 6.601 = INT8 实测值）")
    ap.add_argument("--trials", type=int, default=400)
    args = ap.parse_args()
    run(args.armor, args.step, args.trials)
