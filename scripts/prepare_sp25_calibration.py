#!/usr/bin/env python3
"""生成 sp_vision_25 tiny_resnet 数字分类器的 32x32 灰度校准集。

同济 classifier.cpp 的真实输入分布：传统检测器裁出的装甲板 pattern
（局部亮区为主，含黑边），预处理为等比缩放贴左上角 + 黑底填充 + /255。
本脚本从既有 640 校准可视化图（calibration_data_640/source）合成接近
该分布的样本：

    每张源图产出 2 个样本：
      1. 整图缩放   —— 覆盖全局亮度动态范围（暗背景 ~ 强光灯条）
      2. 随机局部裁剪 —— 模拟真实 pattern 的局部视角
    处理链路与 classifier.cpp 严格一致：
      灰度 -> 等比缩放(左上角对齐, 黑底填充 32x32) -> uint8 [1,1,32,32] NCHW

输出：
    <out>/images/*.bin   校准数据本体（cal_data_dir 直接指向 images）
    <out>/preview/*.jpg  人工抽检用的可视化图
    <out>/manifest.csv   样本清单
    <out>/README.md      使用说明与 yaml 片段

用法：
    python3 scripts/prepare_sp25_calibration.py \
        --src data/horizon_x5/data/calibration_data_640/source \
        --out data/horizon_x5/data/calibration_data_32gray
"""

import argparse
import csv
import os
import shutil

import cv2
import numpy as np

TARGET = 32


def gray_letterbox32(gray, fill=0):
    """classifier.cpp 同款：等比缩放 + 左上角贴放 + 黑底填充。"""
    h, w = gray.shape[:2]
    scale = min(TARGET / w, TARGET / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((TARGET, TARGET), fill, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas


def random_crop(gray, rng):
    """随机裁剪一个局部窗，模拟传统检测器裁出的 pattern 视角。"""
    h, w = gray.shape[:2]
    for _ in range(8):  # 重试几次保证窗内非全黑
        s = rng.uniform(0.08, 0.45)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
        win = gray[y0:y0 + ch, x0:x0 + cw]
        if win.max() > 60:  # 至少含一点亮区，贴近灯条 pattern
            return win
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/data/horizon_x5/data/calibration_data_640/source")
    ap.add_argument("--out", default="/workspace/data/horizon_x5/data/calibration_data_32gray")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, "images")
    prev_dir = os.path.join(args.out, "preview")
    for d in (img_dir, prev_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    names = sorted(f for f in os.listdir(args.src) if f.lower().endswith((".jpg", ".png")))
    if not names:
        raise SystemExit(f"[abort] 源目录无图片: {args.src}")

    rows = []
    for name in names:
        img = cv2.imread(os.path.join(args.src, name))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = [("full", gray_letterbox32(gray)),
                    ("crop", gray_letterbox32(random_crop(gray, rng)))]
        for tag, g32 in variants:
            idx = len(rows)
            bin_name = f"input_{idx:04d}.bin"
            np.ascontiguousarray(g32[None, None], dtype=np.uint8).tofile(
                os.path.join(img_dir, bin_name))
            cv2.imwrite(os.path.join(prev_dir, f"{idx:04d}_{tag}.jpg"), g32)
            rows.append({"index": idx, "bin": bin_name, "src_image": name,
                         "variant": tag, "min": int(g32.min()), "max": int(g32.max())})

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# tiny_resnet 32x32 灰度校准集\n\n"
            f"- 样本数: {len(rows)}（{len(names)} 张源图 x [整图缩放 + 随机裁剪]）\n"
            "- 格式: uint8 [1,1,32,32] NCHW，0~255，BPU 内部 /255\n"
            "- 预处理对齐 sp_vision_25 classifier.cpp（等比缩放贴左上角、黑底、灰度）\n\n"
            "```yaml\ncalibration_parameters:\n"
            f"  cal_data_dir: '/workspace{img_dir}'\n"
            "  cal_data_type: 'uint8'\n```\n")

    size_kb = sum(os.path.getsize(os.path.join(img_dir, f)) for f in os.listdir(img_dir)) / 1024
    print(f"[done] 校准集: {img_dir}  ({len(rows)} 个 .bin, {size_kb:.0f} KB)")
    print(f"[done] 可视化: {prev_dir}")
    print(f"[done] 清单  : {os.path.join(args.out, 'manifest.csv')}")


if __name__ == "__main__":
    main()
