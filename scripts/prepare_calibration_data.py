#!/usr/bin/env python3
"""生成 RDK X5 PTQ 校准集。

处理链路（与部署/训练预处理严格一致，参考 data/AT_NN_Detector/README.md）：
    letterbox 模式：BGR 原图 -> letterbox(缩放+填充 114) -> BGR2RGB -> /255.0 -> float32 -> NCHW/NHWC -> .bin
    stretch   模式：BGR 原图 -> resize 直拉(无填充，对齐 SHtech 预处理) -> BGR2RGB -> 后同上

筛选策略（依据 docs/数据集使用规范.md）：
    1. 剔除低质量小图（min(w,h) < min_side）
    2. aHash 近似重复帧去重
    3. 按亮度分桶分层抽样，保证暗光/强光两端都有覆盖

输出：
    <out>/images/*.bin      校准数据本体（单输入可直接把 <out>/images 作为 cal_data_dir）
    <out>/source/*.jpg      预处理后的可视化图，仅用于人工检查
    <out>/manifest.csv      抽样清单（亮度、分桶、原始尺寸）
    <out>/README.md         使用说明与 yaml 片段
"""

import argparse
import csv
import os
import shutil

import cv2
import numpy as np

# 亮度分桶与配额，总和需等于 --count
BUCKETS = [
    (0, 15, 15),
    (15, 30, 20),
    (30, 50, 55),
    (50, 80, 65),
    (80, 120, 35),
    (120, 180, 8),
    (180, 256, 2),
]


def ahash(gray, size=16):
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return (g > g.mean()).astype(np.uint8).tobytes()


def letterbox(img_bgr, target_h, target_w, fill=114):
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    pad_t, pad_l = (target_h - nh) // 2, (target_w - nw) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, (pad_l, pad_t)


def stretch(img_bgr, target_h, target_w):
    """直拉 resize（无 letterbox/填充），对齐 SHtech_auto_aim 的预处理。"""
    h, w = img_bgr.shape[:2]
    canvas = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    # scale 列记录 x 向缩放系数（y 向可能不同），pad 恒为 0
    return canvas, target_w / w, (0, 0)


def collect(src, min_side):
    """读取图片 -> 尺寸过滤 -> 去重，返回 [(path, img, brightness)]。"""
    names = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    items, seen, tiny, dup = [], set(), 0, 0
    for name in names:
        img = cv2.imread(os.path.join(src, name))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < min_side:
            tiny += 1
            continue
        key = ahash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        items.append((name, img, float(img.mean())))
    print(f"[filter] 总数 {len(names)} -> 剔除小图 {tiny} -> 去重 {dup} -> 可用 {len(items)}")
    return items


def stratified_pick(items, count):
    """按亮度分桶分层抽样，桶内按文件名均匀跨步，保证场景分散。"""
    picked, report = [], []
    for lo, hi, quota in BUCKETS:
        if quota <= 0:
            continue
        pool = sorted([it for it in items if lo <= it[2] < hi], key=lambda x: x[0])
        if not pool:
            report.append((lo, hi, 0, 0))
            continue
        if len(pool) <= quota:
            sel = pool
        else:
            idx = np.linspace(0, len(pool) - 1, quota).round().astype(int)
            sel = [pool[int(i)] for i in np.unique(idx)]
        picked.extend(sel)
        report.append((lo, hi, len(pool), len(sel)))
    picked.sort(key=lambda x: x[0])
    print("[sample] 亮度分桶抽样结果（桶区间 / 池大小 / 实选）")
    for lo, hi, pool_n, sel_n in report:
        print(f"         [{lo:>3},{hi:>3})  池={pool_n:>4}  选={sel_n:>3}")
    print(f"[sample] 合计 {len(picked)} 张（目标 {count}）")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/workspace/RM2025-Armor-Public-Dataset")
    ap.add_argument("--out", default="/workspace/data/horizon_x5/data/calibration_data")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--min-side", type=int, default=360)
    ap.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    ap.add_argument("--resize-mode", choices=["letterbox", "stretch"], default="letterbox",
                    help="letterbox: 缩放+114 填充（AT_NN 等模型）；stretch: 直拉 resize（SHtech 模型，无填充）")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--dtype", choices=["float32", "uint8"], default="float32",
                    help="float32: 已归一化(/255)，配 norm_type:no_preprocess；"
                         "uint8: 原始 0~255，配 input_type_rt:rgb + data_scale(1/255)")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    src_dir = os.path.join(args.out, "source")
    for d in (img_dir, src_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    items = collect(args.src, args.min_side)
    picked = stratified_pick(items, args.count)

    resize_fn = letterbox if args.resize_mode == "letterbox" else stretch
    print(f"[resize] 模式: {args.resize_mode}")

    rows = []
    for i, (name, img, bright) in enumerate(picked):
        canvas, scale, pad = resize_fn(img, args.height, args.width)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if args.dtype == "float32":
            data = rgb.astype(np.float32) / 255.0
            if args.layout == "nchw":
                blob = np.ascontiguousarray(data.transpose(2, 0, 1)[None], dtype=np.float32)
            else:
                blob = np.ascontiguousarray(data[None], dtype=np.float32)
        else:  # uint8，原始 0~255
            if args.layout == "nchw":
                blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.uint8)
            else:
                blob = np.ascontiguousarray(rgb[None], dtype=np.uint8)
        bin_name = f"{args.input_name}_{i:04d}.bin"
        blob.tofile(os.path.join(img_dir, bin_name))
        cv2.imwrite(os.path.join(src_dir, f"{i:04d}.jpg"), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rows.append({
            "index": i,
            "bin": bin_name,
            "src_image": name,
            "orig_w": img.shape[1],
            "orig_h": img.shape[0],
            "brightness": round(bright, 2),
            "scale": round(scale, 4),
            "pad_l": pad[0],
            "pad_t": pad[1],
        })

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    size_mb = sum(os.path.getsize(os.path.join(img_dir, f)) for f in os.listdir(img_dir)) / 1e6
    print(f"\n[done] 校准集: {img_dir}  ({len(rows)} 个 .bin, {size_mb:.0f} MB)")
    print(f"[done] 可视化: {src_dir}")
    print(f"[done] 清单  : {os.path.join(args.out, 'manifest.csv')}")


if __name__ == "__main__":
    main()
