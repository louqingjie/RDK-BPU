#!/usr/bin/env python3
"""在带 YOLO 标注的数据集图上定量验证各模型的角点检测质量。

标注文件为 YOLO bbox 格式：`class cx cy w h`（归一化）。
由于真值只有框、没有角点，评估口径为：
  - 用模型输出的 4 个角点求外接框，与真值框做 IoU
  - IoU ≥ 0.5 记为命中，统计每模型的命中数 / 真值框数 / 平均 IoU

用法：
    python3 scripts/validate_on_dataset.py <图1> [图2 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_armor_detect import (Runner, build_models, decode, preprocess)  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gt(img_path):
    txt = os.path.splitext(img_path)[0] + ".txt"
    if not os.path.exists(txt):
        return []
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    out = []
    for line in open(txt, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        out.append((c, (cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    models = build_models()
    stats: dict[str, dict] = {}

    for img_path in args.images:
        img = cv2.imread(img_path)
        gt = load_gt(img_path)
        print(f"\n===== {os.path.basename(img_path)}  真值框 {len(gt)} 个: "
              + " ".join(f"cls{c}" for c, *_ in gt))
        for year, path, stem, fam in models:
            key = f"{year}_{stem}"
            st = stats.setdefault(key, {"hit": 0, "gt": 0, "ious": [], "req": 0})
            st["gt"] += len(gt)
            try:
                import onnx
                meta = {}
                if path.endswith(".onnx"):
                    m = onnx.load(path, load_external_data=False)
                    md = {p.key: p.value for p in m.metadata_props}
                    meta["names"] = eval(md["names"]) if "names" in md else None
                r = Runner(path)
                blob, s, pl, pt = preprocess(img, r.tw, r.th, nhwc=r.nhwc)
                infer = {"scale": s, "pad_l": pl, "pad_t": pt,
                         "th": r.th, "tw": r.tw, "school": stem.split("_")[0]}
                dets, note = decode(r(blob), fam, meta, infer, img.shape)
                dets = [d for d in dets if d[4] >= args.conf]
                st["req"] += 1
                for c, gx1, gy1, gx2, gy2 in gt:
                    best = max((iou(d[:4], (gx1, gy1, gx2, gy2)) for d in dets), default=0.0)
                    st["ious"].append(best)
                    if best >= 0.5:
                        st["hit"] += 1
            except Exception:  # noqa: BLE001
                continue

    rows = sorted(stats.items(), key=lambda kv: (-kv[1]["hit"], -np.mean(kv[1]["ious"] or [0])))
    print("\n\n| 模型 | 命中/真值 | 平均IoU |")
    print("|---|---|---|")
    for k, v in rows:
        if not v["ious"]:
            continue
        print(f"| `{k}` | {v['hit']}/{v['gt']} | {np.mean(v['ious']):.3f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
