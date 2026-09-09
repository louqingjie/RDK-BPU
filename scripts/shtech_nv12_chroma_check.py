#!/usr/bin/env python3
"""NV12 色度域偏移影响评估：host FP32 上对比 rgb 直接推理 vs rgb 经 NV12(4:2:0) 往返后推理。

原理：input_type_rt=nv12 与 rgb 的唯一差异是板端输入经过 YUV420 色度半分辨率采样
（BPU 内部 NV12->YUV444 再转 RGB）。本脚本用 OpenCV NV12 往返模拟该损失，
在 test.avi 采样帧上统计 top1 的 conf / tag / color / size 一致率与角点误差，
用于判定 NV12 输入格式是否可用（BGR2YUV_I420 与工具链同为 BT.601）。
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort

import sys
sys.path.insert(0, "/workspace/scripts")
from shtech_fp32_baseline import feed_fp32, decode, bgr_to_nv12  # noqa: E402

MODEL = "/workspace/data/onnx/shtech/SKD250526.onnx"
VIDEO = "/workspace/.tmp_shtech_auto_aim/test.avi"


def nv12_roundtrip(bgr):
    """BGR -> NV12 -> BGR（模拟 4:2:0 色度采样 + 工具链 4:4:4 还原）。"""
    nv12 = bgr_to_nv12(bgr)
    h, w = bgr.shape[:2]
    return cv2.cvtColor(nv12.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--sample", type=int, default=120)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = sorted(set(int(i // 5) * 5 for i in np.linspace(0, total - 1, args.sample).round().astype(int)))

    stat = {"n": 0, "has_det": 0, "conf_deltas": [], "pt_errs": [],
            "tag_same": 0, "color_same": 0, "size_same": 0}
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        x_rgb, _ = feed_fp32(frame)
        x_nv, _ = feed_fp32(nv12_roundtrip(frame))
        c_rgb = decode(sess.run(None, {"image": x_rgb})[0])
        c_nv = decode(sess.run(None, {"image": x_nv})[0])
        stat["n"] += 1
        if not c_rgb:
            continue
        stat["has_det"] += 1
        t, b = c_rgb[0], (c_nv[0] if c_nv else None)
        stat["conf_deltas"].append(abs(t["conf"] - b["conf"]) if b else float("nan"))
        if b:
            stat["tag_same"] += int(t["tag"] == b["tag"])
            stat["color_same"] += int(t["color"] == b["color"])
            stat["size_same"] += int(t["size"] == b["size"])
            stat["pt_errs"].append(max(np.hypot(a[0] - c[0], a[1] - c[1])
                                       for a, c in zip(t["pts"], b["pts"])))
    cap.release()

    cd = np.array(stat["conf_deltas"], np.float64)
    pe = np.array([e for e in stat["pt_errs"] if not np.isnan(e)], np.float64)
    d = stat["has_det"] or 1
    print(f"采样帧 {stat['n']}，有检出 {stat['has_det']}")
    print(f"top1 tag   一致率: {stat['tag_same']}/{d} = {stat['tag_same']/d:.1%}")
    print(f"top1 color 一致率: {stat['color_same']}/{d} = {stat['color_same']/d:.1%}")
    print(f"top1 size  一致率: {stat['size_same']}/{d} = {stat['size_same']/d:.1%}")
    print(f"|conf| 差: mean={np.nanmean(cd):.4f} p95={np.nanpercentile(cd,95):.4f} max={np.nanmax(cd):.4f}")
    if len(pe):
        print(f"top1 角点误差 px: mean={pe.mean():.3f} p95={np.percentile(pe,95):.3f} max={pe.max():.3f}")


if __name__ == "__main__":
    main()
