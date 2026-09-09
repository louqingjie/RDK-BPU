#!/usr/bin/env python3
"""Bayer 相机起点：NV12 vs RGB 两条部署链路的 CPU 成本与精度实测。

背景：大恒 USB 相机输出 RAW Bayer（capture.cpp 现状为 SDK 全分辨率 demosaic 出 RGB24）。
以 test.avi 阳性帧模拟 sensor，生成 RGGB Bayer 后实测三条链路：

  A  全分辨率 demosaic -> resize 640x512 -> RGB      （RGB 版模型）
  A' Bayer 2x2 块合并(域内降采样) -> 小图 demosaic -> RGB（RGB 版模型，优化）
  C  Bayer 2x2 块合并 -> 小图 RGB -> NV12 pack        （NV12 版模型）

精度基准：A 的 RGB 直接喂 FP32 ONNX；A'/C 的等效输入还原为 RGB 喂同一 FP32 ONNX
（模拟 BPU 内部 CSC 后的网络输入），对比 top1 conf / argmax / 角点。
"""

import time
import cv2
import numpy as np
import onnxruntime as ort

import sys
sys.path.insert(0, "/workspace/scripts")
from shtech_fp32_baseline import feed_fp32, decode, bgr_to_nv12  # noqa: E402

MODEL = "/workspace/data/onnx/shtech/SKD250526.onnx"
VIDEO = "/workspace/.tmp_shtech_auto_aim/test.avi"
FRAME = 890
TARGET_W, TARGET_H = 640, 512


def bench(fn, n=50):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) / n * 1000, r


def main():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME)
    ok, full = cap.read()
    cap.release()
    H, W = full.shape[:2]
    print(f"sensor 帧 {W}x{H}")

    # 模拟 RGGB Bayer：偶行偶列=R，奇行奇列=B，其余 G
    bayer = np.zeros((H, W), np.uint8)
    bayer[0::2, 0::2] = full[0::2, 0::2, 2]  # R
    bayer[0::2, 1::2] = full[0::2, 1::2, 1]  # G
    bayer[1::2, 0::2] = full[1::2, 0::2, 1]  # G
    bayer[1::2, 1::2] = full[1::2, 1::2, 0]  # B

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])

    # 链路 A：全分辨率 demosaic（capture.cpp 现状）-> resize -> RGB
    tA, rgbA = bench(lambda: cv2.resize(
        cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR), (TARGET_W, TARGET_H)))
    # 链路 A'：Bayer 2x2 块合并 -> 小图 demosaic -> RGB
    def a2():
        small = cv2.resize(bayer, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.cvtColor(small, cv2.COLOR_BayerBG2BGR), (TARGET_W, TARGET_H))
    tA2, rgbA2 = bench(a2)
    # 链路 C：Bayer 2x2 块合并 -> 小图 RGB -> NV12 pack
    def c():
        small = cv2.resize(bayer, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BayerBG2BGR), (TARGET_W, TARGET_H))
        return bgr_to_nv12(rgb)
    tC, nv12C = bench(c)
    # 链路 B：全分辨率 demosaic -> RGB -> NV12 pack（NV12 模型 + 旧采集方式）
    tB, _ = bench(lambda: bgr_to_nv12(rgbA))

    print(f"\nCPU 成本（{W}x{H} Bayer 起点，单帧 ms）:")
    print(f"  A  全分辨率demosaic->resize->RGB      : {tA:6.2f}")
    print(f"  A' 2x2块合并->小图demosaic->RGB       : {tA2:6.2f}")
    print(f"  C  2x2块合并->小图demosaic->NV12 pack : {tC:6.2f}")
    print(f"  B  全分辨率demosaic->RGB->NV12 pack   : {tB:6.2f}")

    # 精度：A 为基准，A'/C 的等效输入（NV12 还原 RGB）喂同一 FP32 模型
    def infer(rgb):
        x, _ = feed_fp32(rgb)
        return decode(sess.run(None, {"image": x})[0])

    ref = sorted(infer(rgbA), key=lambda c: -c["conf"])[0]
    rgbC = cv2.cvtColor(nv12C.reshape(TARGET_H * 3 // 2, TARGET_W), cv2.COLOR_YUV2BGR_NV12)
    for name, rgb_x in (("A'", rgbA2), ("C(NV12还原)", rgbC)):
        cands = infer(rgb_x)
        top = sorted(cands, key=lambda c: -c["conf"])[0] if cands else None
        if top:
            err = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(ref["pts"], top["pts"]))
            print(f"  {name:<12} conf={top['conf']:.4f}(基准 {ref['conf']:.4f}) "
                  f"tag/color/size一致={top['tag']==ref['tag']}/{top['color']==ref['color']}/"
                  f"{top['size']==ref['size']} 角点误差={err:.2f}px")
        else:
            print(f"  {name:<12} 无检出")


if __name__ == "__main__":
    main()
