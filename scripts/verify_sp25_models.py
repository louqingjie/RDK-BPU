#!/usr/bin/env python3
"""验证第三方仓库 IR 模型与原始 ONNX 是否同源（双引擎推理对齐）。

用途：确认从第三方 assets/ 下载的 OpenVINO IR（.xml+.bin）与其引用的
原始 ONNX 模型数值一致，从而允许直接用原始 ONNX 走量化流程（更保真）。

方法：同一随机输入分别送 OpenVINO IR（CPU）与 ONNX Runtime（CPU），
比较输出张量的余弦相似度与最大绝对误差。
阈值：余弦 > 0.999 判定同源（FP16 权重的 IR 会保留 ~1e-1 级逐点误差，
但整张量余弦仍应接近 1）。

示例：
    python3 scripts/verify_sp25_models.py \
        --ir data/onnx/sp_vision_25/yolov5.xml \
        --onnx data/onnx/sp_vision_25/rp24_0708.onnx
"""

import argparse

import numpy as np
import onnxruntime as ort
import openvino as ov


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", required=True, help="OpenVINO IR xml 路径")
    ap.add_argument("--onnx", required=True, help="原始 ONNX 路径")
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    core = ov.Core()
    ir = core.read_model(args.ir)
    shape = list(ir.inputs[0].shape)
    if any(not isinstance(d, int) or d <= 0 for d in shape):
        raise SystemExit(f"[abort] IR 输入含动态维度: {shape}，请先固化")
    rng = np.random.default_rng(args.seed)
    x = rng.random(shape, dtype=np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    y_onnx = np.asarray(sess.run(None, {iname: x})[0])

    comp = core.compile_model(ir, "CPU")
    y_ir = np.asarray(comp([ov.Tensor(x)])[comp.output(0)])

    if y_onnx.shape != y_ir.shape:
        raise SystemExit(f"[abort] 输出形状不一致: onnx {y_onnx.shape} vs ir {y_ir.shape}")

    cos = float((y_onnx * y_ir).sum() /
                (np.linalg.norm(y_onnx) * np.linalg.norm(y_ir) + 1e-12))
    mad = float(np.abs(y_onnx - y_ir).max())
    print(f"onnx out {y_onnx.shape} | ir out {y_ir.shape}")
    print(f"cosine similarity = {cos:.6f}  (threshold {args.threshold})")
    print(f"max abs diff      = {mad:.4f}")
    if cos > args.threshold:
        print(f"[PASS] 判定同源：可直接使用原始 ONNX 走量化流程")
    else:
        raise SystemExit("[FAIL] 未达阈值，两模型可能不同源或转换有损")


if __name__ == "__main__":
    main()
