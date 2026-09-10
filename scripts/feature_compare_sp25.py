#!/usr/bin/env python3
"""颈部特征对比（尺度无关）：FP32 vs 量化仿真 int16 特征，逐通道相关系数。"""

import glob

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from hmct.executor.session import InferenceSession as HzSession

FP32 = "/workspace/data/onnx/sp_vision_25/rp24_0708.onnx"
QUANT = "/workspace/quant/sp25_yolov5_all_int16_rebuild/model_output/sp25_yolov5_640x640_all_int16_quantized_model.onnx"
CALIB_SRC = "/workspace/data/horizon_x5/data/calibration_data_640_u8/source"
FEAT_FP32 = "/m/model.19/cv3/act/Mul_output_0"
FEAT_QUANT = "/m/model.19/cv3/act/Mul_output_0_quantized"


def expose(m, tensor, dtype=TensorProto.FLOAT):
    g = m.graph
    if any(o.name == tensor for o in g.output):
        return
    vi = next((v for v in g.value_info if v.name == tensor), None)
    shape = [d.dim_value for d in vi.type.tensor_type.shape.dim] if vi is not None else []
    g.output.append(helper.make_tensor_value_info(tensor, dtype, shape))


def main():
    m32 = onnx.load(FP32)
    expose(m32, FEAT_FP32)
    onnx.save(m32, "/tmp/fp32_feat.onnx")
    s32 = ort.InferenceSession("/tmp/fp32_feat.onnx", providers=["CPUExecutionProvider"])

    mq = onnx.load(QUANT)
    expose(mq, FEAT_QUANT, TensorProto.INT16)
    onnx.save(mq, "/tmp/quant_feat.onnx")
    sq = HzSession("/tmp/quant_feat.onnx")

    src = sorted(glob.glob(f"{CALIB_SRC}/*.jpg"))
    f = [x for x in src if "0173" in x][0]
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    x = (cv2.resize(img, (640, 640)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    u8_nchw = np.ascontiguousarray(
        (cv2.resize(img, (640, 640)).transpose(2, 0, 1))[None], dtype=np.uint8)

    o32_all = s32.run(None, {s32.get_inputs()[0].name: x})
    o32 = dict(zip([o.name for o in s32.get_outputs()], o32_all))[FEAT_FP32]  # [1,56,80,80]
    iq = sq.get_inputs()[0]
    xq = u8_nchw.view(np.int8) if "int8" in iq.type else u8_nchw
    oq_all = sq.run(None, {iq.name: xq})
    oq = dict(zip([o.name for o in sq.get_outputs()], oq_all))[FEAT_QUANT]
    print(f"FP32 特征: {o32.shape}   量化特征: {oq.shape} {oq.dtype}")

    # NHWC int16 → NCHW 通道维度对齐
    q = oq[0].transpose(2, 0, 1).astype(np.float32)                # [56,80,80]
    a = o32[0]                                                     # [56,80,80]
    print(f"量化特征范围: [{q.min():.1f}, {q.max():.1f}]  FP32 范围: [{a.min():.1f}, {a.max():.1f}]")

    print("\n逐通道相关系数（尺度无关）:")
    coses = []
    for c in range(a.shape[0]):
        u, v = a[c].reshape(-1), q[c].reshape(-1)
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        coses.append(float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else float("nan"))
    for c, s in enumerate(coses):
        flag = " ⚠️" if s < 0.98 else ""
        print(f"  ch{c:<3} corr={s:.4f}{flag}")
    print(f"\n平均 {np.nanmean(coses):.4f}  最差 {np.nanmin(coses):.4f}  "
          f"<0.98 通道数 {sum(1 for s in coses if s < 0.98)}/{len(coses)}")

    # 相对误差（尺度对齐后）
    ratios = []
    for c in range(a.shape[0]):
        u, v = a[c].reshape(-1), q[c].reshape(-1)
        nu = u @ u
        if nu > 0:
            k = (u @ v) / nu          # 最小二乘缩放
            ratios.append(np.abs(v - k * u).std() / (np.abs(k * u).std() + 1e-9))
    print(f"\n尺度对齐后相对误差: 平均 {np.nanmean(ratios):.4f}  最差 {np.nanmax(ratios):.4f}")


if __name__ == "__main__":
    main()
