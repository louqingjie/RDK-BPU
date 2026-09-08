#!/usr/bin/env python3
"""探测裁剪点张量（裸检测头输出）的数值动态范围，评估 int8/int16 量化步长。

用法：
    python3 scripts/probe_output_range.py <model.onnx> <cut_tensor_name> \
        --data data/horizon_x5/data/calibration_data/images --frames 20

输出各分支（box / cls / color / kpt）的 min、max、分位数，
以及按整张量 int8/int16 均匀量化时的步长，用于判断坐标精度是否够用。
"""

import argparse
import glob
import os
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

GROUPS = {"box(0:4)": (0, 4), "cls(4:16)": (4, 16),
          "color(16:20)": (16, 20), "kpt(20:28)": (20, 28)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cut_tensor")
    ap.add_argument("--data", default="/workspace/data/horizon_x5/data/calibration_data/images")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    args = ap.parse_args()

    m = shape_inference.infer_shapes(onnx.load(args.model))
    g = m.graph
    vi = {v.name: v for v in g.value_info}
    if args.cut_tensor not in vi:
        raise SystemExit(f"找不到张量 {args.cut_tensor}")
    g.output.append(vi[args.cut_tensor])

    tmp = os.path.join(tempfile.gettempdir(), "probe_range_tmp.onnx")
    onnx.save(m, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    files = sorted(glob.glob(os.path.join(args.data, "*.bin")))[:args.frames]
    if not files:
        raise SystemExit(f"没有找到输入数据: {args.data}")

    lat, outs, cuts = [], [], []
    for f in files:
        x = np.fromfile(f, dtype=np.float32)
        if args.layout == "nchw":
            x = x.reshape(1, 3, 576, 768)
        else:
            x = x.reshape(1, 576, 768, 3)
        t0 = time.perf_counter()
        o = sess.run(None, {in_name: x})
        lat.append((time.perf_counter() - t0) * 1000)
        outs.append(o[0])
        cuts.append(o[1])

    print(f"模型: {os.path.basename(args.model)}")
    print(f"输出: {np.array(outs[0]).shape}  CPU 延迟 {np.mean(lat):.1f} ms（仅参考）")
    print(f"裁剪张量: {cuts[0].shape}")

    c = np.concatenate([x[0].T for x in cuts], 0)  # (1,28,N) -> (N,28)
    print(f"\n{'分组':<14}{'min':>10}{'max':>10}{'0.1%':>10}{'99.9%':>10}{'std':>9}")
    for name, (a, b) in GROUPS.items():
        v = c[:, a:b]
        print(f"{name:<14}{v.min():>10.2f}{v.max():>10.2f}"
              f"{np.percentile(v, 0.1):>10.2f}{np.percentile(v, 99.9):>10.2f}{v.std():>9.3f}")

    print("\n按整张量范围均匀量化的步长：")
    for name, (a, b) in GROUPS.items():
        v = c[:, a:b]
        rng = max(abs(v.min()), abs(v.max()))
        print(f"  {name.split('(')[0]:<6} int8={2*rng/255:>8.3f}   int16={2*rng/65535:>8.5f}")

    os.remove(tmp)


if __name__ == "__main__":
    main()
