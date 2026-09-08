#!/usr/bin/env python3
"""RDK 板端 ONNX Runtime FP32 兜底推理。

输入是与 input_layout_rt=NHWC 一致的 uint8 RGB HWC bin；输出保持原模型的
[25200, 22] float32 语义（第 8 通道为 conf logit）。
"""

import argparse
import time

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("input", help="640x640 RGB uint8 HWC bin")
    parser.add_argument("--output", default="ort_output.bin")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    rgb = np.fromfile(args.input, np.uint8).reshape(1, 640, 640, 3)
    images = rgb.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    for _ in range(args.warmup):
        session.run(None, {"images": images})

    started = time.perf_counter()
    output = session.run(None, {"images": images})[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    output.astype(np.float32).tofile(args.output)

    conf = output.reshape(-1, 22)[:, 8]
    best = float(conf.max())
    print(f"infer_ms={elapsed_ms:.2f} max_conf_logit={best:.4f} "
          f"max_conf_prob={1 / (1 + np.exp(-best)):.4f} "
          f"candidates@0.25={(conf > -1.0986123).sum()}")


if __name__ == "__main__":
    main()
