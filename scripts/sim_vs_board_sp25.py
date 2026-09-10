#!/usr/bin/env python3
"""实验E：量化模型 host 仿真（hmct 定制 ORT）vs 板端输出 vs FP32 三方分通道对照。

仿真输入：uint8 NCHW 0~255（按位存入 int8 张量，与校准喂入方式一致）。
判据：仿真 vs 板端一致 → 误差在量化本身；不一致 → 误差在板端执行。
"""

import glob
import os

import cv2
import numpy as np
import onnxruntime as ort
from hmct.executor.session import InferenceSession as HzSession

SRC_DIR = "/workspace/data/horizon_x5/data/calibration_data_640_u8/source"
OUT_DIR = "/workspace/data/sp25_requant"
FP32_MODEL = "/workspace/data/onnx/sp_vision_25/rp24_0708.onnx"
SIM_MODELS = {
    "E_INT8_sim": "/workspace/quant/sp25_yolov5/model_output/sp25_yolov5_640x640_quantized_model.onnx",
    "E_INT16_sim": "/workspace/quant/sp25_yolov5_all_int16/model_output/sp25_yolov5_640x640_all_int16_quantized_model.onnx",
}
BOARD_FILES = {
    "C_INT8": f"{OUT_DIR}/board_C_INT8_0173.bin",
    "D_INT16": f"{OUT_DIR}/board_D_INT16_0173.bin",
}
GROUPS = {"kpt": slice(0, 8), "conf": slice(8, 9), "color": slice(9, 13), "num": slice(13, 22)}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")


def main():
    s0 = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    files = sorted(glob.glob(f"{SRC_DIR}/*.jpg"))
    picks = [f for f in files if any(k in f for k in ("0173", "0118", "0129", "0026", "0001"))]

    results = {}
    for tag, mp in SIM_MODELS.items():
        sess = HzSession(mp)
        inp = sess.get_inputs()[0]
        results[tag] = {}
        for f in picks:
            name = os.path.basename(f).replace(".jpg", "")
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            data = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            nchw = np.ascontiguousarray(data.transpose(2, 0, 1)[None])
            xi = nchw.view(np.int8) if "int8" in inp.type else nchw
            o = sess.run(None, {inp.name: xi})[0].reshape(-1, 22)
            results[tag][name] = o
        print(f"[sim] {tag}: {len(results[tag])} 张完成")

    o32_all = {}
    for f in picks:
        name = os.path.basename(f).replace(".jpg", "")
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        x = (cv2.resize(img, (640, 640)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        o32_all[name] = s0.run(None, {"images": x})[0].reshape(25200, 22)

    board = {}
    for tag, pat in (("C_INT8", "board_C_INT8_*.bin"), ("D_INT16", "board_D_INT16_*.bin")):
        for p in glob.glob(f"{OUT_DIR}/{pat}"):
            name = os.path.basename(p).replace(".bin", "").replace(f"board_{tag}_", "")
            board[(tag, name)] = np.fromfile(p, np.float32).reshape(25200, 22)

    # 逐图分通道报告
    lines = []
    for f in picks:
        name = os.path.basename(f).replace(".jpg", "")
        o32 = o32_all[name]
        c32 = sigmoid(o32[:, 8])
        lines.append(f"### {name}（FP32 conf 峰值 {c32.max():.3f}，候选 {(c32 > 0.25).sum()}）")
        stages = [("A_FP32", o32)]
        for tag in SIM_MODELS:
            stages.append((tag, results[tag].get(name)))
        for tag in ("C_INT8", "D_INT16"):
            stages.append((tag, board.get((tag, name))))
        for tag, o in stages:
            if o is None:
                lines.append(f"- {tag}: 缺失")
                continue
            c = sigmoid(o[:, 8])
            coses = " ".join(f"{g} {cos(o32[:, sl], o[:, sl]):.4f}" for g, sl in GROUPS.items())
            lines.append(f"- {tag:<12} conf峰 {c.max():.3f}  候选 {(c > 0.25).sum():>4}  | {coses}")
        lines.append("")

    # 仿真 vs 板端同配置对照
    lines.append("## 仿真 vs 板端（同配置同输入）")
    verdict = {}
    for tag_sim, tag_board in (("E_INT8_sim", "C_INT8"), ("E_INT16_sim", "D_INT16")):
        for name in [os.path.basename(f).replace(".jpg", "") for f in picks]:
            a = results.get(tag_sim, {}).get(name)
            b = board.get((tag_board, name))
            if a is None or b is None:
                continue
            coses = {g: cos(a[:, sl], b[:, sl]) for g, sl in GROUPS.items()}
            ok = all(v > 0.99 for v in coses.values())
            verdict[(tag_sim, tag_board, name)] = ok
            lines.append(f"- {name} [{tag_sim[:8]} vs {tag_board[:8]}]: " +
                         " ".join(f"{g} {coses[g]:.4f}" for g in GROUPS) +
                         ("  ✅一致" if ok else "  ❌不一致"))

    md = "\n".join(lines)
    print(md)
    with open(f"{OUT_DIR}/sim_vs_board.md", "w") as f:
        f.write("# 实验E：量化仿真 vs 板端分通道对照\n\n```\n" + md + "\n```\n")
    with open(f"{OUT_DIR}/sim_vs_board.json", "w") as f:
        json.dump({f"{t}|{n}": o.tolist() for (t, n), o in report.items()} |
                  {f"{t}|{n}": o.tolist() for (t, n), o in board.items()}, f)


if __name__ == "__main__":
    import json
    main()
