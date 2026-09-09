#!/usr/bin/env python3
"""SHtech SKD250526 host FP32 基准 + 阳性样本生成。

流程（P14/P16 验收方法论）：
  1. test.avi 均匀抽帧 -> 直拉 resize 640x512 -> BGR2RGB -> /255 -> NCHW FP32
  2. ORT 跑 SKD250526.onnx，按 AXCL.cpp 语义解码（conf=ch8 概率、kpt 相对偏移、
     tag ch9..16 / color ch17..18 / size ch19..20 argmax）
  3. 选 conf 最高的 --positive N 帧存为阳性样本：
     <out>/rgb/*.bin      uint8 RGB NCHW（rgb 输入对照用）
     <out>/nv12/*.bin     uint8 NV12 Y+UV（板端 nv12 喂数用）
     <out>/jpg/*.jpg      人工复核图
  4. 打印每帧基准：max conf / 候选数@0.1 / top1 解码角点与类别
"""

import argparse
import os
import csv

import cv2
import numpy as np
import onnxruntime as ort

INPUT_H, INPUT_W = 512, 640
KEEP_THRES = 0.1


def feed_fp32(frame_bgr):
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
    chw = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(chw), rgb


def bgr_to_nv12(bgr):
    """BGR uint8 -> NV12 (Y plane + interleaved UV), 尺寸 H*W*3/2。"""
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)  # Y(H*W) + U(H*W/4) + V(H*W/4)
    y = yuv[: h * w]
    u = yuv[h * w : h * w + h * w // 4]
    v = yuv[h * w + h * w // 4 :]
    uv = np.empty(h * w // 2, np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return np.concatenate([y, uv])


def decode(output):
    """[1,6720,21] -> (candidates, anchors)。按 AXCL.cpp 网格遍历顺序。"""
    out = output.reshape(6720, 21)
    stride, x_center, y_center = 8, 0, 0
    cands = []
    for i in range(6720):
        row = out[i]
        if row[8] >= KEEP_THRES:
            pts = []
            for k in range(4):
                pts.append((row[2 * k] * 2 * stride + x_center,
                            row[2 * k + 1] * 2 * stride + y_center))
            cands.append({
                "conf": float(row[8]),
                "pts": pts,
                "tag": int(np.argmax(row[9:17])),
                "color": int(np.argmax(row[17:19])),
                "size": int(np.argmax(row[19:21])),
            })
        x_center += stride
        x_center = 0 if x_center == INPUT_W else x_center
        y_center += stride if x_center == 0 else 0
        y_center = 0 if y_center == INPUT_H else y_center
        stride *= 2 if (x_center == 0 and y_center == 0) else 1
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/workspace/.tmp_shtech_auto_aim/test.avi")
    ap.add_argument("--model", default="/workspace/data/onnx/shtech/SKD250526.onnx")
    ap.add_argument("--out", default="/workspace/data/shtech_skd_positive")
    ap.add_argument("--sample", type=int, default=120, help="均匀抽帧数")
    ap.add_argument("--positive", type=int, default=3, help="保存的阳性帧数")
    ap.add_argument("--stride-frames", type=int, default=5, help="相邻采样帧间隔，避免近似重复")
    args = ap.parse_args()

    for d in ("rgb", "nv12", "jpg"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.video} 总帧数 {total}")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"[model] input {inp_name} {sess.get_inputs()[0].shape}")

    picks = np.linspace(0, total - 1, args.sample).round().astype(int)
    picks = sorted(set(int(i // args.stride_frames) * args.stride_frames for i in picks))

    rows = []
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        x, rgb = feed_fp32(frame)
        out = sess.run(None, {inp_name: x})[0]
        cands = decode(out)
        cands.sort(key=lambda c: -c["conf"])
        top = cands[0] if cands else None
        rows.append({
            "frame": fi,
            "n_cand@0.1": len(cands),
            "max_conf": round(max((c["conf"] for c in cands), default=0.0), 4),
            "top1_conf": round(top["conf"], 4) if top else 0.0,
            "top1_pts": ";".join(f"({px:.0f},{py:.0f})" for px, py in top["pts"]) if top else "",
            "top1_tag": top["tag"] if top else -1,
            "top1_color": top["color"] if top else -1,
            "top1_size": top["size"] if top else -1,
        })

    rows.sort(key=lambda r: -r["max_conf"])
    with open(os.path.join(args.out, "fp32_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = [r for r in rows if r["max_conf"] > 0.5][: args.positive]
    print(f"[positive] conf>0.5 的帧 {len(pos)} / 采样 {len(rows)}，取前 {len(pos)} 张保存")
    for r in pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(cv2.resize(frame, (INPUT_W, INPUT_H)), cv2.COLOR_BGR2RGB)
        stem = f"pos_{r['frame']:06d}_conf{r['max_conf']:.2f}"
        rgb.transpose(2, 0, 1)[None].astype(np.uint8).tofile(
            os.path.join(args.out, "rgb", stem + ".bin"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_to_nv12(bgr).tofile(os.path.join(args.out, "nv12", stem + ".bin"))
        cv2.imwrite(os.path.join(args.out, "jpg", stem + ".jpg"), frame)
        print(f"  saved {stem}: cand={r['n_cand@0.1']} tag={r['top1_tag']} "
              f"color={r['top1_color']} size={r['top1_size']} pts={r['top1_pts']}")

    cap.release()
    print(f"[done] 基准: {args.out}/fp32_baseline.csv")
    print("[done] 基准概览（按 max_conf 降序前 10）:")
    for r in rows[:10]:
        print(f"  frame {r['frame']:>5}  cand={r['n_cand@0.1']:>3}  "
              f"max_conf={r['max_conf']:.3f}  tag={r['top1_tag']} color={r['top1_color']}")


if __name__ == "__main__":
    main()
