#!/usr/bin/env python3
"""SHtech SKD 板端输出 vs host FP32 基准对比（P14 阳性对照验收）。

对比维度：max conf、候选数@0.1、top1 的 tag/color/size argmax 与解码角点误差。
用法: python3 shtech_board_compare.py <board_out.bin> <frame_idx>
"""

import sys
import numpy as np

sys.path.insert(0, "/workspace/scripts")
from shtech_fp32_baseline import feed_fp32, decode  # noqa: E402

import cv2
import onnxruntime as ort  # noqa: E402

FP32_CSV = "/workspace/data/shtech_skd_positive/fp32_baseline.csv"
VIDEO = "/workspace/.tmp_shtech_auto_aim/test.avi"
MODEL = "/workspace/data/onnx/shtech/SKD250526.onnx"


def main():
    board_bin, frame_idx = sys.argv[1], int(sys.argv[2])
    board = np.fromfile(board_bin, np.float32).reshape(6720, 21)
    board_cands = sorted(decode(board[None]), key=lambda c: -c["conf"])

    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    x, _ = feed_fp32(frame)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    fp32_cands = sorted(decode(sess.run(None, {"image": x})[0]), key=lambda c: -c["conf"])
    cap.release()

    def top_summary(cands, n=3):
        s = []
        for c in cands[:n]:
            pts = ";".join(f"({px:.0f},{py:.0f})" for px, py in c["pts"])
            s.append(f"conf={c['conf']:.4f} tag={c['tag']} color={c['color']} size={c['size']} pts={pts}")
        return s

    print(f"frame {frame_idx}")
    print(f"  host FP32 : cand@0.1={len(fp32_cands)}  max_conf={fp32_cands[0]['conf']:.4f}")
    print(f"  board bin : cand@0.1={len(board_cands)}  max_conf={board_cands[0]['conf']:.4f}")
    print("  host top3:")
    for s in top_summary(fp32_cands):
        print("    ", s)
    print("  board top3:")
    for s in top_summary(board_cands):
        print("    ", s)

    t1, b1 = fp32_cands[0], board_cands[0]
    pt_err = max(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(t1["pts"], b1["pts"]))
    print(f"  top1 角点最大欧氏误差: {pt_err:.2f} px")
    print(f"  top1 argmax 一致: tag={t1['tag']==b1['tag']} color={t1['color']==b1['color']} size={t1['size']==b1['size']}")
    print(f"  结论: {'✅ PASS' if (len(board_cands) > 0 and t1['tag'] == b1['tag'] and t1['color'] == b1['color'] and pt_err < 8) else '❌ FAIL'}")


if __name__ == "__main__":
    main()
