#!/usr/bin/env python3
"""真实相机帧 -> 640x640 uint8 bin -> 板端推理 -> 后处理标注。

用法：python3 scripts/real_camera_test.py <bin模型名>
前置：/workspace/data/camera_test/frames/*.raw（1280x1024 RGB raw）
"""

import glob
import os
import subprocess
import sys

import cv2
import numpy as np

HOST_MODEL_DIR = "/workspace/quant/rp0526/model_output"
BOARD = "sunrise@192.168.127.10"
FRAME_DIR = "/workspace/data/camera_test/frames"
OUT_DIR = "/workspace/data/camera_test"
BOARD_IN = "/tmp/rp_in"
BOARD_OUT = "/tmp/rp_out"
CONF_TH = 0.25
NMS_IOU = 0.5
NUM_CLASSES = ["G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"]
COLORS = ["blue", "red", "gray", "purple"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_bins():
    """把相机帧与一张校准图转成 640x640 uint8 RGB NCHW bin，上传板端。"""
    bins = []
    os.makedirs(f"{OUT_DIR}/pre", exist_ok=True)
    for f in sorted(glob.glob(f"{FRAME_DIR}/*.raw"))[:5]:
        img = np.fromfile(f, np.uint8).reshape(1024, 1280, 3)  # RGB
        name = os.path.basename(f).replace(".raw", "")
        data = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{OUT_DIR}/pre/{name}_org.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        bins.append((name, data))
    # 对照组：一张含装甲板的校准图
    cal = sorted(glob.glob("/workspace/data/horizon_x5/data/calibration_data_640_u8/source/*.jpg"))[0]
    img = cv2.cvtColor(cv2.imread(cal), cv2.COLOR_BGR2RGB)
    bins.append(("calib_ref", img))
    cv2.imwrite(f"{OUT_DIR}/pre/calib_ref_org.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    os.makedirs(f"{BOARD_IN}", exist_ok=True) if False else None
    os.makedirs(f"{OUT_DIR}/input_bins", exist_ok=True)
    for name, data in bins:
        # 模型 input_layout_rt=NHWC：uint8 交错排列（H,W,C）
        blob = np.ascontiguousarray(data[None], dtype=np.uint8)
        blob.tofile(f"{OUT_DIR}/input_bins/{name}.bin")
    print(f"生成 {len(bins)} 个输入 bin")

    subprocess.run(["ssh", "-o", "BatchMode=yes", BOARD,
                    f"rm -rf {BOARD_IN} {BOARD_OUT} && mkdir -p {BOARD_IN} {BOARD_OUT}"], check=True)
    subprocess.run(["scp", "-o", "BatchMode=yes"] +
                   glob.glob(f"{OUT_DIR}/input_bins/*.bin") + [f"{BOARD}:{BOARD_IN}/"], check=True)
    return [os.path.basename(b[0]) if False else b[0] for b in bins]


def run_board(names):
    """板端逐帧推理，拉回输出。"""
    model = sys.argv[1] if len(sys.argv) > 1 else "~/models/rp0526_nhwc.bin"
    # 单 SSH 会话内串行推理全部帧（避免频繁加载导致的瞬时 -6000006）
    loop = (
        "rm -f " + BOARD_OUT + "/*.bin " + BOARD_OUT + "/times.txt; "
        "for f in " + BOARD_IN + "/*.bin; do "
        "n=$(basename $f .bin); ok=0; "
        "for try in 1 2 3 4 5; do "
        "t=$(hrt_model_exec infer --model_file " + model +
        " --input_file $f --enable_dump --dump_format bin --dump_path " +
        BOARD_OUT + " 2>&1 | grep -i 'infer time'); "
        "if [ -f " + BOARD_OUT + "/model_infer_output_0_output.bin ]; then "
        "mv " + BOARD_OUT + "/model_infer_output_0_output.bin " + BOARD_OUT + "/$n.bin; "
        "echo \"$n: $t\" >> " + BOARD_OUT + "/times.txt; ok=1; break; fi; "
        "echo \"$n TRY$try FAIL: $(echo \"$t\" | grep -oE 'error code:-[0-9]+' | head -1)\" >> " + BOARD_OUT + "/times.txt; "
        "sleep 2; done; "
        "done; cat " + BOARD_OUT + "/times.txt"
    )
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", BOARD, loop],
                       capture_output=True, text=True)
    print(r.stdout)
    for name in names:
        src = f"{BOARD_OUT}/{name}.bin"
        dst = f"{OUT_DIR}/board_out/{name}.bin"
        if subprocess.run(["scp", "-o", "BatchMode=yes", f"{BOARD}:{src}", dst],
                          capture_output=True).returncode != 0:
            print(f"[warn] {name} 未获得板端输出")
    return times


def postprocess(names):
    """归一化输出: conf 已 sigmoid 直接用, kpt ×640 还原, argmax 数字/颜色, NMS。"""
    dets_all = {}
    for name in names:
        ob = f"{OUT_DIR}/board_out/{name}.bin"
        if not os.path.exists(ob):
            print(f"[post] {name}: 无输出文件")
            continue
        o = np.fromfile(ob, np.float32).reshape(25200, 22)
        conf = o[:, 8]
        keep_rows = np.where(conf > CONF_TH)[0]
        dets = []
        for i in keep_rows:
            row = o[i]
            kpt = row[0:8].reshape(4, 2) * 640.0
            num_id = int(np.argmax(row[13:22]))
            color_id = int(np.argmax(row[9:13]))
            x0, y0 = kpt[:, 0].min(), kpt[:, 1].min()
            x1, y1 = kpt[:, 0].max(), kpt[:, 1].max()
            dets.append(dict(kpt=kpt, num=num_id, color=color_id,
                             conf=float(conf[i]), rect=(x0, y0, x1, y1)))
        # NMS（按外接矩形 IoU）
        dets.sort(key=lambda d: -d["conf"])
        keep = []
        for d in dets:
            ok = True
            for k in keep:
                ix0 = max(d["rect"][0], k["rect"][0]); iy0 = max(d["rect"][1], k["rect"][1])
                ix1 = min(d["rect"][2], k["rect"][2]); iy1 = min(d["rect"][3], k["rect"][3])
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                a1 = (d["rect"][2] - d["rect"][0]) * (d["rect"][3] - d["rect"][1])
                a2 = (k["rect"][2] - k["rect"][0]) * (k["rect"][3] - k["rect"][1])
                if inter / (a1 + a2 - inter + 1e-9) > NMS_IOU:
                    ok = False
                    break
            if ok:
                keep.append(d)
        dets_all[name] = keep
        print(f"[post] {name}: conf>0.25 候选 {len(keep_rows)} -> NMS 后 {len(keep)}")
        for d in keep[:5]:
            print(f"        num={NUM_CLASSES[d['num']]:<3} color={COLORS[d['color']]:<7} "
                  f"conf={d['conf']:.2f} rect={np.round(d['rect'],0).astype(int)}")
    return dets_all


def annotate(dets_all):
    scale_x, scale_y = 1280 / 640, 1024 / 640
    for name, dets in dets_all.items():
        org = f"{OUT_DIR}/pre/{name}_org.jpg"
        if not os.path.exists(org):
            continue
        img = cv2.imread(org)
        for d in dets:
            k = d["kpt"].copy()
            k[:, 0] *= scale_x
            k[:, 1] *= scale_y
            pts = k.astype(int)
            for a in range(4):
                b = (a + 1) % 4
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
            label = f"{NUM_CLASSES[d['num']]} {COLORS[d['color']]} {d['conf']:.2f}"
            cv2.putText(img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(f"{OUT_DIR}/{name}_det.jpg", img)
    print(f"标注图已保存到 {OUT_DIR}/*_det.jpg")


if __name__ == "__main__":
    names = make_bins()
    run_board(names)
    dets = postprocess(names)
    annotate(dets)
