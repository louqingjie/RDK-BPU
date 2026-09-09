#!/usr/bin/env python3
"""实验①：sp25_yolov5 逐阶段误差定位。

阶段：
  A  rp24_0708.onnx                     host ORT FP32（原始参考）
  B  optimized_float_model.onnx         host ORT（验证模型转换是否无损）
  C  INT8 bin                           板端（已知坏）
  D  all_int16 bin                      板端（head conv 已恢复 0.997+）

固定测试集：按 FP32 conf 排序取校准图 top2（阳性）+ 中位 1 张 + 最低 2 张（阴性），
共 5 张。指标全部为"分通道 + 检出对照"，禁用整体余弦下结论。

产出：/workspace/data/sp25_requant/report.md + stage_results.json
"""

import glob
import json
import os
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

BOARD = "sunrise@192.168.127.10"
SRC_DIR = "/workspace/data/horizon_x5/data/calibration_data_640_u8/source"
FP32_MODEL = "/workspace/data/onnx/sp_vision_25/rp24_0708.onnx"
OPT_MODEL = "/workspace/quant/sp25_yolov5/model_output/sp25_yolov5_640x640_optimized_float_model.onnx"
BOARD_MODELS = {
    "C_INT8": ("/workspace/quant/sp25_yolov5/model_output/sp25_yolov5_640x640.bin",
               "~/models/sp25_yolov5_640x640.bin"),
    "D_INT16": ("/workspace/quant/sp25_yolov5_all_int16/model_output/sp25_yolov5_640x640_all_int16.bin",
                "~/models/sp25_yolov5_640x640_all_int16.bin"),
}
OUT_DIR = "/workspace/data/sp25_requant"
BOARD_IN = "/tmp/rp_in"
BOARD_OUT = "/tmp/rp_out"
N_IMAGES = 5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def preprocess_fp32(img_rgb):
    """host ORT 输入：fp32 /255 NCHW（若模型输入是 uint8 则由调用方转换）。"""
    return (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def select_images():
    """按 FP32 conf 排序选图：top2（阳性）+ 中位 1 + 最低 2（阴性）。"""
    sess = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    files = sorted(glob.glob(f"{SRC_DIR}/*.jpg"))
    scores = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        x = preprocess_fp32(cv2.resize(img, (640, 640)))
        xi = x.astype(np.float16) if "float16" in inp.type else x
        o = sess.run(None, {inp.name: xi})[0].reshape(-1, 22)
        scores.append((float(sigmoid(o[:, 8]).max()), f))
    scores.sort(reverse=True)
    picks = [scores[0][1], scores[1][1], scores[len(scores) // 2][1],
             scores[-2][1], scores[-1][1]]
    print("[select] 测试集（FP32 conf 排序）:")
    for c, f in scores[:2]:
        print(f"  阳性 {os.path.basename(f)}  conf={c:.3f}")
    print(f"  中位 {os.path.basename(picks[2])}")
    for f in picks[3:]:
        print(f"  阴性 {os.path.basename(f)}")
    return picks


def load_optimized_strip_preprocess(path):
    """optimized_float_model 内嵌 Hz 融合算子（HzPreprocess/HzSwish/HzHardSwish），
    host ORT 无法加载。手术：
      1. HzPreprocess（×1/255）→ Identity（host 侧直接喂 0~1，等价）
      2. HzSwish(x) → x * Sigmoid(x)
      3. HzHardSwish(x) → x * Clip(x+3, 0, 6) / 6
    返回 ORT session（输入为 fp32 NCHW 0~1）。"""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(path)
    g = m.graph

    def const(name, vals):
        t = numpy_helper.from_array(np.asarray(vals, dtype=np.float32), name)
        g.initializer.append(t)
        return name

    new_nodes = []
    remove = []
    for n in g.node:
        if n.op_type == "HzPreprocess":
            # 输入语义已是 0~1 float（host 喂 /255 后的值），替换为直连
            for k, o in enumerate(n.output):
                for n2 in g.node:
                    for k2, i2 in enumerate(n2.input):
                        if i2 == o:
                            n2.input[k2] = "images"
            remove.append(n)
        elif n.op_type == "HzSwish":
            x = n.input[0]
            s = n.output[0] + "_sig"
            new_nodes.append(helper.make_node("Sigmoid", [x], [s], name=n.name + "_sig"))
            new_nodes.append(helper.make_node("Mul", [x, s], list(n.output), name=n.name + "_mul"))
            remove.append(n)
        elif n.op_type == "HzHardSwish":
            x = n.input[0]
            a = n.output[0] + "_a"
            c = n.output[0] + "_c"
            m1 = n.output[0] + "_m"
            new_nodes.append(helper.make_node("Add", [x, const(n.name + "_b3", [3.0])], [a], name=n.name + "_add"))
            new_nodes.append(helper.make_node("Clip", [a, const(n.name + "_lo", [0.0]), const(n.name + "_hi", [6.0])], [c], name=n.name + "_clip"))
            new_nodes.append(helper.make_node("Mul", [x, c], [m1], name=n.name + "_mul1"))
            new_nodes.append(helper.make_node("Mul", [m1, const(n.name + "_inv6", [1.0 / 6.0])], list(n.output), name=n.name + "_mul2"))
            remove.append(n)
    for n in remove:
        g.node.remove(n)
    for n in new_nodes:
        g.node.append(n)

    # 拓扑排序（新节点被追加到了末尾）
    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    sorted_nodes, pending = [], list(g.node)
    while pending:
        progressed = False
        for n in list(pending):
            if all((i in avail) or (i == "") for i in n.input):
                sorted_nodes.append(n)
                avail.update(n.output)
                pending.remove(n)
                progressed = True
        if not progressed:
            raise RuntimeError("图存在无法拓扑排序的环/缺失输入: " +
                               str(pending[0].input[:3]))
    del g.node[:]
    g.node.extend(sorted_nodes)

    onnx.checker.check_model(m)
    tmp = "/tmp/sp25_optimized_decomposed.onnx"
    onnx.save(m, tmp)
    return ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])


def run_host(model_path, x):
    if "optimized_float" in model_path:
        sess = load_optimized_strip_preprocess(model_path)
    else:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    xi = x
    if "uint8" in inp.type:
        xi = (x[0].transpose(1, 2, 0) * 255.0).round().astype(np.uint8)[None]
    elif "float16" in inp.type:
        xi = x.astype(np.float16)
    o = sess.run(None, {inp.name: xi})[0].reshape(-1, 22)
    return o


def run_board(tag, model_local, model_board, bins_local):
    """上传 bins 与模型，板上批量推理，拉回输出。"""
    subprocess.run(["ssh", "-o", "BatchMode=yes", BOARD,
                    f"rm -rf {BOARD_IN} {BOARD_OUT} && mkdir -p {BOARD_IN} {BOARD_OUT}"],
                   check=True)
    for name, blob in bins_local.items():
        p = f"/tmp/{name}.bin"
        with open(p, "wb") as f:
            f.write(blob.tobytes())
        subprocess.run(["scp", "-o", "BatchMode=yes", p, f"{BOARD}:{BOARD_IN}/{name}.bin"],
                       check=True)
        os.remove(p)
    subprocess.run(["scp", "-o", "BatchMode=yes", model_local, f"{BOARD}:{model_board}"],
                   check=True)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", BOARD,
                        f"bash /tmp/run_infer.sh {model_board}"],
                       capture_output=True, text=True)
    print(r.stdout.strip()[:400])
    outs = {}
    for name in bins_local:
        p = f"{OUT_DIR}/board_{tag}_{name}.bin"
        if subprocess.run(["scp", "-o", "BatchMode=yes",
                           f"{BOARD}:{BOARD_OUT}/{name}_out0.bin", p],
                          capture_output=True).returncode == 0:
            outs[name] = np.fromfile(p, np.float32).reshape(25200, 22)
        else:
            outs[name] = None
    return outs


def channel_metrics(ref, out):
    """分通道指标：kpt/conf/color/num。"""
    c_ref, c_out = sigmoid(ref[:, 8]), sigmoid(out[:, 8])
    m = {}
    m["conf_peak"] = (float(c_ref.max()), float(c_out.max()))
    m["conf_cand>0.25"] = (int((c_ref > 0.25).sum()), int((c_out > 0.25).sum()))
    m["conf_cos"] = cos(ref[:, 8], out[:, 8])
    m["kpt_cos"] = cos(ref[:, 0:8], out[:, 0:8])
    m["color_cos"] = cos(ref[:, 9:13], out[:, 9:13])
    m["num_cos"] = cos(ref[:, 13:22], out[:, 13:22])
    # top10 锚点重合（按 conf 排序的 anchor 索引）
    t_ref = set(np.argsort(-c_ref)[:10].tolist())
    t_out = set(np.argsort(-c_out)[:10].tolist())
    m["top10_overlap"] = f"{len(t_ref & t_out)}/10"
    # top 行的类别一致性
    i = int(np.argmax(c_out))
    m["top_row"] = dict(conf=float(c_out[i]),
                        num=int(np.argmax(out[i, 13:22])),
                        color=int(np.argmax(out[i, 9:13])),
                        num_ref=int(np.argmax(ref[i, 13:22])),
                        color_ref=int(np.argmax(ref[i, 9:13])),
                        kpt=[round(float(v), 1) for v in out[i, 0:8]])
    return m


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/board_out", exist_ok=True)
    picks = select_images()

    # host 阶段 A/B（统一走 run_host，自动处理 HzPreprocess 手术）
    host = {}
    for tag, mp in (("A_FP32", FP32_MODEL), ("B_FLOAT", OPT_MODEL)):
        host[tag] = {}
        for f in picks:
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            x = preprocess_fp32(cv2.resize(img, (640, 640)))
            o = run_host(mp, x)
            host[tag][os.path.basename(f).replace(".jpg", "")] = o
    print("[host] 阶段 A/B 完成")

    # 板端阶段 C/D：输入统一为 uint8 NHWC
    bins, names = {}, []
    for f in picks:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        data = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        name = os.path.basename(f).replace(".jpg", "")
        bins[name] = np.ascontiguousarray(data[None], dtype=np.uint8)
        names.append(name)

    board = {}
    for tag, (local, board_path) in BOARD_MODELS.items():
        print(f"[board] {tag} 推理中 ...")
        board[tag] = run_board(tag, local, board_path, bins)

    # 指标：以阶段 A 为参考
    report, jout = [], {}
    for name in names:
        ref = host["A_FP32"][name]
        row = {"image": name, "fp32_conf_peak": round(float(sigmoid(ref[:, 8]).max()), 3),
               "stages": {}}
        for tag, outs in (("B_FLOAT", host["B_FLOAT"]), ("C_INT8", board["C_INT8"]),
                          ("D_INT16", board["D_INT16"])):
            o = outs.get(name)
            if o is None:
                row["stages"][tag] = "缺失"
                continue
            row["stages"][tag] = channel_metrics(ref, o)
        jout[name] = row
        report.append(f"### {name}（FP32 conf 峰值 {row['fp32_conf_peak']}）")
        for tag, st in row["stages"].items():
            if st == "缺失":
                report.append(f"- {tag}: 缺失")
                continue
            report.append(
                f"- {tag}: conf峰 {st['conf_peak'][1]:.3f}(ref {st['conf_peak'][0]:.3f}) "
                f"候选 {st['conf_cand>0.25'][1]}(ref {st['conf_cand>0.25'][0]}) "
                f"top10重合 {st['top10_overlap']} | cos: kpt {st['kpt_cos']:.4f} "
                f"conf {st['conf_cos']:.4f} color {st['color_cos']:.4f} num {st['num_cos']:.4f} "
                f"| top行 num {st['top_row']['num']}(ref {st['top_row']['num_ref']}) "
                f"color {st['top_row']['color']}(ref {st['top_row']['color_ref']})")
        report.append("")

    md = "\n".join(report)
    print("\n" + md)
    with open(f"{OUT_DIR}/report.md", "w") as f:
        f.write("# 实验①：sp25_yolov5 逐阶段分通道误差报告\n\n```\n" + md + "\n```\n")
    with open(f"{OUT_DIR}/stage_results.json", "w") as f:
        json.dump(jout, f, ensure_ascii=False, indent=1)
    print(f"\n报告: {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
