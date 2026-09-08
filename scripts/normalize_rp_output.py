#!/usr/bin/env python3
"""把深大 RP 模型的输出归一化到 0~1，解决输出张量 INT8 量化被 kpt 离群值撑爆的问题。

手术（在最终 output 上）：
    output[0:8]   kpt   /640          -> 0~1（CPU 侧 ×640 还原）
    output[8:9]   conf  Sigmoid       -> 0~1（CPU 侧不再做 sigmoid）
    output[9:13]  color Sigmoid       -> 0~1（argmax 不变）
    output[13:22] num   Sigmoid       -> 0~1（argmax 不变）
    Concat -> 新 output

各分支成为独立张量后，工具链为每个张量单独计算量化 scale，
且全张量值域 0~1，INT8 步长 = 1/255，conf 分辨率 0.004，kpt 误差 ±1.25 px。

用法：
    python3 scripts/normalize_rp_output.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


def main(src, dst):
    m = onnx.load(src)
    g = m.graph
    out_name = g.output[0].name

    # 原输出改名 output_raw
    for o in g.output:
        if o.name == out_name:
            o.name = out_name + "_raw"
    raw = out_name + "_raw"
    for n in g.node:
        for k, o in enumerate(n.output):
            if o == out_name:
                n.output[k] = raw

    inits = {}

    def add_init(name, arr, dtype=None):
        t = numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)
        g.initializer.append(t)
        inits[name] = True
        return name

    def add_init_i64(name, vals):
        t = numpy_helper.from_array(np.array(vals, dtype=np.int64), name)
        g.initializer.append(t)
        return name

    def slice_channels(start, end, tag):
        outs = [f"{raw}_slice_{tag}"]
        n = helper.make_node(
            "Slice", [raw,
                      add_init_i64(f"sl_{tag}_starts", [start]),
                      add_init_i64(f"sl_{tag}_ends", [end]),
                      add_init_i64(f"sl_{tag}_axes", [2])],
            outs, name=f"slice_{tag}")
        g.node.append(n)
        return outs[0]

    kpt = slice_channels(0, 8, "kpt")
    conf = slice_channels(8, 9, "conf")
    color = slice_channels(9, 13, "color")
    num = slice_channels(13, 22, "num")

    # kpt 截断到 [0,640] 再 /640（背景 anchor 的离群值会被 conf 过滤，截断无副作用）
    kpt_c = "kpt_clipped"
    g.node.append(helper.make_node(
        "Clip", [kpt,
                 add_init("clip_min", np.float32(0.0)),
                 add_init("clip_max", np.float32(640.0))],
        [kpt_c], name="kpt_clip"))
    kpt_n = "output_kpt"
    g.node.append(helper.make_node(
        "Mul", [kpt_c, add_init("kpt_scale", np.float32(1.0 / 640.0))],
        [kpt_n], name="kpt_normalize"))

    # conf / color / num 过 Sigmoid
    sig = {"conf": "conf_sig", "color": "output_color", "num": "output_num"}
    for tag, src in (("conf", conf), ("color", color), ("num", num)):
        g.node.append(helper.make_node("Sigmoid", [src], [sig[tag]], name=f"sig_{tag}"))

    # conf 单独成 [1,25200,1] 会在 runtime 布局转换中被清零（实测），
    # 因此把它并入 kpt 输出：[8 kpt/640 + 1 conf] = [1,25200,9]，值域同为 0~1
    kpt_conf = "output_kpt_conf"
    g.node.append(helper.make_node(
        "Concat", [kpt_n, sig["conf"]], [kpt_conf], name="kpt_conf_concat", axis=2))

    # 拆成 3 个独立图输出：每个输出张量拥有独立的量化 scale，
    # 避免 22 通道拼接后被 kpt/conf 量级差异互相拖垮
    g.output.remove(g.output[0])
    outs = [
        ("output_kpt_conf", kpt_conf, [1, 25200, 9]),
        ("output_color", sig["color"], [1, 25200, 4]),
        ("output_num", sig["num"], [1, 25200, 9]),
    ]
    for name, src, shape in outs:
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))

    m = shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"已生成 {dst}：4 个独立输出（kpt/640 归一化 + conf/color/num sigmoid），各自独立量化")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
