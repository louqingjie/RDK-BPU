#!/usr/bin/env python3
"""分支级归一化手术：在三个尺度解码 Add 输出 [1,3,H,W,22] 处按通道组归一化。

通道布局（每组 22）：kpt[0:8] conf[8] color[9:13] num[13:22]
手术：
  kpt [0:8]  ÷1024          → 约 [-0.23, 0.89]（CPU 侧 ×1024 还原）
  conf+color [8:13] Sigmoid → 0~1（CPU 侧 conf 直接用、color/num argmax 不变）
  num        [13:22] Sigmoid → 0~1
  Concat 还原 22 通道，下游 Reshape/Transpose/Concat 不变

效果：整条链的量化张量单一量纲（0~1），INT8 步长 1/255 即满足
  kpt ±2 px、conf ±0.002、argmax 不变——消除混量纲损伤。

用法：
  python3 scripts/normalize_decode_sp25.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(src, dst):
    m = onnx.load(src)
    g = m.graph

    targets = [n for n in g.node
               if n.op_type == "Add" and "model.26" in n.name]
    if not targets:
        raise SystemExit("未找到 model.26 解码 Add 节点")

    new_nodes_all = []
    for node in targets:
        out = node.output[0]
        tag = out.replace("/", "_")[-32:]

        def add_init(name, arr):
            a = np.asarray(arr)
            dtype = np.float32 if a.dtype.kind == "f" else a.dtype
            g.initializer.append(numpy_helper.from_array(a.astype(dtype), name))
            return name

        def slice_node(name, start, end):
            s, e, a = f"{name}_s", f"{name}_e", f"{name}_a"
            g.initializer.append(numpy_helper.from_array(np.array([start], dtype=np.int64), s))
            g.initializer.append(numpy_helper.from_array(np.array([end], dtype=np.int64), e))
            g.initializer.append(numpy_helper.from_array(np.array([4], dtype=np.int64), a))
            outn = f"{name}_out"
            new_nodes_all.append(helper.make_node("Slice", [out, s, e, a], [outn], name=name))
            return outn

        # kpt: [0:8] ÷1024（范围约 [-0.23, 0.89]，CPU 侧 ×1024 还原）
        kpt_sl = slice_node(f"{tag}_kpt_sl", 0, 8)
        inv = add_init(f"{tag}_inv", 1.0 / 1024.0)
        kpt_n = f"{tag}_kpt_n"
        new_nodes_all.append(helper.make_node("Mul", [kpt_sl, inv], [kpt_n], name=f"{tag}_kpt_norm"))

        # conf+color: [8:13] Sigmoid（合并切片避免 [.,1] 简并中间张量）
        cc_sl = slice_node(f"{tag}_cc_sl", 8, 13)
        cc_s = f"{tag}_cc_s"
        new_nodes_all.append(helper.make_node("Sigmoid", [cc_sl], [cc_s], name=f"{tag}_cc_sig"))

        # num: [13:22] Sigmoid
        num_sl = slice_node(f"{tag}_num_sl", 13, 22)
        num_s = f"{tag}_num_s"
        new_nodes_all.append(helper.make_node("Sigmoid", [num_sl], [num_s], name=f"{tag}_num_sig"))

        # Concat 还原 22 通道（顺序不变：kpt8 + conf1 + color4 + num9）
        cat = f"{tag}_norm_out"
        new_nodes_all.append(helper.make_node(
            "Concat", [kpt_n, cc_s, num_s], [cat], name=f"{tag}_norm_concat", axis=4))

        # 下游消费者改接归一化输出
        for n2 in g.node:
            if n2 is node:
                continue
            for k, i in enumerate(n2.input):
                if i == out:
                    n2.input[k] = cat

    # 新节点插入图中（Add 之后即可，最后统一拓扑排序）
    for n in new_nodes_all:
        g.node.append(n)

    # 拓扑排序
    avail = {i.name for i in g.initializer} | {i.name for i in g.input}
    pending = list(g.node)
    sorted_nodes = []
    while pending:
        progressed = False
        for n in list(pending):
            if all((i in avail) or i == "" for i in n.input):
                sorted_nodes.append(n)
                avail.update(n.output)
                pending.remove(n)
                progressed = True
        if not progressed:
            raise RuntimeError("拓扑排序失败: " + str(pending[0].input[:3]))
    del g.node[:]
    g.node.extend(sorted_nodes)

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"已生成 {dst}：{len(targets)} 个尺度的解码输出已归一化"
          f"（kpt/1024、conf+color/num Sigmoid）")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
