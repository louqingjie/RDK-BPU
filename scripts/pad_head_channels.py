#!/usr/bin/env python3
"""把 head Conv（66 出通道）pad 到 128 出通道，验证"剪枝非规整通道数触发 BPU
codegen bug"假设，同时可能直接绕过 P21（conf/color/num 通道全零）。

手术：
  1. 找到所有 weight.shape[0] == 66 的 Conv（三个尺度的 head）
  2. 权重/bias 补零到 128 出通道
  3. Conv 输出改名 *_padded，其后接 Slice(axis=1, 0:66) 还原原输出名
     （下游 Reshape/decode 逻辑完全不变；补零的 62 通道被丢弃）

用法：
  python3 scripts/pad_head_channels.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(src, dst):
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    padded = []

    for n in g.node:
        if n.op_type != "Conv" or len(n.input) < 2:
            continue
        w = init.get(n.input[1])
        if w is None:
            continue
        arr = numpy_helper.to_array(w)
        if arr.ndim != 4 or arr.shape[0] != 66:
            continue
        oc = arr.shape[0]
        target = 128
        # 权重补零
        w2 = np.zeros((target,) + arr.shape[1:], dtype=arr.dtype)
        w2[:oc] = arr
        g.initializer.remove(init[n.input[1]])
        g.initializer.append(numpy_helper.from_array(w2, n.input[1] + "_padded"))
        n.input[1] = n.input[1] + "_padded"
        # bias 补零
        if len(n.input) > 2 and n.input[2] in init:
            b = numpy_helper.to_array(init[n.input[2]])
            b2 = np.zeros(target, dtype=b.dtype)
            b2[:oc] = b
            g.initializer.remove(init[n.input[2]])
            g.initializer.append(numpy_helper.from_array(b2, n.input[2] + "_padded"))
            n.input[2] = n.input[2] + "_padded"
        # Conv 输出改名 + Slice 还原
        old_out = n.output[0]
        new_out = old_out + "_padded128"
        n.output[0] = new_out
        tag = n.name.replace("/", "_")[-28:]
        s_name, e_name, a_name = f"sl_{tag}_s", f"sl_{tag}_e", f"sl_{tag}_a"
        g.initializer.append(numpy_helper.from_array(np.array([0], dtype=np.int64), s_name))
        g.initializer.append(numpy_helper.from_array(np.array([oc], dtype=np.int64), e_name))
        g.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), a_name))
        sl = helper.make_node(
            "Slice", [new_out, s_name, e_name, a_name],
            [old_out], name="slice_" + tag)
        # 插到紧随其后的位置（拓扑序：Slice 必须在消费者之前）
        idx = list(g.node).index(n)
        g.node.insert(idx + 1, sl)
        padded.append((n.name, oc, target))

    if not padded:
        print("未找到 66 通道 head Conv")
        return

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"已生成 {dst}")
    for name, oc, t in padded:
        print(f"  {name}: {oc} -> {t} 通道（Slice 还原 {oc}）")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
