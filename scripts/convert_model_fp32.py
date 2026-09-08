#!/usr/bin/env python3
"""把整个 ONNX 模型从 FP16 转成 FP32（hb_mapper 不接受图内 FP16 张量）。

处理：
    1. initializers: FLOAT16 -> FLOAT32
    2. 图输入/输出/value_info 的 dtype 同步转换
    3. 移除"图首 Cast(fp32->fp16)"，把 FP32 输入直接接回主干
    4. 其余 Cast 节点若目标是 FLOAT16，改为 FLOAT（保持图连通，量化在工具链内部进行）

用法：
    python3 scripts/convert_model_fp32.py <in.onnx> <out.onnx>
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


def to_fp32(t):
    return TensorProto.FLOAT


def main(src, dst):
    m = onnx.load(src)
    g = m.graph

    # 1) initializers
    n_init = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new)
            n_init += 1

    # 2) value_info / input / output 的 dtype
    def fix_type(vi):
        n = 0
        for v in vi:
            if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
                v.type.tensor_type.elem_type = TensorProto.FLOAT
                n += 1
        return n

    n_vi = fix_type(g.value_info) + fix_type(g.input) + fix_type(g.output)

    # 3) 移除图首的 fp32->fp16 Cast
    removed = []
    producer = {}
    for node in g.node:
        for o in node.output:
            producer[o] = node
    for node in list(g.node):
        if node.op_type == "Cast" and node.input[0].endswith("_f32"):
            # 把 Cast 的输出名（原输入名）重接为 FP32 输入名
            old_out = node.output[0]
            new_in = node.input[0]
            for n2 in g.node:
                for k, i in enumerate(n2.input):
                    if i == old_out:
                        n2.input[k] = new_in
            g.node.remove(node)
            removed.append(f"{old_out} <- {new_in}")

    # 4) 其余 Cast 到 FLOAT16 的改为 FLOAT
    n_cast = 0
    for node in g.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    n_cast += 1

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"转换完成 -> {dst}")
    print(f"  FP16 初始化张量: {n_init} 个")
    print(f"  FP16 类型声明:   {n_vi} 处")
    print(f"  移除图首 Cast:   {removed}")
    print(f"  改写内部 Cast:   {n_cast} 个")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
