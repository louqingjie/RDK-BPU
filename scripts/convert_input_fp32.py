#!/usr/bin/env python3
"""把 ONNX 的 FP16 输入改为 FP32（hb_mapper 要求），在图首插入 Cast 节点。

用法：
    python3 scripts/convert_input_fp32.py <in.onnx> <out.onnx>
"""

import sys

import onnx
from onnx import TensorProto, helper


def main(src, dst):
    m = onnx.load(src)
    g = m.graph
    inp = g.input[0]
    t = inp.type.tensor_type
    if t.elem_type != TensorProto.FLOAT16:
        print(f"输入已是 {TensorProto.DataType.Name(t.elem_type)}，无需转换")
        onnx.save(m, dst)
        return

    old_name = inp.name
    dims = [d.dim_value for d in t.shape.dim]

    # 新的 FP32 输入
    new_inp = helper.make_tensor_value_info(old_name + "_f32",
                                            TensorProto.FLOAT, dims)
    # 原名变成 Cast 的输出，后续节点不受影响
    cast = helper.make_node("Cast", [old_name + "_f32"], [old_name],
                            name="input_fp32_to_fp16", to=TensorProto.FLOAT16)

    g.input.remove(inp)
    g.input.insert(0, new_inp)
    g.node.insert(0, cast)

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"已生成 {dst}：输入 {old_name}_f32 [FLOAT {dims}] -> Cast -> {old_name} [FLOAT16]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
