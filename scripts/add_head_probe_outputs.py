#!/usr/bin/env python3
"""导出包含三个检测头输入特征的 ONNX 调试模型。"""

import sys

import onnx
from onnx import helper, shape_inference


HEAD_INPUTS = {
    "/m/model.17/cv3/act/Mul_output_0": "probe_head0_in",
    "/m/model.20/cv3/act/Mul_output_0": "probe_head1_in",
    "/m/model.23/cv3/act/Mul_output_0": "probe_head2_in",
}

CV3_INPUTS = {
    "/m/model.17/Concat_output_0": "probe_head0_cv3_in",
    "/m/model.20/Concat_output_0": "probe_head1_cv3_in",
    "/m/model.23/Concat_output_0": "probe_head2_cv3_in",
}

CONCAT23_INPUTS = {
    "/m/model.23/m/m.0/cv2/act/Mul_output_0": "probe_23_residual",
    "/m/model.23/cv2/act/Mul_output_0": "probe_23_bypass",
    "/m/model.22/Concat_output_0": "probe_22_concat",
}

CONCAT22_INPUTS = {
    "/m/model.21/act/Mul_output_0": "probe_21_downsample",
    "/m/model.10/act/Mul_output_0": "probe_10_backbone",
}

MODEL10_STAGES = {
    "/m/model.9/cv2/act/Mul_output_0": "probe_10_in",
    "/m/model.10/conv/Conv_output_0": "probe_10_conv",
    "/m/model.10/act/Mul_output_0": "probe_10_out",
}


def main(src: str, dst: str, probes: dict[str, str]) -> None:
    model = shape_inference.infer_shapes(onnx.load(src))
    known = {v.name: v for v in (*model.graph.value_info, *model.graph.input,
                                 *model.graph.output)}
    for tensor, output_name in probes.items():
        if tensor not in known:
            raise RuntimeError(f"missing inferred shape for {tensor}")
        value = known[tensor]
        output = helper.make_tensor_value_info(
            output_name, value.type.tensor_type.elem_type,
            [d.dim_value for d in value.type.tensor_type.shape.dim],
        )
        model.graph.node.append(helper.make_node("Identity", [tensor], [output_name],
                                                 name=f"debug_{output_name}"))
        model.graph.output.append(output)
    onnx.checker.check_model(model)
    onnx.save(model, dst)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(f"usage: {sys.argv[0]} INPUT.onnx OUTPUT.onnx [head|cv3|concat23|concat22|model10]")
    stages = {"head": HEAD_INPUTS, "cv3": CV3_INPUTS, "concat23": CONCAT23_INPUTS,
              "concat22": CONCAT22_INPUTS}
    stages["model10"] = MODEL10_STAGES
    stage = sys.argv[3] if len(sys.argv) == 4 else "head"
    main(sys.argv[1], sys.argv[2], stages[stage])
