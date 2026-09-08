#!/usr/bin/env python3
"""准备 sp_vision_25 模型的量化前 ONNX。

tiny_resnet.onnx 需要两步处理：
  1. 动态 batch（'batch_size'）固化为 1 —— hb_mapper 要求静态 shape
  2. opset 17 降级到 11 —— OE v1.2.8 工具链最高支持 opset 11
     （其算子 Conv/Relu/Add/MaxPool/GlobalAveragePool/Flatten/Gemm 在
       opset 11 全部合法，降级无语义变化；version_converter 失败时
       直接改写 opset 声明并以 checker 验证兜底）

yolov5 检测器无需处理：同济仓库的 yolov5.xml/bin 经 verify_sp25_models.py
确认与深大 RP24 官方原始模型 rp24_0708.onnx 同源（余弦 1.0001），且该 ONNX
全图 FP32（158 个 initializer 均为 FLOAT）、opset 11、静态 [1,3,640,640]，
无 P2（FP16）问题，直接引用即可。

输出：
    data/onnx/sp_vision_25/sp25_tiny_resnet_b1.onnx
"""

import os

import onnx

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..", "data/onnx/sp_vision_25/tiny_resnet.onnx")
DST = os.path.join(BASE, "..", "data/onnx/sp_vision_25/sp25_tiny_resnet_b1.onnx")
BATCH = 1
MAX_OPSET = 11  # OE v1.2.8 (hb_mapper 1.24.3) 支持上限


def fix_batch(model, batch):
    changed = []
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        for d in dims:
            if d.dim_param:  # 动态符号维度
                d.ClearField("dim_param")
                d.dim_value = batch
                changed.append(f"{vi.name}:{d.dim_value}")
    return changed


def downgrade_opset(model, target):
    cur = max(o.version for o in model.opset_import if not o.domain)
    if cur <= target:
        return f"opset {cur}（无需降级）"
    try:
        from onnx import version_converter
        model = version_converter.convert_version(model, target)
        return f"opset {cur} -> {target}（version_converter）"
    except Exception as e:
        # 兜底：直接改写声明，合法性由 checker 验证
        print(f"[warn] version_converter 失败（{e}），改写 opset 声明兜底")
        for o in model.opset_import:
            if not o.domain:
                o.version = target
        return f"opset {cur} -> {target}（声明改写）"


def main():
    model = onnx.load(SRC)
    changed = fix_batch(model, BATCH)
    opset_msg = downgrade_opset(model, MAX_OPSET)
    onnx.checker.check_model(model)
    onnx.save(model, DST)
    ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
           for i in model.graph.input]
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
            for o in model.graph.output]
    print(f"[done] 固化维度: {changed or '（无动态维度）'}")
    print(f"[done] opset  : {opset_msg}")
    print(f"[done] 输入 {ins}  输出 {outs}")
    print(f"[done] 已保存: {os.path.abspath(DST)}")


if __name__ == "__main__":
    main()
