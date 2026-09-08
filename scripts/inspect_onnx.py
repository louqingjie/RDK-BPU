#!/usr/bin/env python3
"""解析 ONNX 图结构，输出 RDK X5（BPU bayes-e）量化可行性所需的算子/形状信息。

用法：
    python3 scripts/inspect_onnx.py <model.onnx> [...]
"""

import collections
import sys

import onnx

# BPU 明确不支持 / 高风险算子（数据依赖、动态形状、非逐元素）
BLOCKER = {
    "NonMaxSuppression", "NonMaxSuppressionCustom", "TopK", "ScatterND", "GatherND",
    "ScatterElements", "GatherElements", "CumSum", "Loop", "If", "Scan",
}
# 需重点确认精度/是否回落 CPU
RISKY = {
    "Softmax", "Sigmoid", "HardSigmoid", "Swish", "HardSwish", "Erf", "Pow", "Exp",
    "Log", "Sqrt", "Div", "Resize", "Upsample", "Where", "Range", "Cast", "Tile",
    "Expand", "Shape", "Equal", "Greater", "Less", "Not", "And", "Or", "InstanceNormalization",
    "LayerNormalization", "ReduceMean", "MatMul", "Gemm", "LSTM", "Attention",
}


def dim_str(dims):
    out = []
    for d in dims:
        out.append(d.dim_param if d.dim_param else str(d.dim_value))
    return "x".join(out)


def inspect(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    opsets = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    print("=" * 78)
    print(f"文件 : {path}")
    print(f"大小 : {__import__('os').path.getsize(path)/1e6:.2f} MB  "
          f"ir={m.ir_version}  opset={opsets}  producer={m.producer_name!r}")

    print("\n[输入]")
    for i in g.input:
        if i.name in {init.name for init in g.initializer}:
            continue
        t = i.type.tensor_type
        print(f"  {i.name:<20} {onnx.TensorProto.DataType.Name(t.elem_type):<8} "
              f"[{dim_str(t.shape.dim)}]")
    print("[输出]")
    for o in g.output:
        t = o.type.tensor_type
        print(f"  {o.name:<20} {onnx.TensorProto.DataType.Name(t.elem_type):<8} "
              f"[{dim_str(t.shape.dim)}]")

    cnt = collections.Counter(n.op_type for n in g.node)
    params = sum(1 for _ in g.initializer)
    print(f"\n[结构] 节点 {len(g.node)}  参数张量 {params}  "
          f"算子种类 {len(cnt)}")

    print("\n[算子统计]  标记: ✗=BPU不支持  ?=需确认  ·=常规")
    for op, n in cnt.most_common():
        flag = "✗" if op in BLOCKER else ("?" if op in RISKY else "·")
        print(f"  {flag} {op:<22} {n:>4}")

    blockers = [(n.name, n.op_type) for n in g.node if n.op_type in BLOCKER]
    risky = collections.Counter(n.op_type for n in g.node if n.op_type in RISKY)
    print("\n[阻断算子明细]" if blockers else "\n[阻断算子] 无")
    for name, op in blockers:
        print(f"  ✗ {op:<20} {name}")
    if risky:
        print("[需确认算子] " + ", ".join(f"{k}x{v}" for k, v in risky.most_common()))

    # 动态维度检查
    dyn = []
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        t = vi.type.tensor_type
        if any(d.dim_param for d in t.shape.dim):
            dyn.append((vi.name, dim_str(t.shape.dim)))
    print(f"\n[动态维度] {len(dyn)} 处" + ("" if not dyn else ""))
    for n, s in dyn[:10]:
        print(f"  {n} -> {s}")

    # 输出分支：哪些节点直接连到图输出
    outs = {o.name for o in g.output}
    print("[输出前级算子]")
    for n in g.node:
        if set(n.output) & outs:
            print(f"  {n.op_type:<20} -> {sorted(set(n.output) & outs)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        inspect(p)
