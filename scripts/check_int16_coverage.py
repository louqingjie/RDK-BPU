#!/usr/bin/env python3
"""核查 all_int16 的"配置声明"与"实际生效"是否一致。

三件事：
  1. quant_info 对比：INT8 vs all_int16 中同名节点的校准阈值（找"阈值数千"的张量）
  2. quantized_model.onnx 图普查：各 dtype 张量计数、head Conv 输出的实际类型
  3. head Conv 之后链条上的量化/反量化算子模式（判断 INT16 是否真正延伸到解码段）

用法：
  python3 scripts/check_int16_coverage.py
"""

import json
import collections

import onnx
from onnx import TensorProto

DT = {TensorProto.FLOAT: "fp32", TensorProto.FLOAT16: "fp16",
      TensorProto.INT8: "int8", TensorProto.INT16: "int16",
      TensorProto.INT32: "int32", TensorProto.UINT8: "uint8"}

INT8_DIR = "/workspace/quant/sp25_yolov5/model_output"
INT16_DIR = "/workspace/quant/sp25_yolov5_all_int16/model_output"
HEAD = "model.26"


def load_quant_info(path):
    with open(path) as f:
        return json.load(f)


def compare_thresholds(d8, d16):
    """同名节点的阈值对比：找出"阈值数千"的张量与 INT8/INT16 差异。"""
    print("=" * 72)
    print("[1] 同名节点校准阈值对比（INT8 vs all_int16）")
    common = set(d8) & set(d16)
    big, changed = [], []
    for name in sorted(common):
        t8 = d8[name].get("thresholds")
        t16 = d16[name].get("thresholds")
        if not t8:
            continue
        flat = [abs(x) for grp in t8 for x in grp]
        mx = max(flat) if flat else 0
        if mx >= 1000:
            big.append((name, mx, t8, t16))
    print(f"  阈值 ≥1000 的节点数: {len(big)} / {len(common)}")
    for name, mx, t8, t16 in big[:12]:
        same = "阈值相同" if json.dumps(t8) == json.dumps(t16) else "阈值不同"
        print(f"  {name[:58]:<58} max|thr|={mx:>9.1f}  [{same}]")
    if big:
        print("\n  -- 逐个展示前 3 个节点的 INT8/INT16 阈值差 --")
        for name, mx, t8, t16 in big[:3]:
            print(f"  ▸ {name}")
            print(f"    INT8  thresholds: {json.dumps(t8)[:180]}")
            print(f"    INT16 thresholds: {json.dumps(t16)[:180]}")
            if json.dumps(t8) == json.dumps(t16):
                print("    ⇒ INT16 编译下该节点阈值未变（INT16 未覆盖此处或该张量本就高精度）")
    return common, big


def dtype_census(path, tag):
    """量化图的 dtype 普查 + head Conv 链条上的张量类型。"""
    print("=" * 72)
    print(f"[2] {tag}: quantized graph dtype 普查")
    m = onnx.load(path)
    g = m.graph
    dt = collections.Counter()
    for vi in list(g.value_info) + list(g.output):
        dt[DT.get(vi.type.tensor_type.elem_type, str(vi.type.tensor_type.elem_type))] += 1
    for init in g.initializer:
        dt["init_" + DT.get(init.data_type, str(init.data_type))] += 1
    print("  ", dict(dt))

    ops = collections.Counter(n.op_type for n in g.node)
    qops = {k: v for k, v in ops.items() if "Hz" in k or "Quant" in k or "Dequant" in k}
    print("   量化相关算子:", qops)

    # head Conv 及其后继链条上的张量类型
    print(f"\n  -- {HEAD} 检测头链条上的张量类型 --")
    shown = 0
    for n in g.node:
        if HEAD in n.name or any(HEAD in i for i in n.input if i):
            outs = []
            for o in n.output:
                t = None
                for vi in list(g.value_info) + list(g.output):
                    if vi.name == o:
                        t = DT.get(vi.type.tensor_type.elem_type,
                                   str(vi.type.tensor_type.elem_type))
                outs.append(f"{o[-40:]}:{t}")
            print(f"   {n.op_type:<28} -> {' | '.join(outs)}")
            shown += 1
            if shown >= 14:
                break
    return g


def head_chain_models():
    """对 INT8 / INT16 两个量化图分别打印 head 链条。"""
    import os
    for tag, d in (("INT8", INT8_DIR), ("all_int16", INT16_DIR)):
        p = os.path.join(d, "sp25_yolov5_640x640_quantized_model.onnx")
        p16 = os.path.join(d, "sp25_yolov5_640x640_all_int16_quantized_model.onnx")
        path = p16 if os.path.exists(p16) else p
        if not os.path.exists(path):
            import glob
            cands = glob.glob(os.path.join(d, "*quantized_model.onnx"))
            path = cands[0] if cands else None
        if path:
            dtype_census(path, tag)


def main():
    d8 = load_quant_info(f"{INT8_DIR}/sp25_yolov5_640x640_quant_info.json")
    d16 = load_quant_info(f"{INT16_DIR}/sp25_yolov5_640x640_all_int16_quant_info.json")
    common, big = compare_thresholds(d8, d16)
    head_chain_models()
    print("=" * 72)
    print("[3] head Conv 余弦相似度（quant_info 自带，佐证用）")
    for tag, d in (("INT8", d8), ("all_int16", d16)):
        sims = [(float(it["cosine_similarity"]), n) for n, it in d.items()
                if it.get("cosine_similarity") and HEAD + "/m." in n and "/Conv" in n]
        sims.sort()
        for s, n in sims:
            print(f"  [{tag}] {s:.4f} {n[:60]}")


if __name__ == "__main__":
    main()
