#!/usr/bin/env python3
"""定位 ONNX 中「可上 BPU 部分」与「后处理部分」的分界点。

思路：以 BPU 阻断算子（TopK / NMS / GatherND / ScatterND ...）为种子，
反向求其祖先节点集合 = 后处理子图；该子图的输入边界即推荐裁剪点。
"""

import argparse
import collections

import onnx
from onnx import shape_inference

BLOCKER = {"NonMaxSuppression", "NonMaxSuppressionCustom", "TopK", "GatherND",
           "ScatterND", "NonZero", "GatherElements", "ScatterElements", "Where"}


def dim_str(t):
    return "x".join(str(d.dim_value) if d.dim_value else (d.dim_param or "?")
                    for d in t.shape.dim)


def main(path, seeds=None):
    m = onnx.load(path)
    m = shape_inference.infer_shapes(m)
    g = m.graph

    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
    init = {i.name: i for i in g.initializer}

    seed_nodes = [n for n in g.node
                  if n.op_type in (seeds or BLOCKER)]
    print(f"=== {path} ===")
    print(f"阻断/后处理种子算子: {collections.Counter(n.op_type for n in seed_nodes)}")

    # 反向求祖先
    producer = {}
    for n in g.node:
        for o in n.output:
            producer[o] = n
    # 以「张量」为单位做深度优先回溯，避免节点名/张量名混用
    ancestors, stack = set(), []
    for n in seed_nodes:
        ancestors.add(n.name)
        stack.extend(i for i in n.input if i in producer and i not in init)
    while stack:
        t = stack.pop()
        nd = producer.get(t)
        if nd is None or nd.name in ancestors:
            continue
        ancestors.add(nd.name)
        stack.extend(i for i in nd.input if i in producer and i not in init)

    post = [n for n in g.node if n.name in ancestors]
    body = [n for n in g.node if n.name not in ancestors]
    print(f"\n后处理子图: {len(post)} 节点 / {len(g.node)}  "
          f"主干(可上BPU候选): {len(body)} 节点")

    # 边界张量：后处理子图消费、但由主干产生
    boundary = {}
    for n in post:
        for i in n.input:
            if i in producer and producer[i].name not in ancestors and i not in init:
                boundary.setdefault(i, []).append(n.op_type)
    print("\n[推荐裁剪点] 主干输出 -> 后处理消费方")
    for name, consumers in boundary.items():
        shape = dim_str(vi[name].type.tensor_type) if name in vi else "未知"
        prod = producer[name].op_type
        print(f"  {name:<28} [{shape:<20}] 由 {prod:<12} 产生 -> "
              f"{sorted(set(consumers))}")

    # 主干 FLOPs / 参数量估算（仅 Conv）
    params, flops = 0, 0
    for n in body:
        if n.op_type != "Conv":
            continue
        w = init.get(n.input[1])
        if w is None:
            continue
        dims = list(w.dims)
        if len(dims) != 4:
            continue
        oc, ic, kh, kw = dims
        nw = oc * ic * kh * kw
        params += nw + (oc if len(n.input) > 2 else 0)
        out = vi.get(n.output[0])
        if out and out.type.tensor_type.shape.dim:
            d = out.type.tensor_type.shape.dim
            if len(d) == 4 and d[2].dim_value and d[3].dim_value:
                flops += nw * d[2].dim_value * d[3].dim_value * 2
    print(f"\n主干 Conv 参数量: {params/1e6:.2f} M   "
          f"乘加: {flops/1e6:.0f} MMAC  ({flops/1e9:.2f} GMAC, 即 {2*flops/1e9:.2f} GFLOPs)")

    print("\n主干尾部算子链（最后 12 个节点）:")
    for n in body[-12:]:
        out_shape = dim_str(vi[n.output[0]].type.tensor_type) if n.output[0] in vi else "?"
        print(f"  {n.op_type:<14} {n.name:<34} -> [{out_shape}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seeds", nargs="*", default=None)
    main(ap.parse_args().model, ap.parse_args().seeds)
