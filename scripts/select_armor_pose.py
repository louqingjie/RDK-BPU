#!/usr/bin/env python3
"""按硬性要求评估并筛选权重：① 装甲板（Armor） ② 必须输出角点（Pose / 四点）。

数据源 `data/rm_weights/organized.json`（146 份）。通过者拷贝到
`data/rm_weights_selected/<年份>/`，并生成 `EVALUATION.md`。

角点输出的核验分四档：
  A 确认   ONNX 内嵌 Ultralytics 元数据 `task=pose`；或 PyTorch `.pt` 的 `train_args.task=pose`
           且类别为装甲（非 COCO 通用）。
  B 推定   无元数据，按输出通道拆解出 `4(bbox)+1(conf)+1(color)+4×2(角点)+m(num)`。
  C 继承   非 ONNX/PT（OpenVINO IR、TensorRT engine、AX650 axmodel、ncnn param），
           按「去掉学校名后的文件名」匹配到同源已确认模型。
  D 待验证 既非 ONNX/PT，又匹配不到同源模型 → **不入选**，单列待人工确认。

用法：
    python3 scripts/select_armor_pose.py            # 预览
    python3 scripts/select_armor_pose.py --apply    # 拷贝并写清单
"""
from __future__ import annotations

import argparse
import collections
import importlib.abc
import importlib.machinery
import json
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "data", "rm_weights")
DST_DIR = os.path.join(ROOT, "data", "rm_weights_selected")


# ----------------------------------------------------------- .pt 读取支持
def _install_stubs() -> None:
    """注入 ultralytics / models 桩模块，使 torch.load 能反序列化检查点。"""
    if getattr(_install_stubs, "_done", False):
        return

    def make(name):
        def _init(self, *a, **k):
            pass

        def _setstate(self, s):
            if isinstance(s, dict):
                self.__dict__.update(s)
            elif isinstance(s, tuple) and len(s) == 2 and isinstance(s[1], dict):
                self.__dict__.update(s[1])

        return type(name, (), {"forward": staticmethod(lambda *a, **k: None),
                               "__init__": _init, "__setstate__": _setstate})

    class Stub(types.ModuleType):
        def __getattr__(self, name):
            cls = make(name)
            setattr(self, name, cls)
            return cls

    class Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return Stub(spec.name)

        def exec_module(self, module):
            pass

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in ("ultralytics", "models", "utils"):
                return importlib.machinery.ModuleSpec(fullname, Loader())
            return None

    sys.meta_path.insert(0, Finder())
    _install_stubs._done = True


# --------------------------------------------------------------- 证据提取
def onnx_evidence(path: str) -> tuple[str, str]:
    import onnx
    try:
        m = onnx.load(path, load_external_data=False)
    except Exception as exc:  # noqa: BLE001
        return "D", f"ONNX 解析失败: {type(exc).__name__}"
    md = {p.key: p.value for p in m.metadata_props}
    if md.get("task") == "pose":
        return "A", (f"Ultralytics task=pose, kpt_shape={md.get('kpt_shape','?')}, "
                     f"names={md.get('names','?')[:40]}")
    if md.get("task") == "detect":
        return "X", f"Ultralytics task=detect（无关键点），names={md.get('names','?')[:40]}"
    try:
        dims = [d.dim_value for d in m.graph.output[-1].type.tensor_type.shape.dim]
        rest = [d for d in dims[1:] if d > 0]
        ch = min(rest) if rest else 0
        anchors = max(rest) if rest else 0
    except Exception:  # noqa: BLE001
        return "D", "无法读取输出形状"
    if any(d == 0 for d in dims):
        return "B", f"动态形状 {dims}，无法按通道拆解，按来源与命名认定"
    if ch >= 14:
        return "B", (f"输出 {dims} = 4(bbox)+1(conf)+1(color)+4×2(角点)+"
                     f"{ch - 14}(num)，锚点 {anchors}")
    return "X", f"输出通道 {ch}（{dims}）< 14，装不下四点分量"


def pt_evidence(path: str) -> tuple[str, str]:
    try:
        import torch
        _install_stubs()
        ck = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return "D", f"PyTorch 检查点无法读取: {type(exc).__name__}"
    if not isinstance(ck, dict):
        return "D", "非标准检查点（TorchScript / state_dict）"
    task = (ck.get("train_args") or {}).get("task")
    names = getattr(ck.get("model"), "names", None)
    info = f"PyTorch task={task}, names={str(names)[:46]}"
    if task == "pose":
        armorish = names is not None and not any(
            str(v).lower() in ("person", "bicycle", "car", "motorcycle", "airplane")
            for v in (names.values() if isinstance(names, dict) else names))
        return ("A", info) if armorish else ("X", f"{info} → 通用数据集，非装甲")
    if task == "detect":
        return "X", f"{info}（无关键点）"
    return "B", info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正拷贝并写清单（默认仅预览）")
    args = ap.parse_args()

    with open(os.path.join(SRC_DIR, "organized.json"), encoding="utf-8") as fh:
        entries = json.load(fh)

    # 可读权重（ONNX / PyTorch）的判定结果，供 IR / engine / axmodel 继承（key = sha256）
    verdict: dict[str, tuple[str, str]] = {}
    for e in entries:
        ext = os.path.splitext(e["path"])[1]
        if ext not in (".onnx", ".pt") or e["sha256"] in verdict:
            continue
        p = os.path.join(SRC_DIR, e["path"])
        if os.path.exists(p):
            verdict[e["sha256"]] = onnx_evidence(p) if ext == ".onnx" else pt_evidence(p)
    # 继承基准：(arch, pose, task, 归一化 detail) -> (档位, 文件名)
    FORMAT_TOKENS = ("_TensorRT", "_AX650", "_engine", "_ncnn", "_batch", "_fp16", "_fp8", "_IR")

    def norm_detail(d: str) -> str:
        s = d or ""
        for t in FORMAT_TOKENS:
            s = s.replace(t, "")
        return s.strip("_")

    proven: list[dict] = []
    for e in entries:
        ext = os.path.splitext(e["path"])[1]
        if ext not in (".onnx", ".pt"):
            continue
        p = os.path.join(SRC_DIR, e["path"])
        if not os.path.exists(p):
            continue
        tier = verdict.get(e["sha256"], ("D", ""))[0]
        if tier in ("A", "B"):
            proven.append({"school": e["school"], "arch": e["arch"], "pose": e["pose"],
                           "task": e["task"], "nd": norm_detail(e["detail"]),
                           "name": os.path.basename(e["path"]), "tier": tier})

    def inherit_from(e: dict) -> tuple[str, str] | None:
        """匹配同源已确认模型：先按 detail 前缀，再退化为同校同架构同首词。"""
        nd = norm_detail(e["detail"])
        exact, weak = [], []
        for p in proven:
            if (p["arch"], p["pose"], p["task"]) != (e["arch"], e["pose"], e["task"]):
                continue
            if nd and p["nd"] and (nd == p["nd"]
                                   or (len(p["nd"]) >= 3 and nd.startswith(p["nd"]))
                                   or (len(nd) >= 3 and p["nd"].startswith(nd))):
                exact.append((len(os.path.commonprefix([nd, p["nd"]])), p["tier"] == "A", p))
            elif (p["school"] == e["school"] and nd and p["nd"]
                  and nd.split("_")[0] == p["nd"].split("_")[0]
                  and len(nd.split("_")[0]) >= 3):
                weak.append((len(nd.split("_")[0]), p["tier"] == "A", p))
        if exact:
            exact.sort(key=lambda x: (x[0], x[1]), reverse=True)
            p = exact[0][2]
            return "C", f"非可读权重，同源已确认（{p['tier']} 档）模型 `{p['name']}`"
        if weak:
            weak.sort(key=lambda x: (x[0], x[1]), reverse=True)
            p = weak[0][2]
            return "C", f"非可读权重，同校同架构已确认（{p['tier']} 档）`{p['name']}`（同系列变体）"
        return None

    selected, excluded, pending = [], [], []
    for e in entries:
        path = os.path.join(SRC_DIR, e["path"])
        name = os.path.basename(e["path"])
        ext = os.path.splitext(name)[1]
        if e.get("task") != "Armor":
            excluded.append((name, f"非装甲板（task={e.get('task')}）"))
            continue
        if e.get("pose") != "Pose":
            excluded.append((name, "仅输出检测框（Det），无角点"))
            continue
        if not os.path.exists(path):
            excluded.append((name, "文件缺失"))
            continue

        if ext in (".onnx", ".pt"):
            tier, why = verdict.get(e["sha256"], ("D", "无法读取权重"))
        else:
            got = inherit_from(e)
            tier, why = got if got else ("D", "非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx）")

        if tier in ("X",):
            excluded.append((name, why))
        elif tier == "D":
            pending.append((name, why))
        else:
            selected.append({**e, "tier": tier, "why": why})

    selected.sort(key=lambda x: (x["year"], x["path"]))
    by = collections.Counter(x["year"] for x in selected)
    print(f"总 {len(entries)} 份 → 入选 {len(selected)} / 排除 {len(excluded)} / 待验证 {len(pending)}")
    print("入选年份:", dict(sorted(by.items())),
          "| 档位:", dict(collections.Counter(x["tier"] for x in selected)))
    print(f"合计 {sum(x['bytes'] or 0 for x in selected) / 1048576:.1f} MB")
    for x in selected:
        print(f"  [{x['tier']}] {x['year']} {os.path.basename(x['path'])}")
    if pending:
        print("\n  待验证（未入选）：")
        for n, w in pending:
            print(f"    - {n}: {w}")

    if not args.apply:
        print("\n（预览模式，未拷贝；加 --apply 执行）")
        return 0

    if os.path.isdir(DST_DIR):
        shutil.rmtree(DST_DIR)
    for x in selected:
        dst = os.path.join(DST_DIR, str(x["year"]), os.path.basename(x["path"]))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(SRC_DIR, x["path"]), dst)
    for name, _why in pending:                 # 待验证单列，不混入精选
        src = os.path.join(SRC_DIR, next(e["path"] for e in entries
                                         if os.path.basename(e["path"]) == name))
        dst = os.path.join(DST_DIR, "待验证", name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    TIER = {"A": ("A 确认", "元数据证明关键点回归（ONNX `task=pose` / PyTorch `train_args.task=pose` 且类别为装甲）"),
            "B": ("B 推定", "无元数据，按输出通道拆出 4×2 角点分量"),
            "C": ("C 继承", "IR / engine / axmodel，匹配到同源已确认模型")}
    total = sum(x["bytes"] or 0 for x in selected)
    lines = [
        "# 装甲板 + 角点权重评估（第一批精选）",
        "",
        "> 由 `scripts/select_armor_pose.py --apply` 生成。硬性要求：**① 装甲板权重**、**② 必须输出角点**。",
        f"> 从 `data/rm_weights/` 的 {len(entries)} 份中入选 **{len(selected)} 份**，合计约 {total / 1048576:.1f} MB；",
        f"> 排除 {len(excluded)} 份（非装甲 / 仅检测框），另 {len(pending)} 份列「待验证」不入选。",
        "",
        "## 一、置信度分档",
        "",
        "| 档位 | 判据 |",
        "|---|---|",
    ]
    for k in ("A", "B", "C"):
        lines.append(f"| {TIER[k][0]} | {TIER[k][1]} |")
    lines += ["", "## 二、入选清单（按年份）", ""]
    for year in sorted(by):
        group = [x for x in selected if x["year"] == year]
        lines += [f"### {year}（{len(group)} 份，{sum(g['bytes'] or 0 for g in group) / 1048576:.1f} MB）", "",
                  "| 文件 | 档位 | 证据 |", "|---|---|---|"]
        for x in sorted(group, key=lambda y: (y["tier"], os.path.basename(y["path"]))):
            lines.append(f"| `{os.path.basename(x['path'])}` | {x['tier']} | {x['why']} |")
        lines.append("")

    lines += ["## 三、可迁移性", "", "| 格式 | 数量 | 说明 |", "|---|---|---|"]
    for ext, n in sorted(collections.Counter(
            os.path.splitext(x["path"])[1] for x in selected).items()):
        locked = ext in (".engine", ".axmodel")
        lines.append(f"| `{ext}` | {n} | {'绑定平台，不可直接用于 RDK X5' if locked else '可迁移，量化首选'} |")
    lines.append("")

    if pending:
        lines += ["## 四、待验证（未入选）", "", "| 文件 | 原因 |", "|---|---|"]
        for n, w in pending:
            lines.append(f"| `{n}` | {w} |")
        lines.append("")

    lines += ["## 五、排除明细", "", "| 文件 | 原因 |", "|---|---|"]
    for n, w in sorted(excluded):
        lines.append(f"| `{n}` | {w} |")
    lines.append("")

    with open(os.path.join(DST_DIR, "EVALUATION.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n已写入 {os.path.relpath(DST_DIR, ROOT)}/（{len(selected)} 份 + EVALUATION.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
