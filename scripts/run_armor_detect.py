#!/usr/bin/env python3
"""用精选权重库里的可运行模型，对给定图片跑装甲板检测，输出带标注的结果图。

支持的张量布局（均从各战队开源推理代码中核实，详见 EVALUATION.md / 本文件注释）：
  deepu_v5   深大 V5：  [kpt(8, 绝对像素), conf(1, logit), color(4), num(n)]
  deepu_v8   深大 V8-21：[color(4), class(9), kpt(8, 绝对像素)]（通道在前）
  yolox_rm   华科/武科大：[kpt(8, grid 相对), obj(1), color(4), class(n)]
  ul_pose    Ultralytics pose：[box(4, cxcywh 像素), class(nc), kpt(8)]
  e2e14      talos：    [x1,y1,x2,y2, conf, cls_idx, kpt(8)]
  e2e18      武科大 praysky：[x1,y1,x2,y2, conf, cls_idx, color(4), kpt(8)]

用法：
    python3 scripts/run_armor_detect.py <img1> [img2 ...] [--conf 0.3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL_DIR = os.path.join(ROOT, "data", "rm_weights_selected")
OUT_DIR = os.path.join(ROOT, "data", "armor_query", "results")

COLOR_NAMES = ["B", "R", "W", "P"]          # 蓝 / 红 / 白 / 紫（ASCII，避免字体缺字）
PALETTE = [(255, 90, 0), (0, 0, 255), (240, 240, 240), (200, 0, 200)]

# 文件名关键字 -> 布局家族
FAMILY_BY_HINT = [
    ("华中科技大学_YOLOX", "yolox_rm"),
    ("武汉科技大学_YOLOX", "yolox_rm"),
    ("深圳大学_YOLOv5", "deepu_v5"),
    ("深圳大学_YOLOv8", "deepu_v8"),
    ("上海科技大学_YOLOv5", "deepu_v5_grid"),     # 该校版本为 anchor-free，角点是 grid 相对坐标
    ("中国科学院大学_YOLOv5", "deepu_v5"),
    ("浙江师范大学_YOLOv5", "zlion"),
    ("RPS战队_EfficientNet", "ul_pose"),
    ("中国科学院大学_YOLO11", "ul_pose"),
    ("talos战队_YOLO26", "e2e14"),
    ("武汉科技大学_YOLO26", "e2e18"),
]
YOLOX_CLASSES = {"华中科技大学": 12, "武汉科技大学": 8}


# ------------------------------------------------------------------ 预处理
def letterbox(img, tw, th, value=114):
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), value, np.uint8)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, s, left, top


def preprocess(img_bgr, tw, th, nhwc=False):
    canvas, s, pl, pt = letterbox(img_bgr, tw, th)
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob[None] if nhwc else blob.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob), s, pl, pt


# ------------------------------------------------------------------ 推理
class Runner:
    """统一封装 ONNXRuntime / OpenVINO，自动处理 NHWC、fp16 输入与 opset 过高。"""

    def __init__(self, path):
        self.path = path
        self.tmp = None
        self.kind = "onnx" if path.endswith(".onnx") else "openvino"
        if self.kind == "onnx":
            import onnxruntime as ort
            try:
                self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            except Exception:
                import onnx                          # opset 过高 → 降到 17 再试
                import onnx.version_converter
                m = onnx.load(path, load_external_data=False)
                m = onnx.version_converter.convert_version(m, 17)
                self.tmp = path + ".op17.onnx"
                onnx.save(m, self.tmp)
                self.sess = ort.InferenceSession(self.tmp, providers=["CPUExecutionProvider"])
            i = self.sess.get_inputs()[0]
            self.in_name = i.name
            self.dtype = np.float16 if "float16" in i.type else np.float32
            d = list(i.shape)
        else:
            import openvino as ov
            self.compiled = ov.Core().compile_model(path, "CPU")
            i = self.compiled.inputs[0]
            self.in_name = 0
            self.dtype = np.float16 if "f16" in str(i.element_type) else np.float32
            d = list(i.shape)

        self.nhwc = (len(d) == 4 and (d[3] == 3 or d[-1] == 3))
        try:
            if self.nhwc:
                self.th, self.tw = int(d[1]), int(d[2])
            else:
                self.th, self.tw = int(d[2]), int(d[3])
        except (TypeError, ValueError):
            self.th, self.tw = 640, 640
        if self.th <= 0 or self.tw <= 0:
            self.th, self.tw = 640, 640

    def __call__(self, blob):
        blob = np.ascontiguousarray(blob.astype(self.dtype))
        if self.kind == "onnx":
            return np.asarray(self.sess.run(None, {self.in_name: blob})[0])
        return np.asarray(self.compiled([blob])[self.compiled.output(0)])


def nms(boxes, scores, iou_th=0.45):
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes([b.tolist() for b in boxes], [float(s) for s in scores],
                           0.0, iou_th)
    return list(np.array(idx).flatten()) if len(idx) else []


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------------ 解码
def decode(out, family, meta, infer, img_shape):
    """返回 ([(x1,y1,x2,y2, conf, cls_id, cls_name, color_id, kpts(4,2))], note)

    note 非空表示该模型的输出布局异常（无法按开源代码的解码方式还原）。
    """
    s, pl, pt = infer["scale"], infer["pad_l"], infer["pad_t"]
    H, W = img_shape[:2]
    names = meta.get("names") or {}
    dets = []

    def back(pts):                       # letterbox 像素 -> 原图
        return np.stack([(pts[:, 0] - pl) / s, (pts[:, 1] - pt) / s], 1)

    if family == "e2e14":
        for row in out[0]:
            x1, y1, x2, y2, conf, cid = row[0], row[1], row[2], row[3], row[4], int(round(row[5]))
            if conf < 0.05:
                continue
            kp = back(row[6:14].reshape(4, 2).astype(np.float32))
            dets.append(((x1 - pl) / s, (y1 - pt) / s, (x2 - pl) / s, (y2 - pt) / s,
                         float(conf), cid, names.get(cid, f"cls{cid}"), -1, kp))
    elif family == "e2e18":
        for row in out[0]:
            x1, y1, x2, y2, conf, cid = row[0], row[1], row[2], row[3], row[4], int(round(row[5]))
            if conf < 0.05:
                continue
            color = int(np.argmax(row[6:10]))
            kp = back(row[10:18].reshape(4, 2).astype(np.float32))
            dets.append(((x1 - pl) / s, (y1 - pt) / s, (x2 - pl) / s, (y2 - pt) / s,
                         float(conf), cid, names.get(cid, f"cls{cid}"), color, kp))
    elif family in ("deepu_v5", "deepu_v5_grid"):
        col8 = out[0][:, 8]
        if float(col8.max() - col8.min()) < 1e-9:
            return [], "conf channel(ch8) constant -> layout mismatch"
        grid = None
        if family == "deepu_v5_grid":
            grid = []
            for st in (8, 16, 32):
                for gy in range(infer["th"] // st):
                    for gx in range(infer["tw"] // st):
                        grid.append((gx, gy, st))
        # 深大 V5 的 conf 是原始 logit（官方代码不自带 sigmoid）；SKD 版已 sigmoid。
        # 判据：通道出现负值 → logit；全为非负 → 已是概率。
        need_sig = float(col8.min()) < 0.0
        for ai, row in enumerate(out[0]):
            conf = float(sigmoid(row[8])) if need_sig else float(row[8])
            if conf < 0.05:
                continue
            pts = row[0:8].reshape(4, 2).astype(np.float32)
            if grid is not None:
                if ai >= len(grid):
                    break
                gx, gy, st = grid[ai]
                pts = np.array([[p[0] + gx, p[1] + gy] for p in pts], np.float32) * st
            kp = back(pts)
            color = int(np.argmax(row[9:13]))
            cid = int(np.argmax(row[13:])) if row.shape[0] > 13 else -1
            dets.append((float(kp[:, 0].min()), float(kp[:, 1].min()),
                         float(kp[:, 0].max()), float(kp[:, 1].max()),
                         conf, cid, f"num{cid}", color, kp))
    elif family == "deepu_v8":
        cols = out[0]                                   # [C, N]
        need_sig = float(np.max(cols[4:13])) > 1.0
        for i in range(cols.shape[1]):
            c = cols[:, i]
            color = int(np.argmax(c[0:4]))
            cid = int(np.argmax(c[4:13]))
            kp = back(c[13:21].reshape(4, 2).astype(np.float32))
            # 用类别分数当置信度（V8 21 维无独立 obj 通道）
            rawc = float(np.max(c[4:13]))
            conf = float(sigmoid(rawc)) if need_sig else rawc
            dets.append((float(kp[:, 0].min()), float(kp[:, 1].min()),
                         float(kp[:, 0].max()), float(kp[:, 1].max()),
                         conf, cid, names.get(cid, f"cls{cid}"), color, kp))
    elif family == "yolox_rm":
        strides = [8, 16, 32]
        grid = []
        for st in strides:
            for gy in range(infer["th"] // st):
                for gx in range(infer["tw"] // st):
                    grid.append((gx, gy, st))
        ncls = YOLOX_CLASSES.get(infer["school"], 8)
        oc = out[0][:, 8]
        if float(oc.max() - oc.min()) < 1e-9:
            return [], "obj channel(ch8) constant -> export lost objectness, cannot decode"
        need_sig = float(oc.min()) < 0.0
        for i, row in enumerate(out[0]):
            if i >= len(grid):
                break
            obj = float(sigmoid(row[8])) if need_sig else float(row[8])
            if obj < 0.05:
                continue
            gx, gy, st = grid[i]
            pts = np.array([[float(row[j]) + gx, float(row[j + 1]) + gy]
                            for j in (0, 2, 4, 6)], np.float32) * st
            kp = back(pts)
            color = int(np.argmax(row[9:13]))
            cid = int(np.argmax(row[13:13 + ncls]))
            dets.append((float(kp[:, 0].min()), float(kp[:, 1].min()),
                         float(kp[:, 0].max()), float(kp[:, 1].max()),
                         obj, cid, f"cls{cid}", color, kp))
    elif family == "zlion":
        # 浙师大 ZLion2025：[bbox4, conf1, kpt8, cls11]
        for row in out[0]:
            conf = float(row[4])
            if conf < 0.02:
                continue
            kp = back(row[5:13].reshape(4, 2).astype(np.float32))
            cid = int(np.argmax(row[13:]))
            cx, cy, bw, bh = (float(v) for v in row[0:4])
            dets.append(((cx - bw / 2 - pl) / s, (cy - bh / 2 - pt) / s,
                         (cx + bw / 2 - pl) / s, (cy + bh / 2 - pt) / s,
                         conf, cid, f"cls{cid}", -1, kp))
    elif family == "ul_pose":
        a = out[0]
        # 通道在前 [C,N] 还是 [N,C]
        if a.shape[0] < a.shape[1]:
            a = a.T
        nk = 4
        for row in a:
            nc = row.shape[0] - 4 - nk * 2
            cls_scores = row[4:4 + nc]
            cid = int(np.argmax(cls_scores))
            conf = float(cls_scores[cid])
            if conf < 0.05:
                continue
            cx, cy, bw, bh = (float(v) for v in row[0:4])
            x1, y1, x2, y2 = (cx - bw / 2 - pl) / s, (cy - bh / 2 - pt) / s, \
                             (cx + bw / 2 - pl) / s, (cy + bh / 2 - pt) / s
            kp = back(row[4 + nc:4 + nc + nk * 2].reshape(nk, 2).astype(np.float32))
            dets.append((x1, y1, x2, y2, conf, cid, names.get(cid, f"cls{cid}"), -1, kp))
    return [d for d in dets if d[4] >= 0.05], ""


# ------------------------------------------------------------------ 绘制
def draw(img, dets, title, conf_th):
    canvas = img.copy()
    kept = [d for d in dets if d[4] >= conf_th]
    if kept:
        idx = nms([np.array(d[:4], np.float32) for d in kept], [d[4] for d in kept])
        kept = [kept[i] for i in idx]
    kept.sort(key=lambda d: -d[4])
    for x1, y1, x2, y2, conf, cid, cname, color, kp in kept[:12]:
        col = PALETTE[color] if color >= 0 else (0, 255, 0)
        # 按质心角排序画轮廓（各队点序不同，否则会画成蝴蝶结）；
        # 原始点序号仍按模型输出顺序标注 0..3
        order = np.argsort(np.arctan2(kp[:, 1] - kp[:, 1].mean(),
                                      kp[:, 0] - kp[:, 0].mean()))
        cv2.polylines(canvas, [kp[order].astype(np.int32).reshape(-1, 1, 2)],
                      True, col, 3, cv2.LINE_AA)
        for i, p in enumerate(kp):
            cv2.circle(canvas, (int(p[0]), int(p[1])), 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(p[0]), int(p[1])), 9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        label = f"id={cid} {cname} {conf:.2f}"
        if color >= 0:
            label += f" {COLOR_NAMES[color]}"
        (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        ty = max(int(y1) - 10, th_ + 10)
        cv2.rectangle(canvas, (int(x1), ty - th_ - 8), (int(x1) + tw_ + 8, ty + 6), col, -1)
        cv2.putText(canvas, label, (int(x1) + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0) if sum(col) > 380 else (255, 255, 255), 3, cv2.LINE_AA)
    bar = f"{title}  |  dets>={conf_th}: {len(kept)}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(canvas, bar, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3, cv2.LINE_AA)
    return canvas, kept


# ------------------------------------------------------------------ 主流程
def build_models():
    items = []
    for year in ("2024", "2025", "2026"):
        d = os.path.join(SEL_DIR, year)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith((".onnx", ".xml")):
                continue
            stem = os.path.splitext(f)[0]
            fam = next((v for k, v in FAMILY_BY_HINT if stem.startswith(k)), None)
            if fam is None:
                continue
            items.append((year, os.path.join(d, f), stem, fam))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--diag", action="store_true", help="额外打印原始输出统计")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    models = build_models()
    print(f"可运行模型 {len(models)} 个\n")

    summary = []
    for img_path in args.images:
        img = cv2.imread(img_path)
        tag = os.path.splitext(os.path.basename(img_path))[0][:8]
        print(f"===== {os.path.basename(img_path)} {img.shape}")
        per_img = []
        for idx, (year, path, stem, fam) in enumerate(models, 1):
            try:
                import onnx
                meta = {}
                if path.endswith(".onnx"):
                    m = onnx.load(path, load_external_data=False)
                    md = {p.key: p.value for p in m.metadata_props}
                    meta["names"] = eval(md["names"]) if "names" in md else None
                try:
                    r = Runner(path)
                except Exception as exc:  # noqa: BLE001
                    note = f"load failed: {type(exc).__name__} (opset too new for ort 1.16)"
                    print(f"  [{idx:2d}] {fam:<9} {stem[:52]:<52} <<{note}>>")
                    per_img.append({"model": f"{year}_{stem}", "family": fam, "note": note,
                                    "dets": 0, "best": 0.0})
                    continue
                blob, s, pl, pt = preprocess(img, r.tw, r.th, nhwc=r.nhwc)
                infer = {"scale": s, "pad_l": pl, "pad_t": pt,
                         "th": r.th, "tw": r.tw, "school": stem.split("_")[0]}
                out = r(blob)
                dets, note = decode(out, fam, meta, infer, img.shape)
                title = f"[{idx:02d}] {year} {fam} {r.tw}x{r.th}" + (f"   [!! {note}]" if note else "")
                canvas, kept = draw(img, dets, title, args.conf)
                name = f"{tag}__{year}_{stem}.jpg"
                cv2.imwrite(os.path.join(OUT_DIR, name), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                best = max((d[4] for d in kept), default=0.0)
                print(f"  [{idx:2d}] {fam:<9} {stem[:52]:<52} dets={len(kept):2d} best={best:.3f}"
                      + (f"  top: id={kept[0][5]} {kept[0][6]} {kept[0][4]:.2f}" if kept else "")
                      + (f"  <<{note[:40]}>>" if note else ""))
                per_img.append({"model": f"{year}_{stem}", "family": fam, "note": note,
                                "dets": len(kept), "best": round(best, 4),
                                "top": (None if not kept else
                                        {"id": kept[0][5], "name": kept[0][6],
                                         "conf": round(kept[0][4], 4),
                                         "color": (COLOR_NAMES[kept[0][7]]
                                                   if kept[0][7] >= 0 else None),
                                         "kpts": [[round(float(p[0]), 1), round(float(p[1]), 1)]
                                                  for p in kept[0][8]]}),
                                "file": name})
            except Exception as exc:  # noqa: BLE001
                print(f"  [{idx:2d}] {stem[:52]:<52} ERR {type(exc).__name__}: {str(exc)[:70]}")
                per_img.append({"model": f"{year}_{stem}", "family": fam, "error": str(exc)[:120]})
        summary.append({"image": img_path, "models": per_img})

    sum_path = os.path.join(ROOT, "data", "armor_query", "summary.json")
    merged: dict[str, dict] = {}
    if os.path.exists(sum_path):                     # 累加，避免覆盖历史批次
        try:
            with open(sum_path, encoding="utf-8") as fh:
                for e in json.load(fh):
                    merged[e["image"]] = e
        except Exception:  # noqa: BLE001
            pass
    for e in summary:
        merged[e["image"]] = e
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(sorted(merged.values(), key=lambda x: x["image"]), fh,
                  ensure_ascii=False, indent=2)
    print(f"\n结果图 -> {os.path.relpath(OUT_DIR, ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
