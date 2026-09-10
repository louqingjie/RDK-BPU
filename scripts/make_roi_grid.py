#!/usr/bin/env python3
"""生成 ROI 矩阵图：一行一个模型，一列一张输入图，格内只放装甲板 ROI。

要点：
- 像素来源是**原图**（不裁源结果图），避免烧进图里的标签横幅盖住装甲板
- 四点角点由脚本按 summary.json 的 kpts 自行重绘，缩放后依然清晰
- 行首用中文字体渲染模型名（wqy-microhei）
- 列首为输入图名、真值框数与 ROI 尺寸
- 绿框 = 检出，灰框 = 未检出，橙 = 模型不可用(N/A)
- ROI 锚定「主目标」（面积最大的真值框）；若某模型的四点框不能完整落入，
  该格改以它自身为中心裁切并标 *

用法：python3 scripts/make_roi_grid.py [--cell 228] [--rows-per-sheet 12] [--margin 1.9]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "armor_query")
RES = os.path.join(DIR, "results")
FONT_PATH = os.path.join(DIR, "fonts", "wqy-microhei.ttc")
FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

KPT_FILL = (255, 232, 0)
KPT_EDGE = (0, 0, 0)
QUAD_COLOR = (0, 255, 128)
QUAD_SHIFT = (170, 255, 190)


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in FONT_CACHE:
        FONT_CACHE[size] = ImageFont.truetype(FONT_PATH, size)
    return FONT_CACHE[size]


def tag_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0][:8]


def gt_boxes(img_path: str) -> list[tuple[float, float, float, float]]:
    txt = os.path.splitext(img_path)[0] + ".txt"
    if not os.path.exists(txt):
        return []
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    out = []
    for line in open(txt, encoding="utf-8"):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append(((cx - w / 2) * W, (cy - h / 2) * H,
                    (cx + w / 2) * W, (cy + h / 2) * H))
    return out


def box_of(pts) -> tuple[float, float, float, float]:
    a = np.array(pts, np.float32)
    return (float(a[:, 0].min()), float(a[:, 1].min()),
            float(a[:, 0].max()), float(a[:, 1].max()))


def center_crop(shape, cx: float, cy: float, side: float):
    """以 (cx, cy) 为中心取 4:3 窗口并裁剪到图内。"""
    H, W = shape[:2]
    side = float(min(max(side, 24.0), min(W, H * 4 / 3)))
    x0 = int(max(0, min(cx - side / 2, W - side)))
    y0 = int(max(0, min(cy - side * 3 / 8, H - side * 3 / 4)))
    return x0, y0, int(min(W, x0 + side)), int(min(H, y0 + side * 3 / 4))


def roi_of(entry: dict, margin: float = 1.9):
    """锚定「主目标」而非并集：有真值取面积最大的框；无真值取各模型检出的中位框。"""
    img = cv2.imread(entry["image"])
    gt = gt_boxes(entry["image"])
    if gt:
        b = max(gt, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
    else:
        det = [box_of(m["top"]["kpts"]) for m in entry["models"]
               if (m.get("top") or {}).get("kpts")]
        if not det:
            return None
        b = np.median(np.array(det, np.float32), axis=0)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    side = max(b[2] - b[0], (b[3] - b[1]) * 4 / 3) * margin
    return center_crop(img.shape, cx, cy, side)


def draw_kpts(cell: Image.Image, kpts, roi, cw: int, ch: int, color) -> None:
    """把四点框按 ROI→cell 的仿射缩放到格内重绘（角点 + 边框）。"""
    sx = cw / max(roi[2] - roi[0], 1)
    sy = ch / max(roi[3] - roi[1], 1)
    pts = [((x - roi[0]) * sx, (y - roi[1]) * sy) for x, y in kpts]
    dc = ImageDraw.Draw(cell)
    dc.line(pts + [pts[0]], fill=color, width=2, joint="curve")
    r = max(4, cw // 45)
    for px, py in pts:
        dc.ellipse([px - r, py - r, px + r, py + r], fill=KPT_FILL,
                   outline=KPT_EDGE, width=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, default=228)
    ap.add_argument("--rows-per-sheet", type=int, default=12)
    ap.add_argument("--margin", type=float, default=1.9)
    args = ap.parse_args()
    cw = args.cell
    ch = int(cw * 3 / 4)

    summary = json.load(open(os.path.join(DIR, "summary.json"), encoding="utf-8"))
    summary = [e for e in summary if e["models"]]
    tags = [tag_of(e["image"]) for e in summary]
    rois = {tag_of(e["image"]): roi_of(e, args.margin) for e in summary}
    fulls = {t: cv2.imread(e["image"]) for t, e in zip(tags, summary)}

    order: list[str] = []
    for e in summary:
        for m in e["models"]:
            if m["model"] not in order:
                order.append(m["model"])
    print(f"{len(order)} 个模型 × {len(tags)} 张图；ROI(margin={args.margin}):")
    for t in tags:
        r = rois.get(t)
        print(f"   {t}: {r}  {r[2] - r[0]}x{r[3] - r[1]}" if r else f"   {t}: None")

    label_w, header_h, gap, pad = 470, 84, 6, 12
    total_w = pad * 2 + label_w + gap + len(tags) * (cw + gap)
    rows_per = args.rows_per_sheet
    chunks = [order[i:i + rows_per] for i in range(0, len(order), rows_per)]

    for ci, chunk in enumerate(chunks, 1):
        row_h = ch + 34
        total_h = pad * 2 + header_h + gap + len(chunk) * (row_h + gap)
        sheet = Image.new("RGB", (total_w, total_h), (24, 24, 28))
        d = ImageDraw.Draw(sheet)

        d.text((pad, pad + 8), "模型 \\ 图片", font=font(24), fill=(255, 220, 80))
        d.text((pad, pad + 42), "* = 该模型四点框非主目标，格内改以它为中心裁切",
               font=font(15), fill=(200, 200, 200))
        for j, tg in enumerate(tags):
            x = pad + label_w + gap + j * (cw + gap)
            r = rois.get(tg)
            d.text((x + 4, pad + 6), tg, font=font(20), fill=(255, 255, 255))
            info = f"GT={len(gt_boxes(summary[j]['image']))}框"
            if r:
                info += f" ROI {r[2] - r[0]}×{r[3] - r[1]}"
            d.text((x + 4, pad + 38), info, font=font(16), fill=(160, 200, 255))

        for i, model in enumerate(chunk):
            y = pad + header_h + gap + i * (row_h + gap)
            year, rest = model.split("_", 1)
            d.text((pad + 4, y + 4), f"[{order.index(model) + 1:02d}] {year}",
                   font=font(21), fill=(255, 220, 80))
            parts = rest.split("_")
            d.text((pad + 4, y + 32), "_".join(parts[:2]), font=font(19),
                   fill=(235, 235, 235))
            d.text((pad + 4, y + 58), "_".join(parts[2:])[:26], font=font(17),
                   fill=(190, 190, 190))

            for j, tg in enumerate(tags):
                x = pad + label_w + gap + j * (cw + gap)
                m = next((mm for mm in summary[j]["models"] if mm["model"] == model), {})
                t = m.get("top") or {}
                kpts = t.get("kpts") or []
                roi = rois.get(tg)
                mark = ""
                if kpts and roi:
                    b = box_of(kpts)
                    if not (roi[0] <= b[0] and roi[1] <= b[1]
                            and b[2] <= roi[2] and b[3] <= roi[3]):
                        # 四点框不能完整落入主目标 ROI：改以它自身为中心
                        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        roi = center_crop(fulls[tg].shape, cx, cy,
                                          max(b[2] - b[0], (b[3] - b[1]) * 4 / 3) * 2.6)
                        mark = " *"
                cell = Image.new("RGB", (cw, ch), (40, 40, 44))
                if roi:
                    x0, y0, x1, y1 = roi
                    crop = fulls[tg][y0:y1, x0:x1]
                    if crop.size:
                        cell = Image.fromarray(cv2.cvtColor(
                            cv2.resize(crop, (cw, ch)), cv2.COLOR_BGR2RGB))
                if kpts:
                    draw_kpts(cell, kpts, roi, cw, ch,
                              QUAD_SHIFT if mark else QUAD_COLOR)
                dc = ImageDraw.Draw(cell)
                if kpts:
                    txt = f"id={t.get('id')} {t.get('name')} {t.get('conf'):.2f}"
                    if t.get("color"):
                        txt += f" {t['color']}"
                    col = (0, 255, 128) if not mark else (170, 255, 190)
                elif m.get("note"):
                    txt, col = "N/A", (255, 170, 0)
                else:
                    txt, col = "NONE", (150, 150, 150)
                dc.rectangle([0, ch - 26, cw, ch], fill=(0, 0, 0))
                dc.text((4, ch - 24), txt + mark, font=font(16), fill=col)
                sheet.paste(cell, (x, y))
                d.rectangle([x - 1, y - 1, x + cw, y + ch],
                            outline=(0, 210, 0) if kpts else (95, 95, 95), width=2)

        out = os.path.join(DIR, f"roi_grid_{ci}.jpg")
        sheet.save(out, quality=90)
        print(f"已生成 {os.path.relpath(out, ROOT)}  {sheet.size[0]}x{sheet.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
