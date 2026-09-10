#!/usr/bin/env python3
"""把逐模型识别结果拼成对比图，便于横向查看 id / 置信度 / pose 角点。

生成：
  data/armor_query/sheet_<tag>_crop.jpg   各模型结果在装甲板区域的裁剪拼图（带序号）
  data/armor_query/sheet_<tag>_full.jpg   各模型整图缩略拼图
  data/armor_query/SHEET_LEGEND.md        序号 -> 模型 对照表
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "armor_query")
RES = os.path.join(DIR, "results")


def label_tile(tile, idx, txt, ok):
    h, w = tile.shape[:2]
    bar = np.zeros((44, w, 3), np.uint8)
    col = (0, 220, 0) if ok else (90, 90, 90)
    cv2.putText(bar, f"[{idx:02d}] {txt}", (6, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, col, 2, cv2.LINE_AA)
    return np.vstack([bar, tile])


def grid(tiles, cols, tw, th):
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((th + 44, tw, 3), np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> int:
    summary = json.load(open(os.path.join(DIR, "summary.json"), encoding="utf-8"))
    legend = []
    for img in summary:
        tag = os.path.splitext(os.path.basename(img["image"]))[0][:8]
        files = sorted(glob.glob(os.path.join(RES, f"{tag}__*.jpg")))
        if not files:
            continue

        # 用成功检出框的中位数定位装甲板区域
        pts = []
        for m in img["models"]:
            t = m.get("top")
            if t and t.get("kpts"):
                pts += [p for p in t["kpts"]]
        if pts:
            a = np.array(pts, np.float32)
            cx, cy = float(a[:, 0].mean()), float(a[:, 1].mean())
        else:
            cx, cy = 1600.0, 1200.0
        cw, ch = 1150, 880
        x0, y0 = int(cx - cw / 2), int(cy - ch / 2)

        tiles_c, tiles_f = [], []
        for i, f in enumerate(files, 1):
            name = os.path.basename(f)[len(tag) + 2:-4]
            note = next((m.get("note") for m in img["models"] if m["model"] == name), "")
            dets = next((m.get("dets", 0) for m in img["models"] if m["model"] == name), 0)
            best = next((m.get("best", 0) for m in img["models"] if m["model"] == name), 0)
            sc = next((m.get("top") or {} for m in img["models"] if m["model"] == name), {})
            short = (f"id={sc.get('id')} {sc.get('name')} {best:.2f} "
                     f"{sc.get('color') or ''}") if dets else ("NONE" if not note else "N/A")
            full = cv2.imread(f)
            big = cv2.resize(full, (1280, 962))
            tiles_f.append(label_tile(big, i, f"{name[:34]} | {short}", dets > 0))
            crop = full[max(0, y0):y0 + ch, max(0, x0):x0 + cw]
            if crop.size == 0:
                crop = full[:ch, :cw]
            tiles_c.append(label_tile(cv2.resize(crop, (360, 276)), i, short, dets > 0))
            legend.append((tag, i, name, dets, sc))

        cv2.imwrite(os.path.join(DIR, f"sheet_{tag}_crop.jpg"),
                    grid(tiles_c, 6, 360, 276), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(os.path.join(DIR, f"sheet_{tag}_full.jpg"),
                    grid(tiles_f, 2, 1280, 962), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{tag}: crop 拼图 {len(tiles_c)} 格, full 拼图 {len(tiles_f)} 格")

    lines = ["# 拼图序号对照表", ""]
    for tag, i, name, dets, sc in legend:
        t = f"id={sc.get('id')} {sc.get('name')} conf={sc.get('conf')} {sc.get('color') or ''}" \
            if dets else "未识别"
        lines.append(f"- `{tag}` [{i:02d}] **{name}** — {t}")
    open(os.path.join(DIR, "SHEET_LEGEND.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("已写 SHEET_LEGEND.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
