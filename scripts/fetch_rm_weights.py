#!/usr/bin/env python3
"""从各 RoboMaster 自瞄/视觉开源仓库抓取公开的模型权重文件。

设计要点：
- 只下载「权重文件」，不克隆整个仓库（避免把数据集/构建产物拉回来）。
- 仓库分支通过 `git ls-remote --symref` 解析，不消耗 GitHub API 配额。
- 输出目录保持「<host>/<owner>/<repo>/<仓库内原始路径>」，便于溯源。
- 每个文件记录 md5/sha256/大小/来源 URL 到 MANIFEST.md 与 manifest.json。
- 单文件超过 MAX_BYTES 的（主要是 TensorRT `.engine`、AX650 `.axmodel` 等
  与硬件强绑定的编译产物）默认跳过并记录原因，可用 --include-large 强制下载。

用法：
    python3 scripts/fetch_rm_weights.py                # 下载（跳过超限文件）
    python3 scripts/fetch_rm_weights.py --include-large
    python3 scripts/fetch_rm_weights.py --only wust     # 只跑名字含 wust 的仓库
    python3 scripts/fetch_rm_weights.py --list          # 只列出计划下载的条目
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "rm_weights")
MAX_BYTES = 100 * 1024 * 1024  # 100 MB，与 GitHub raw 单文件上限一致

UA = {"User-Agent": "rm-weights-collector/1.0 (+technical-exchange)"}

# ---------------------------------------------------------------- 仓库清单
# 路径均来自各仓库 git tree 的实测枚举结果（2026-09-10）。
# 仅收录「模型权重」类文件：onnx / pt / xml+bin(OpenVINO IR) / axmodel /
# engine / ncnn param+bin / caffemodel / darknet weights / npy 等。
GITHUB_REPOS: dict[str, list[str]] = {
    # ---- 深圳大学 RobotPilots ----
    "TOL0VE/RP24-DectionModel": ["0708.onnx"],
    "broalantaps/RobotDetectionModel": ["Model/0526.onnx", "Model/0708.onnx"],
    # ---- 同济大学 SuperPower（自瞄主模型 + 打符 + 分类器）----
    "TongjiSuperPower/sp_vision_25": [
        "assets/best2-sim.onnx", "assets/tiny_resnet.onnx",
        "assets/yolo11.xml", "assets/yolo11.bin",
        "assets/yolo11_buff_int8.xml", "assets/yolo11_buff_int8.bin",
        "assets/yolov5.xml", "assets/yolov5.bin",
        "assets/yolov8.xml", "assets/yolov8.bin",
    ],
    # ---- 同源/衍生仓库（同济框架的搬运版，权重同名）----
    "GKD-RM-Lab/gkd_vision_26": [
        "assets/best2-sim.onnx", "assets/tiny_resnet.onnx",
        "assets/yolo11.xml", "assets/yolo11.bin",
        "assets/yolo11_buff_int8.xml", "assets/yolo11_buff_int8.bin",
        "assets/yolov5.xml", "assets/yolov5.bin",
        "assets/yolov8.xml", "assets/yolov8.bin",
    ],
    "SHM-white/RM2026-AutoAim": [
        "assets/best2-sim.onnx", "assets/tiny_resnet.onnx",
        "assets/yolo11.xml", "assets/yolo11.bin",
        "assets/yolo11_buff_int8.xml", "assets/yolo11_buff_int8.bin",
        "assets/yolov5.xml", "assets/yolov5.bin",
        "assets/yolov8.xml", "assets/yolov8.bin",
    ],
    "Justin186/vision_assessment_2026": [
        "assets/0526.bin", "assets/0526.xml", "assets/tiny_resnet.onnx",
    ],
    # ---- 上海科技大学 magician ----
    "Astra-Whale/SHtech_auto_aim": [
        "asset/models/SKD250526.axmodel",
        "asset/models/SKD250526.onnx",
        "asset/models/SZU0526_fp32input_512x640_nopre_fixoutput.onnx",
    ],
    # ---- 北京科技大学 Reborn ----
    "RebornVision/Reborn-Vision-2024-armor-Inference": [
        "model/number_classifier.onnx",
        "model/mobilenetv3_last_int_all_new/last.bin",
        "model/mobilenetv3_last_int_all_new/last.xml",
    ],
    "gaoxinstudent/RM-Armor-Detect": [
        "ultralytics/runs/pose/train/weights/best.pt",
        "ultralytics/runs/pose/train/weights/last.pt",
        "yolov8n-pose.pt", "yolov8n.pt",
    ],
    # ---- 中国石油大学（北京）SPR ----
    "SPR-Algorithm/SPR-Vision-2025": [
        "Main_ws/src/rm_auto_aim/armor_detector_classic/model/lenet.onnx",
        "Main_ws/src/rm_auto_aim/armor_detector_classic/model/mlp.onnx",
        "Main_ws/src/rm_auto_aim/armor_detector_network/model/0708.onnx",
        "Main_ws/src/rm_auto_aim/armor_detector_network/model/0708.bin",
        "Main_ws/src/rm_auto_aim/armor_detector_network/model/0708.xml",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune.onnx",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune.bin",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune.xml",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune_3.6m.onnx",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune_3.6m.bin",
        "Main_ws/src/rm_rune/rune_detector/model/yolox_rune_3.6m.xml",
    ],
    # ---- 中南大学 FYT ----
    "CSU-FYT-Vision/FYT2024_vision": [
        "rm_auto_aim/armor_detector/model/lenet.onnx",
        "rm_auto_aim/armor_detector/model/mlp.onnx",
        "rm_rune/rune_detector/model/yolox_rune.onnx",
        "rm_rune/rune_detector/model/yolox_rune.bin",
        "rm_rune/rune_detector/model/yolox_rune.xml",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.onnx",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.bin",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.xml",
    ],
    # ---- rm_vision 系（华南师范大学 chenjunnn 血统，分类器 mlp）----
    "chenjunnn/rm_auto_aim": ["armor_detector/model/mlp.onnx"],
    "LihanChen2004/rm_auto_aim": ["armor_detector/model/mlp.onnx"],
    "FaterYU/rm_auto_aim": ["armor_detector/model/mlp.onnx"],
    "HDU-PHOENIX/rm_auto_aim": [
        "armor_auto_aim/armor_detector/model/mlp.onnx",
        "rune_auto_aim/rune_detector/model/yolox.onnx",
        "rune_auto_aim/rune_detector/model/yolox_fp16.onnx",
        "rune_auto_aim/rune_detector/model/2023/yolox.onnx",
        "rune_auto_aim/rune_detector/model/2023/yolox.bin",
        "rune_auto_aim/rune_detector/model/2023/yolox.xml",
    ],
    "0X5A0X480X52/hfut_rm_auto_aim_ws": [
        "src/rm_auto_aim/armor_detector/model/lenet.onnx",
        "src/rm_auto_aim/armor_detector/model/mlp.onnx",
    ],
    # ---- RPS 战队（前哨站/步兵自瞄）----
    "aiyo472/auto_aim_rps26": [
        "src/rm_auto_aim/armor_detector/model/516.onnx",
        "src/rm_auto_aim/armor_detector/model/aim0731.onnx",
        "src/rm_auto_aim/armor_detector/model/aims0717.onnx",
        "src/rm_auto_aim/armor_detector/model/armor_classifier.onnx",
        "src/rm_auto_aim/armor_detector/model/lenet.onnx",
        "src/rm_auto_aim/armor_detector/model/mlp.onnx",
    ],
    # ---- 传统 + LeNet 系 ----
    "hxfhxy/RoboMaster_vision_Armor": [
        "src/cpp08_armor_detector/model/0526.onnx",
        "src/cpp08_armor_detector/model/Zenet-已训练好.onnx",
    ],
    "ecjtu-cx/ECJTU_RM_Vision": ["model/fc.onnx", "model/mlp.onnx"],
    "wildwolf-team/WolfVision": ["module/ml/mnist-8.onnx"],
    # ---- 浙江师范大学 浙狮 ----
    "dielivelr/Z_LION_AutoAim2025": ["src/auto_aim/models/yolo.onnx"],
    # ---- 联盟赛/复旦 Alliance ----
    "Alliance-Algorithm/rmcs_auto_aim_v2": [
        "models/shenzhen-0526.onnx",
        "models/shenzhen-0708.onnx",
        "models/tongji-yolov5.xml",
        "models/tongji-yolov5.bin",
    ],
    # ---- 华中科技大学（AX650 部署 + 多种量化版本）----
    "Blackjack200/talos_26": [
        "models/best.onnx", "models/best_fp16.onnx",
        "models/lenet.onnx", "models/mlp.onnx", "models/mlp_finetuned.onnx",
        "models/opt-1208-001.onnx", "models/tiny_resnet.onnx",
        "models/best_fp16_sm87_fp16.engine",
        "models/compiled.axmodel", "models/qat_u16_all.axmodel",
        "models/qat_u8u16.axmodel", "models/u16_attn.axmodel",
        "models/the_most_expected_one.axmodel",
    ],
    # ---- 武汉科技大学 崇实（模型最全：检测/分类/雷达/量化）----
    "WUST-RM/awakening": [
        "model/0526.onnx", "model/0708.onnx", "model/cb_rune.onnx",
        "model/lenet.onnx", "model/mlp.onnx", "model/mlp_finetuned.onnx",
        "model/opt-0527-001.onnx", "model/opt-1208-001.onnx",
        "model/opt-1208-001.bin", "model/opt-1208-001.param",
        "model/opt_1208_001.ncnn.bin", "model/opt_1208_001.ncnn.param",
        "model/praysky_c2psa_e2e_0228_576x768.onnx",
        "model/praysky_c2psa_e2e_0228_640x640.onnx",
        "model/reborn_number_classifier.onnx", "model/tiny_resnet.onnx",
        "src/tasks/radar_detect/data/Horizon-Armor-yolov8n-ghost-p2-2026-04-04.onnx",
        "src/tasks/radar_detect/data/Horizon-Car-yolov8s-ghost-p2-2026-04-06.onnx",
        "model/0526.engine", "model/0708.engine", "model/cb_rune.engine",
        "model/opt-1208-001.engine", "model/reborn_number_classifier.engine",
        "model/praysky_c2psa_e2e_0228_576x768.engine",
        "model/praysky_c2psa_e2e_0228_640x640.engine",
        "src/tasks/radar_detect/data/Horizon-Armor-yolov8n-ghost-p2-2026-04-04.engine",
        "src/tasks/radar_detect/data/Horizon-Car-yolov8s-ghost-p2-2026-04-06.engine",
    ],
    # ---- 西南石油 Total3DAutoAim ----
    "soloplayl/Total3DAutoAim_md": [
        "c++_AutoAim_example/example_AutoAim/ir_pred_model/pred_model.bin",
        "c++_AutoAim_example/example_AutoAim/ir_pred_model/pred_model.xml",
    ],
    # ---- 早期 Caffe / Darknet 权重（yarkable，深大 RP 传统方案）----
    "yarkable/RP_Infantry_Plus": [
        "extraFile/caffemodel/armornet_iter_200000.caffemodel",
        "extraFile/caffemodel/lenet_iter_200000.caffemodel",
        "extraFile/Rune/lenet/lenet_iter_80000.caffemodel",
        "extraFile/Rune/lenet/lenet_iter_80000_旋转后.caffemodel",
        "extraFile/Rune/lenet/lenet_iter_80000加了负样本.caffemodel",
        "extraFile/Rune/yolo/tiny-yolov2-trial3-noBatch_235000.weights",
        "extraFile/Rune/yolo/tiny-yolov2-trial3-noBatch_80000.weights",
        "extraFile/Rune/yolo/单通道/tiny-yolov2-trial3-noBatch_80000.weights",
    ],
}

GITEE_REPOS: dict[str, list[str]] = {
    # ---- 中国科学院大学 SAS ----
    "ucas-sas-robot-team/RMVision-2026": [
        "src/rm_auto_aim/armor_detector/model/mlp.onnx",
        "src/rm_auto_aim/rm4p_2026/model/yolo11-012345-x.onnx",
        "src/rm_auto_aim/rm4p_inference/model/model-opt-4.onnx",
    ],
    "ucas-sas-robot-team/RMVision-RM4P-2025": [
        "src/rm_auto_aim/armor_detector/model/mlp.onnx",
        "src/rm_auto_aim/rm4p_inference/model/model-opt-4.onnx",
    ],
    # ---- 装甲板数字识别（PyTorch）----
    "starrysky9959/Digital-recognition": [
        "C++模型部署/model/model.pt",
        "libtorch_model/model.pt",
        "model_param/state_dict.pt",
    ],
}

# ================================================================
# 第二轮：2026-09-10 从官方论坛（bbs.robomaster.com）/ RM Search 检索补充
# 来源帖子见 data/rm_weights/BBS_SOURCES.md
# ================================================================
GITHUB_REPOS.update({
    # 深圳大学 RM2026：视觉模型统一部署库（V5/V8 装甲四点 + 能量机关五点）
    "SZURPVision/26_NNDeployment_Lib_and_Detection_Models": [
        "所有模型/onnx/Infantry-v8n-fp16-20260726-D1.8w-B16.onnx",
        "所有模型/onnx/Rune-v8n-fp16-20260624.onnx",
        "所有模型/openvino/Infantry-v5n-fp16-20250725/Infantry-v5n-Release-20260725.xml",
        "所有模型/openvino/Infantry-v5n-fp16-20250725/Infantry-v5n-Release-20260725.bin",
        "所有模型/openvino/Infantry-v8n-fp16-20260726-D1.8w-B16/Infantry-v8n-fp16-20260726-D1.8w-B16.xml",
        "所有模型/openvino/Infantry-v8n-fp16-20260726-D1.8w-B16/Infantry-v8n-fp16-20260726-D1.8w-B16.bin",
        "所有模型/openvino/Rune-v8n-fp16-20260624-D14367-B16/Rune-v8n-fp16-20260624-D14367-B16.xml",
        "所有模型/openvino/Rune-v8n-fp16-20260624-D14367-B16/Rune-v8n-fp16-20260624-D14367-B16.bin",
    ],
    "SZURPVision/RuneDetectionModel": ["model/model-0624.onnx"],
    # 仲恺农业工程学院 奇点战队
    "NOMANE-0/QD_Vision2026": [
        "src/rm_auto_aim/armor_detector/model/IR/0526.xml",
        "src/rm_auto_aim/armor_detector/model/IR/0526.bin",
        "src/rm_auto_aim/armor_detector/model/lenet.onnx",
        "src/rm_auto_aim/armor_detector/model/mlp.onnx",
        "src/rm_rune/rune_detector/model/yolox_rune.onnx",
        "src/rm_rune/rune_detector/model/yolox_rune.xml",
        "src/rm_rune/rune_detector/model/yolox_rune.bin",
        "src/rm_rune/rune_detector/model/yolox_rune_3.6m.onnx",
        "src/rm_rune/rune_detector/model/yolox_rune_3.6m.xml",
        "src/rm_rune/rune_detector/model/yolox_rune_3.6m.bin",
    ],
    # 吉林大学 TARS Go
    "Fskaaaaaaaa/jlu_vision_26": ["assets/0526.onnx", "assets/yolox_rune_3.6m.onnx"],
    # 浙江大学 Hello World（能量机关）
    "IC-Alan/HWauto_buff2026": ["buff.onnx"],
    # 常州大学 Climber
    "CCZU-Climber/Climber_Vision_26": [
        "assets/best.onnx", "assets/tiny_resnet.onnx",
        "assets/buff_repvgg.xml", "assets/buff_repvgg.bin",
        "assets/yolo11.xml", "assets/yolo11.bin",
        "assets/yolov5.xml", "assets/yolov5.bin",
        "assets/yolov8.xml", "assets/yolov8.bin",
    ],
    # 复旦大学 星云 EGA（雷达视觉 + 激光雷达）
    "Ryaxwn7/EGA_Radar_Algorithm": [
        "Main_ws/weights/MobileNetv2.2.pth",
        "Main_ws/weights/armor/armor_best.onnx",
        "Main_ws/weights/armor/armor_best.pt",
        "Main_ws/weights/armor/armor_best.engine",
        "Main_ws/weights/car/best.onnx",
        "Main_ws/weights/car/best.pt",
        "Main_ws/weights/car/best.engine",
    ],
    # 江南大学 SHARK（雷达）
    "JNU-SHARK/shark-radar-vision": [
        "models/armor.onnx", "models/armor.pt", "models/armor.engine",
        "models/armor_batch.engine",
        "models/car.onnx", "models/car.pt", "models/car.engine",
    ],
    # 西北工业大学 WMJ（雷达反制）
    "zplszz/WMJRadar": ["anti_drone/wmj_anti_drone.pt"],
    # 东南大学：神经网络压缩低码率图传
    "ElainaXD/rm_compress": [
        "rm_qvrf_receiver/models/qvrf_gs_rlfn_x2_192.onnx",
        "rm_qvrf_receiver/models/qvrf_gs_rlfn_x2_448.onnx",
        "rm_qvrf_receiver/models/msssim_g_s_fp32.xml",
        "rm_qvrf_receiver/models/msssim_g_s_fp32.bin",
        "rm_qvrf_receiver/models/msssim_h_a_fp32.xml",
        "rm_qvrf_receiver/models/msssim_h_a_fp32.bin",
        "rm_qvrf_receiver/models/msssim_h_s_fp32.xml",
        "rm_qvrf_receiver/models/msssim_h_s_fp32.bin",
        "rm_qvrf_receiver/models/realesr-general-x4v3.pth",
    ],
    # 深圳职业技术大学 RCIA
    "BenmaoNeko/RCIA_Benmao_Vision": ["models/yolo/best.pt", "models/yolo/best-v2.pt"],
    # 中南大学 FYT 雷达
    "baiyeweiguang/FYT2024_radar": [
        "radar_detector/model/armor.onnx", "radar_detector/model/armor.pt",
        "radar_detector/model/armor_new2.pt",
        "radar_detector/model/car.onnx", "radar_detector/model/car.pt",
    ],
    # 华中科技大学 狼牙：英雄视觉（帖子 41998）
    "HUSTLYRM/HUST_HeroAim_2024": [
        "src/utils/model/best_06_02.onnx",
        "src/utils/model/best_06_02.xml",
        "src/utils/model/best_06_02.bin",
        "src/utils/model/outpost_nano_100epoch_1_8.onnx",
        "src/utils/model/0715_v10_416n_openvino_model/0715_v10_416n.xml",
        "src/utils/model/0715_v10_416n_openvino_model/0715_v10_416n.bin",
        "src/utils/model/0801_384nano_green_openvino_model/0801_384nano_green.xml",
        "src/utils/model/0801_384nano_green_openvino_model/0801_384nano_green.bin",
        "src/utils/model/svm_numbers.xml",
        "src/utils/model/svm_numbers_rbf.xml",
        "src/utils/tools/armor.onnx",
    ],
    # 深圳大学 RM2026 能量机关算法（帖子 1942229）
    "SZURPVision/RP-26Rune": [
        "src/app_plugin/detector/config/openvino/model-0624.onnx",
    ],
})

GITEE_REPOS.update({
    # 厦门理工学院 PFA 单目雷达（车 + 装甲板识别模型）
    "CarryzhangZKY/pfa_vision_radar": [
        "models/armor.onnx",
        "models/car.onnx",
    ],
    # 辽宁科技大学 COD / 中国石油大学（华东）RPS 前哨站自瞄
    "ustl-cod/sentry-auto-aim": [
        "rm_auto_aim/armor_detector/model/mlp.onnx",
        "rm_auto_aim/armor_detector/model/lenet.onnx",
        "rm_rune/rune_detector/model/yolox_rune.onnx",
        "rm_rune/rune_detector/model/yolox_rune.xml",
        "rm_rune/rune_detector/model/yolox_rune.bin",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.onnx",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.xml",
        "rm_rune/rune_detector/model/yolox_rune_3.6m.bin",
    ],
})

# GitHub Release 资产（华北理工大学 Horizon 先行开源的原始 .pt 权重）
RELEASE_ASSETS: dict[str, tuple[str, list[str]]] = {
    "BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025": ("2026.04.24", [
        "Horizon-Armor-yolov8n-ghost-p2-2026-04-04.pt",
        "Horizon-Car-yolov8s-ghost-p2-2026-04-06.pt",
    ]),
}


# ---------------------------------------------------------------- 工具函数
def resolve_branch(host: str, repo: str) -> str:
    """用 git ls-remote 解析默认分支，不消耗 API 配额。"""
    url = f"https://{host}/{repo}.git"
    try:
        out = subprocess.run(
            ["git", "ls-remote", "--symref", url, "HEAD"],
            capture_output=True, text=True, timeout=60, check=True,
        ).stdout
        for line in out.splitlines():
            if line.startswith("ref:") and line.endswith("HEAD"):
                return line.split()[1].removeprefix("refs/heads/")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 解析分支失败 {repo}: {exc}", file=sys.stderr)
    return "main"


def raw_url(host: str, repo: str, branch: str, path: str) -> str:
    if host == "github.com":
        return f"https://raw.githubusercontent.com/{repo}/{branch}/{urllib.parse.quote(path)}"
    if host == "github-release":          # Release 资产，branch 位置传 tag
        return (f"https://github.com/{repo}/releases/download/"
                f"{urllib.parse.quote(branch)}/{urllib.parse.quote(path)}")
    return f"https://gitee.com/{repo}/raw/{branch}/{urllib.parse.quote(path)}"


def http_get(url: str, method: str = "GET", timeout: int = 120):
    req = urllib.request.Request(url, headers=UA, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def remote_size(url: str) -> int | None:
    try:
        with http_get(url, method="HEAD") as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:  # noqa: BLE001
        return None


def digest(path: str) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def gitee_api_download(repo: str, branch: str, path: str, tmp: str) -> bool:
    """Gitee raw 对大文件返回 403，退回官方 API 的 base64 内容。

    返回 True 表示内容完整；False 表示 API 响应被截断（Gitee 对 >10MB
    响应有长度上限），需要改用 git 浅克隆。
    """
    api = (f"https://gitee.com/api/v5/repos/{repo}/contents/"
           f"{urllib.parse.quote(path)}?ref={urllib.parse.quote(branch)}")
    with http_get(api) as resp:
        payload = json.load(resp)
    if not isinstance(payload, dict) or "content" not in payload:
        return False                   # 目录/空返回 → 交给浅克隆回退
    data = base64.b64decode(payload["content"])
    if payload.get("size") is not None and len(data) != payload["size"]:
        return False
    with open(tmp, "wb") as fh:
        fh.write(data)
    return True


def gitee_clone_download(repo: str, branch: str, path: str, tmp: str) -> None:
    """Gitee 大文件（raw 需登录 / API 截断）通过浅克隆取回。"""
    workdir = tempfile.mkdtemp(prefix="rm_weights_")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch,
             f"https://gitee.com/{repo}.git", workdir],
            check=True, capture_output=True, text=True, timeout=3600,
        )
        shutil.copyfile(os.path.join(workdir, *path.split("/")), tmp)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def fetch_one(host: str, repo: str, branch: str, path: str,
              include_large: bool) -> dict:
    url = raw_url(host, repo, branch, path)
    dest = os.path.join(OUT_DIR, host, repo, *path.split("/"))
    rec = {"host": host, "repo": repo, "path": path, "url": url,
           "dest": os.path.relpath(dest, ROOT), "status": "pending"}

    size = remote_size(url)
    if size is not None and size > MAX_BYTES and not include_large:
        rec.update(status="skipped-large", bytes=size,
                   note=f">{MAX_BYTES // (1 << 20)}MB，硬件相关编译产物，用 --include-large 可下载")
        return rec

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        try:
            with http_get(url) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except urllib.error.HTTPError as exc:
            # Gitee raw 对大体积文件返回 403，退回官方 API / 浅克隆
            if host != "gitee.com" or exc.code not in (403, 404):
                raise
            if not gitee_api_download(repo, branch, path, tmp):
                gitee_clone_download(repo, branch, path, tmp)
        os.replace(tmp, dest)
        md5, sha = digest(dest)
        rec.update(status="ok", bytes=os.path.getsize(dest), md5=md5, sha256=sha)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        rec.update(status="error", note=str(exc))
    return rec


def human_size(n: int | None) -> str:
    if n is None:
        return "-"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def write_manifest_md(records: list[dict], path: str) -> None:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(f"{r['host']}/{r['repo']}", []).append(r)
    ok = [r for r in records if r["status"] == "ok"]
    total = sum(r.get("bytes") or 0 for r in ok)
    lines = [
        "# RoboMaster 自瞄 / 视觉开源权重清单",
        "",
        "> 本文件由 `scripts/fetch_rm_weights.py --manifest-only` 自动生成，请勿手工编辑。",
        f"> 共 {len(records)} 个条目：成功 {len(ok)} 个，合计约 {human_size(total)}。",
        "> 逐文件 sha256 见同目录 `manifest.json`。",
        "",
        "## 一、仓库汇总",
        "",
        "| 平台/仓库 | 文件数 | 大小 |",
        "|---|---|---|",
    ]
    for name in sorted(groups):
        items = groups[name]
        size = sum(i.get("bytes") or 0 for i in items if i["status"] == "ok")
        lines.append(f"| {name} | {len(items)} | {human_size(size)} |")
    lines += ["", "## 二、逐文件明细", ""]
    for name in sorted(groups):
        lines += [f"### {name}", "",
                  "| 仓库内路径 | 大小 | md5 | 状态 |", "|---|---|---|---|"]
        for r in sorted(groups[name], key=lambda x: x["path"]):
            mark = {"ok": "✅", "skipped-large": "⏭️"}.get(r["status"], "❌")
            note = r.get("note", "")
            lines.append(f"| `{r['path']}` | {human_size(r.get('bytes'))} | "
                         f"`{r.get('md5', '-')}` | {mark} {note} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-large", action="store_true",
                    help="下载超过 100MB 的文件（TensorRT engine / axmodel）")
    ap.add_argument("--only", default="",
                    help="仅处理名字包含这些子串的仓库（逗号分隔）")
    ap.add_argument("--list", action="store_true", help="只列出计划条目")
    ap.add_argument("--manifest-only", action="store_true",
                    help="不下载，仅由 manifest.json 重新生成 MANIFEST.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if args.manifest_only:
        src = os.path.join(OUT_DIR, "manifest.json")
        with open(src, encoding="utf-8") as fh:
            recs = json.load(fh)
        dst = os.path.join(OUT_DIR, "MANIFEST.md")
        write_manifest_md(recs, dst)
        print(f"已生成 {os.path.relpath(dst, ROOT)}（{len(recs)} 条）")
        return 0

    targets: list[tuple[str, str, list[str]]] = []
    for repo, paths in GITHUB_REPOS.items():
        targets.append(("github.com", repo, paths))
    for repo, paths in GITEE_REPOS.items():
        targets.append(("gitee.com", repo, paths))
    keys = [k.strip().lower() for k in args.only.split(",") if k.strip()]
    rel_jobs = [("github-release", repo, tag, n)
                for repo, (tag, names) in RELEASE_ASSETS.items() for n in names]
    if keys:
        targets = [t for t in targets if any(k in t[1].lower() for k in keys)]
        rel_jobs = [j for j in rel_jobs if any(k in j[1].lower() for k in keys)]

    print(f"计划抓取 {len(targets)} 个仓库 / "
          f"{sum(len(p) for _, _, p in targets)} 个文件 → {OUT_DIR}")
    if args.list:
        for host, repo, paths in targets:
            for p in paths:
                print(f"  {host}/{repo}/{p}")
        return 0

    branches = {repo: resolve_branch(host, repo) for host, repo, _ in targets}
    for repo, br in branches.items():
        print(f"  {repo} @ {br}")

    jobs = [(host, repo, branches[repo], p)
            for host, repo, paths in targets for p in paths]
    jobs += rel_jobs
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, *j, args.include_large): j for j in jobs}
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            mark = {"ok": "OK ", "error": "ERR", "skipped-large": "SKP"}[rec["status"]]
            print(f"  [{mark}] {rec['repo']}/{rec['path']} "
                  f"({rec.get('bytes', '?')} B) {rec.get('note', '')}")

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    merged: dict[tuple[str, str, str], dict] = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                for r in json.load(fh):
                    merged[(r["host"], r["repo"], r["path"])] = r
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 读取旧清单失败，将重建：{exc}", file=sys.stderr)
    for r in records:
        merged[(r["host"], r["repo"], r["path"])] = r
    all_records = sorted(merged.values(), key=lambda r: (r["host"], r["repo"], r["path"]))
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, ensure_ascii=False, indent=2)

    ok = [r for r in all_records if r["status"] == "ok"]
    bad = [r for r in all_records if r["status"] != "ok"]
    print(f"\n本次处理 {len(records)} 条；清单累计 {len(all_records)} 条"
          f"（成功 {len(ok)} / 异常 {len(bad)}）")
    for r in bad:
        print(f"  [!] {r['status']}: {r['repo']}/{r['path']} {r.get('note', '')}")
    write_manifest_md(all_records, os.path.join(OUT_DIR, "MANIFEST.md"))
    print(f"清单：{os.path.relpath(manifest_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
