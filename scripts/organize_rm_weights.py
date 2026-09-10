#!/usr/bin/env python3
"""把 data/rm_weights 下抓来的原始权重，按「年份 / 学校 / 网络版本 / Pose / Armor」重组。

命名规则：
    <年份>/<学校>_<网络版本>_<Pose|Det>_<Armor|Rune|Radar|Car|Cls|Solver|General>_<细节>.<ext>

- Pose  : 网络回归关键点（四点/五点/九点），即 RM 常见的「四点模型」；Det = 仅输出检测框
- Armor : 装甲板识别权重；非装甲板用 Rune(打符)/Radar(雷达)/Car(车辆)/Cls(分类)/Solver(解算预测)/General
- 内容完全相同的文件（sha256 一致）只保留一份，全部来源记录在 organized.json 的 sources 字段。

属性判定依据（详见生成的 ORGANIZED.md）：
  1) ONNX 内嵌 Ultralytics 元数据（task / names / kpt_shape / date）
  2) OpenVINO IR 的图层结构与输出节点命名（/model.N/ = YOLOv8 系；/m/model.N/ = 深大 RP 系）
  3) 公开仓库 README 与本仓库既有文档（如深大 RP 为四肢点模型）
  4) 输出通道拆解（4 bbox + conf + color + 4×2 关键点 + num 兵种）

用法：
    python3 scripts/organize_rm_weights.py            # 预览（dry-run，不改动）
    python3 scripts/organize_rm_weights.py --apply    # 实际执行
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "rm_weights")

# --------------------------------------------------------------- 分类表
# key = "<文件名>|<字节数>"（唯一标识一份内容）
# value = (年份, 学校, 网络版本, Pose|Det, 用途, 细节)
C: dict[str, tuple] = {
    # ============ 装甲板检测 / 四点模型 ============
    "shenzhen-0708.onnx|5752110":        (2024, "深圳大学", "YOLOv5", "Pose", "Armor", "0708"),
    "0708.xml|362334":                   (2024, "深圳大学", "YOLOv5", "Pose", "Armor", "0708_SPR版"),
    "tongji-yolov5.xml|349369":          (2024, "深圳大学", "YOLOv5", "Pose", "Armor", "0708_同济版"),
    "tongji-yolov5.bin|2836140":         (2024, "深圳大学", "YOLOv5", "Pose", "Armor", "0708_同济版"),
    "shenzhen-0526.onnx|4359932":        (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "0526"),
    "0526.xml|197586":                   (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "0526_IR"),
    "0526.bin|3934528":                  (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "0526_IR"),
    "SZU0526_fp32input_512x640_nopre_fixoutput.onnx|4212192":
                                         (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "SZU0526_512x640"),
    "SKD250526.onnx|11146248":           (2025, "上海科技大学", "YOLOv5", "Pose", "Armor", "SKD250526_512x640"),
    "SKD250526.axmodel|3016148":         (2025, "上海科技大学", "YOLOv5", "Pose", "Armor", "SKD250526_AX650"),
    "0526.engine|6966564":               (2025, "武汉科技大学", "YOLOv5", "Pose", "Armor", "0526_TensorRT"),
    "0708.engine|6032604":               (2024, "武汉科技大学", "YOLOv5", "Pose", "Armor", "0708_TensorRT"),
    "yolov8.xml|770007":                 (2024, "北京科技大学", "YOLOv8", "Pose", "Armor", "MobileNetV3_last"),
    "yolov8.bin|3623860":                (2024, "北京科技大学", "YOLOv8", "Pose", "Armor", "MobileNetV3_last"),
    "yolo11.xml|709518":                 (2025, "同济大学", "YOLO11", "Pose", "Armor", "yolo11_int8"),
    "yolo11.bin|2922484":                (2025, "同济大学", "YOLO11", "Pose", "Armor", "yolo11_int8"),
    "yolo.onnx|15782065":                (2025, "浙江师范大学", "YOLOv5", "Pose", "Armor", "ZLion2025"),
    "516.onnx|6455925":                  (2026, "RPS战队", "EfficientNet", "Pose", "Armor", "0516_eff4"),
    "aim0731.onnx|7439598":              (2025, "RPS战队", "EfficientNet", "Pose", "Armor", "0731_eff2"),
    "aims0717.onnx|7709527":             (2025, "RPS战队", "EfficientNet", "Pose", "Armor", "0717_eff2"),
    "model-opt-4.onnx|28440153":         (2025, "中国科学院大学", "YOLOv5", "Pose", "Armor", "RM4P_384x640"),
    "yolo11-012345-x.onnx|10985528":     (2026, "中国科学院大学", "YOLO11", "Pose", "Armor", "6兵种_640"),
    "best.pt|27803246":                  (2024, "北京科技大学", "YOLOv8", "Pose", "Armor", "四点_train_best"),
    "last.pt|27803438":                  (2024, "北京科技大学", "YOLOv8", "Pose", "Armor", "四点_train_last"),
    "yolov8n-pose.pt|6828990":           (2024, "通用", "YOLOv8", "Pose", "General", "n-pose_COCO人体姿态预训练"),
    # ---- YOLO26 端到端四点 ----
    "praysky_c2psa_e2e_0228_640x640.onnx|4560874":
                                         (2026, "武汉科技大学", "YOLO26", "Pose", "Armor", "praysky_C2PSA_640"),
    "praysky_c2psa_e2e_0228_576x768.onnx|4582378":
                                         (2026, "武汉科技大学", "YOLO26", "Pose", "Armor", "praysky_C2PSA_576x768"),
    "praysky_c2psa_e2e_0228_640x640.engine|9847540":
                                         (2026, "武汉科技大学", "YOLO26", "Pose", "Armor", "praysky_C2PSA_640_TensorRT"),
    "praysky_c2psa_e2e_0228_576x768.engine|8997860":
                                         (2026, "武汉科技大学", "YOLO26", "Pose", "Armor", "praysky_C2PSA_576x768_TensorRT"),
    "best.onnx|5234914":                 (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_640"),
    "best_fp16.onnx|1493904":            (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_dwconv_fp16"),
    "best_fp16_sm87_fp16.engine|4397932":
                                         (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_sm87_TensorRT"),
    "compiled.axmodel|2462747":          (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_AX650"),
    "qat_u16_all.axmodel|2509878":       (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_QAT_u16_AX650"),
    "qat_u8u16.axmodel|2573996":         (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_QAT_u8u16_AX650"),
    "u16_attn.axmodel|2662852":          (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_u16attn_AX650"),
    "the_most_expected_one.axmodel|2525654":
                                         (2026, "talos战队", "YOLO26", "Pose", "Armor", "best_final_AX650"),
    # ---- YOLOX 系 / 416 ----
    "opt-1208-001.onnx|2167983":         (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416"),
    "opt-1208-001.bin|2128592":          (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416_ncnn"),
    "opt-1208-001.param|25314":          (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416_ncnn"),
    "opt-1208-001.engine|3165476":       (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416_TensorRT"),
    "opt_1208_001.ncnn.bin|1076688":     (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416_ncnn_fp16"),
    "opt_1208_001.ncnn.param|18444":     (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt1208_416_ncnn_fp16"),
    "opt-0527-001.onnx|3664655":         (2026, "武汉科技大学", "YOLOX", "Pose", "Armor", "opt0527_416"),
    # ---- 雷达（bbox 检测，非四点）----
    "Horizon-Armor-yolov8n-ghost-p2-2026-04-04.onnx|9261874":
                                         (2026, "武汉科技大学", "YOLOv8", "Det", "Armor", "雷达_ghost_p2_1280"),
    "Horizon-Armor-yolov8n-ghost-p2-2026-04-04.engine|7816844":
                                         (2026, "武汉科技大学", "YOLOv8", "Det", "Armor", "雷达_ghost_p2_1280_TensorRT"),
    "Horizon-Car-yolov8s-ghost-p2-2026-04-06.onnx|24074138":
                                         (2026, "武汉科技大学", "YOLOv8", "Det", "Car", "雷达车辆_ghost_p2_1280"),
    "Horizon-Car-yolov8s-ghost-p2-2026-04-06.engine|15984548":
                                         (2026, "武汉科技大学", "YOLOv8", "Det", "Car", "雷达车辆_ghost_p2_1280_TensorRT"),

    # ============ 打符（Rune / Buff）============
    "best2-sim.onnx|12547691":           (2025, "同济大学", "YOLOv8", "Pose", "Rune", "5点_best2-sim"),
    "yolo11_buff_int8.xml|696202":       (2025, "同济大学", "YOLO11", "Pose", "Rune", "buff_int8"),
    "yolo11_buff_int8.bin|2914672":      (2025, "同济大学", "YOLO11", "Pose", "Rune", "buff_int8"),
    "cb_rune.onnx|2632872":              (2026, "武汉科技大学", "RepVGG", "Pose", "Rune", "cb_rune_9点"),
    "cb_rune.engine|3968228":            (2026, "武汉科技大学", "RepVGG", "Pose", "Rune", "cb_rune_9点_TensorRT"),
    "yolox_rune.onnx|10821115":          (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_480"),
    "yolox_rune.xml|411708":             (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_480"),
    "yolox_rune.bin|5363648":            (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_480"),
    "yolox_rune_3.6m.onnx|3608024":      (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_3.6m_480"),
    "yolox_rune_3.6m.xml|410237":        (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_3.6m_480"),
    "yolox_rune_3.6m.bin|1770714":       (2024, "中南大学", "YOLOX", "Pose", "Rune", "打符_3.6m_480"),
    "yolox.onnx|3717163":                (2023, "杭州电子科技大学", "YOLOX", "Pose", "Rune", "打符_2023"),
    "yolox.xml|427802":                  (2023, "杭州电子科技大学", "YOLOX", "Pose", "Rune", "打符_2023"),
    "yolox.bin|3613256":                 (2023, "杭州电子科技大学", "YOLOX", "Pose", "Rune", "打符_2023"),
    "yolox.onnx|3597887":                (2023, "杭州电子科技大学", "YOLOX", "Pose", "Rune", "打符_416"),
    "yolox_fp16.onnx|1826844":           (2023, "杭州电子科技大学", "YOLOX", "Pose", "Rune", "打符_416_fp16"),

    # ============ 分类器（数字 / 兵种）============
    "mlp.onnx|314058":                   (2023, "华南师范大学", "MLP", "Det", "Cls", "数字_20x28_9类"),
    "mlp_finetuned.onnx|314058":         (2023, "华南师范大学", "MLP", "Det", "Cls", "数字_20x28_9类_微调"),
    "lenet.onnx|248674":                 (2023, "通用", "LeNet", "Det", "Cls", "数字_28x28_9类"),
    "tiny_resnet.onnx|1105322":          (2025, "同济大学", "ResNet", "Det", "Cls", "数字_32x32_9类"),
    "number_classifier.onnx|117494":     (2024, "北京科技大学", "CNN", "Det", "Cls", "数字_20x28"),
    "reborn_number_classifier.engine|525788":
                                         (2024, "北京科技大学", "CNN", "Det", "Cls", "数字_TensorRT"),
    "armor_classifier.onnx|324096":      (2026, "RPS战队", "MLP", "Det", "Cls", "兵种_28x28_8类"),
    "fc.onnx|664680":                    (2023, "华东交通大学", "FC", "Det", "Cls", "数字_20x28_9类"),
    "mnist-8.onnx|26454":                (2021, "野狼战队", "LeNet", "Det", "Cls", "MNIST_28x28"),
    "Zenet-已训练好.onnx|248678":        (2023, "个人_hxfhxy", "LeNet", "Det", "Cls", "数字_28x28"),
    "model.pt|198674":                   (2020, "个人_starrysky9959", "CNN", "Det", "Cls", "数字识别_权重"),
    "state_dict.pt|178477":              (2020, "个人_starrysky9959", "CNN", "Det", "Cls", "数字识别_state_dict"),

    # ============ 传统 / 早期（Caffe、Darknet）============
    "armornet_iter_200000.caffemodel|1708978":
                                         (2020, "深圳大学", "Caffe-ArmorNet", "Det", "Cls", "装甲数字_200000"),
    "lenet_iter_200000.caffemodel|1721020":
                                         (2020, "深圳大学", "Caffe-LeNet", "Det", "Cls", "装甲数字_200000"),
    "lenet_iter_80000.caffemodel|1708972": (2020, "深圳大学", "Caffe-LeNet", "Det", "Cls", "打符数字_80000"),
    "lenet_iter_80000_旋转后.caffemodel|1708974":
                                         (2020, "深圳大学", "Caffe-LeNet", "Det", "Cls", "打符数字_80000_旋转"),
    "lenet_iter_80000加了负样本.caffemodel|1708974":
                                         (2020, "深圳大学", "Caffe-LeNet", "Det", "Cls", "打符数字_80000_加负样本"),
    "tiny-yolov2-trial3-noBatch_235000.weights|2191628":
                                         (2020, "深圳大学", "tiny-YOLOv2", "Det", "Rune", "打符_235000"),
    "tiny-yolov2-trial3-noBatch_80000.weights|2191628":
                                         (2020, "深圳大学", "tiny-YOLOv2", "Det", "Rune", "打符_80000"),
    "tiny-yolov2-trial3-noBatch_80000.weights|2190476":
                                         (2020, "深圳大学", "tiny-YOLOv2", "Det", "Rune", "打符_80000_单通道"),

    # ============ 解算 / 通用 ============
    "pred_model.xml|275930":             (2025, "西南石油大学", "PredNet", "Det", "Solver", "弹道预测_IR"),
    "pred_model.bin|14602856":           (2025, "西南石油大学", "PredNet", "Det", "Solver", "弹道预测_IR"),
    "yolov8n.pt|6534387":                (2024, "北京科技大学", "YOLOv8", "Det", "General", "n_预训练_非装甲"),

    # ============ 第二轮：官方论坛 / RM Search 补充（2026-09-10）============
    # 华北理工大学 Horizon：雷达 Car/Armor 权重原始 .pt
    "Horizon-Armor-yolov8n-ghost-p2-2026-04-04.pt|4123842":
                                         (2026, "华北理工大学", "YOLOv8", "Det", "Armor", "雷达_ghost_p2_1280_原始"),
    "Horizon-Car-yolov8s-ghost-p2-2026-04-06.pt|11552322":
                                         (2026, "华北理工大学", "YOLOv8", "Det", "Car", "雷达车辆_ghost_p2_1280_原始"),
    # 深圳大学 RM2026 视觉模型统一部署库（V5/V8 四点装甲 + V8 五点打符）
    "Infantry-v8n-fp16-20260726-D1.8w-B16.onnx|6429469":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Armor", "Infantry_v8n_7类_480x640"),
    "Infantry-v8n-fp16-20260726-D1.8w-B16.xml|328947":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Armor", "Infantry_v8n_7类_IR"),
    "Infantry-v8n-fp16-20260726-D1.8w-B16.bin|6344694":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Armor", "Infantry_v8n_7类_IR"),
    "Infantry-v5n-Release-20260725.xml|265631":
                                         (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "Infantry_v5n_20260725_IR"),
    "Infantry-v5n-Release-20260725.bin|3934528":
                                         (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "Infantry_v5n_20260725_IR"),
    "Rune-v8n-fp16-20260624.onnx|11680556":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Rune", "Rune_v8n_5点_480x640"),
    "Rune-v8n-fp16-20260624-D14367-B16.xml|293927":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Rune", "Rune_v8n_5点_IR"),
    "Rune-v8n-fp16-20260624-D14367-B16.bin|5803892":
                                         (2026, "深圳大学", "YOLOv8", "Pose", "Rune", "Rune_v8n_5点_IR"),
    # 仲恺农业工程学院 奇点：0526 IR 的另一版本
    "0526.xml|197722":                   (2025, "深圳大学", "YOLOv5", "Pose", "Armor", "0526_IR_仲恺版"),
    # 浙江大学 Hello World（能量机关 YOLO11 九点）
    "buff.onnx|39044804":                (2025, "浙江大学", "YOLO11", "Pose", "Rune", "buff_9点"),
    # 常州大学 Climber（RepVGG 九点打符 IR）
    "buff_repvgg.xml|186733":            (2026, "常州大学", "RepVGG", "Pose", "Rune", "buff_repvgg_9点"),
    "buff_repvgg.bin|2502304":           (2026, "常州大学", "RepVGG", "Pose", "Rune", "buff_repvgg_9点"),
    # 江南大学 SHARK（雷达，YOLOv5 三尺度）
    "armor.onnx|30147068":               (2026, "江南大学", "YOLOv5", "Det", "Armor", "雷达_1280_13类"),
    "armor.pt|14370941":                 (2026, "江南大学", "YOLOv5", "Det", "Armor", "雷达_1280_13类"),
    "armor.engine|18276260":             (2026, "江南大学", "YOLOv5", "Det", "Armor", "雷达_1280_13类_TensorRT"),
    "armor_batch.engine|32272596":       (2026, "江南大学", "YOLOv5", "Det", "Armor", "雷达_1280_13类_batch_TensorRT"),
    "car.onnx|28861762":                 (2026, "江南大学", "YOLOv5", "Det", "Car", "雷达_640_5类"),
    "car.pt|14364733":                   (2026, "江南大学", "YOLOv5", "Det", "Car", "雷达_640_5类"),
    "car.engine|17583028":               (2026, "江南大学", "YOLOv5", "Det", "Car", "雷达_640_5类_TensorRT"),
    # 复旦大学 星云EGA（雷达，YOLOv12）
    "armor_best.onnx|80641842":          (2026, "复旦大学", "YOLOv12", "Det", "Armor", "雷达_3类"),
    "armor_best.pt|40744039":            (2026, "复旦大学", "YOLOv12", "Det", "Armor", "雷达_3类"),
    "armor_best.engine|44325759":        (2026, "复旦大学", "YOLOv12", "Det", "Armor", "雷达_3类_TensorRT"),
    "best.onnx|37771288":                (2026, "复旦大学", "YOLOv12", "Det", "Car", "雷达车辆_1280"),
    "best.pt|19106714":                  (2026, "复旦大学", "YOLOv12", "Det", "Car", "雷达车辆_1280"),
    "best.engine|23750482":              (2026, "复旦大学", "YOLOv12", "Det", "Car", "雷达车辆_1280_TensorRT"),
    "MobileNetv2.2.pth|27189835":        (2026, "复旦大学", "MobileNetV2", "Det", "Backbone", "特征提取"),
    # 中南大学 FYT 雷达（YOLOv10 端到端）
    "armor.onnx|9222339":                (2024, "中南大学", "YOLOv10", "Det", "Armor", "雷达_11类"),
    "armor.pt|5705145":                  (2024, "中南大学", "YOLOv10", "Det", "Armor", "雷达_11类"),
    "armor_new2.pt|5702152":             (2024, "中南大学", "YOLOv10", "Det", "Armor", "雷达_11类_new2"),
    "car.onnx|9317887":                  (2024, "中南大学", "YOLOv10", "Det", "Car", "雷达车辆"),
    "car.pt|5723193":                    (2024, "中南大学", "YOLOv10", "Det", "Car", "雷达车辆"),
    # 深圳职业技术大学 RCIA（能量单元 6dof 位姿）—— .pt 元数据实为 task=detect, names={0:'armor'}
    "best-v2.pt|5454426":                (2026, "深圳职业技术大学", "YOLO11", "Det", "Armor", "能量单元6dof"),
    # 西北工业大学 WMJ（反无人机）—— .pt 元数据 task=pose
    "wmj_anti_drone.pt|6426278":         (2026, "西北工业大学", "YOLOv8", "Pose", "AntiDrone", "反无人机"),
    # 东南大学：神经网络压缩低码率图传（非识别，供参考）
    "qvrf_gs_rlfn_x2_192.onnx|270617":   (2026, "东南大学", "QVRF", "Det", "Compress", "图传_192x384"),
    "qvrf_gs_rlfn_x2_448.onnx|270639":   (2026, "东南大学", "QVRF", "Det", "Compress", "图传_448x896"),
    "msssim_g_s_fp32.xml|21940":         (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_生成器g_s_IR"),
    "msssim_g_s_fp32.bin|5972496":       (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_生成器g_s_IR"),
    "msssim_h_a_fp32.xml|8737":          (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_超先验h_a_IR"),
    "msssim_h_a_fp32.bin|4163076":       (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_超先验h_a_IR"),
    "msssim_h_s_fp32.xml|8927":          (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_超先验h_s_IR"),
    "msssim_h_s_fp32.bin|11971972":      (2026, "东南大学", "MSSSIM", "Det", "Compress", "图传_超先验h_s_IR"),
    "realesr-general-x4v3.pth|4885111":  (2026, "东南大学", "Real-ESRGAN", "Det", "Compress", "图传_超分x4v3"),

    # ============ 第三轮：论坛 markdownContent 字段补扫发现（2026-09-10）============
    # 华中科技大学 狼牙：英雄视觉（帖子 41998）
    "best_06_02.onnx|3615824":           (2024, "华中科技大学", "YOLOX", "Pose", "Armor", "英雄自瞄_12类_416"),
    "best_06_02.xml|302342":             (2024, "华中科技大学", "YOLOX", "Pose", "Armor", "英雄自瞄_12类_416_IR"),
    "best_06_02.bin|1774618":            (2024, "华中科技大学", "YOLOX", "Pose", "Armor", "英雄自瞄_12类_416_IR"),
    "outpost_nano_100epoch_1_8.onnx|11020389":
                                         (2024, "华中科技大学", "YOLOv8", "Det", "Armor", "前哨站_green"),
    "0715_v10_416n.xml|317231":          (2024, "华中科技大学", "YOLOv10", "Det", "Armor", "416_IR"),
    "0715_v10_416n.bin|9104456":         (2024, "华中科技大学", "YOLOv10", "Det", "Armor", "416_IR"),
    "0801_384nano_green.xml|223234":     (2024, "华中科技大学", "YOLOv8", "Det", "Armor", "前哨站_green_384_IR"),
    "0801_384nano_green.bin|12059932":   (2024, "华中科技大学", "YOLOv8", "Det", "Armor", "前哨站_green_384_IR"),
    "svm_numbers.xml|956159":            (2024, "华中科技大学", "SVM", "Det", "Cls", "数字_SVM"),
    "svm_numbers_rbf.xml|5978693":       (2024, "华中科技大学", "SVM", "Det", "Cls", "数字_SVM_RBF"),
    "armor.onnx|259246":                 (2024, "华中科技大学", "CNN", "Det", "Cls", "数字_32x32_10类"),
    # 厦门理工学院 PFA：单目相机雷达站（帖子 43538）
    "armor.onnx|14344276":               (2024, "厦门理工学院", "YOLOv5", "Det", "Armor", "雷达_640_11类"),
    "car.onnx|14306466":                 (2024, "厦门理工学院", "YOLOv5", "Det", "Car", "雷达_640_5类"),
}


def build_plan(records: list[dict], known_sha: set[str],
               used: dict[str, int]) -> list[dict]:
    """按 sha256 去重，生成 旧路径 -> 新路径 的计划（跳过已归位内容，可增量执行）。"""
    by_sha: dict[str, list[dict]] = {}
    for r in records:
        if r["status"] != "ok" or r["sha256"] in known_sha:
            continue
        if not os.path.exists(os.path.join(ROOT, r["dest"])):
            continue                       # 已归位或已清理
        by_sha.setdefault(r["sha256"], []).append(r)

    plans: list[dict] = []
    for sha, items in by_sha.items():
        rep = sorted(items, key=lambda r: r["dest"])[0]
        base = os.path.basename(rep["dest"])
        key = f"{base}|{rep.get('bytes')}"
        if key not in C:
            plans.append({"sha": sha, "sources": items, "error": f"未分类: {key}"})
            continue
        year, school, arch, pose, task, detail = C[key]
        ext = os.path.splitext(base)[1]
        stem = f"{school}_{arch}_{pose}_{task}_{detail}"
        name = stem + ext
        if name in used:
            used[name] += 1
            name = f"{stem}_{used[name]}{ext}"
        else:
            used[name] = 1
        plans.append({"sha": sha, "year": year, "name": name, "sources": items,
                      "src": rep["dest"], "dst": f"{year}/{name}",
                      "school": school, "arch": arch, "pose": pose,
                      "task": task, "detail": detail})
    return plans


def repo_of(source: str) -> str:
    """从 'github.com/owner/repo/path' 提取 'owner/repo'。"""
    parts = source.split("/")
    return "/".join(parts[1:3]) if len(parts) >= 3 and parts[0].endswith(".com") else source


def write_organized_md(entries: list[dict], path: str) -> None:
    years = sorted({e["year"] for e in entries})
    lines = [
        "# RM 权重重组清单（按年份 / 学校 / 网络 / Pose / Armor）",
        "",
        "> 由 `scripts/organize_rm_weights.py --apply` 自动生成，请勿手工编辑。",
        f"> 去重后共 {len(entries)} 份内容；年份分档：{', '.join(str(y) for y in years)}。",
        "> 命名：`<学校>_<网络版本>_<Pose|Det>_<Armor|Rune|Radar|Car|Cls|Solver|General>_<细节>.<ext>`",
        "",
    ]
    for y in years:
        group = [e for e in entries if e["year"] == y]
        total = sum(e.get("bytes") or 0 for e in group)
        lines += [f"## {y}（{len(group)} 份，约 {total / 1048576:.1f} MB）", "",
                  "| 文件 | 来源数 | 来源仓库 | md5 |", "|---|---|---|---|"]
        for e in sorted(group, key=lambda x: x["path"]):
            srcs = sorted({repo_of(s) for s in e.get("sources", [])})
            show = "、".join(srcs[:3]) + (f" 等{len(srcs)}个" if len(srcs) > 3 else "")
            lines.append(f"| `{os.path.basename(e['path'])}` | {len(srcs)} | {show} "
                         f"| `{e.get('md5') or '-'}` |")
        lines.append("")
    lines += [
        "## 判定依据",
        "",
        "| 属性 | 依据 |",
        "|---|---|",
        "| 网络版本 | ONNX 内嵌 `description`（如 `YOLOv8n-pose` / `YOLO26n-pose`）；IR 输出节点命名（`/model.N/` 为 YOLOv8 系，`/m/model.N/` 为深大 RP 系） |",
        "| Pose | Ultralytics 元数据 `task=pose` + `kpt_shape`；深大 RP 系据本仓库文档「四点检测模型」；输出通道含 4×2 关键点分量 |",
        "| Armor/Rune | `names` 类别名（如 `B1/B3/BS/R1/R3/RS` 为装甲；`b`/`r_target`/`buff` 为打符） |",
        "| 年份 | 模型版本赛季（`0708`→2024、`0526`→2025）或元数据 `date`；无标记时取发布仓库赛季 |",
        "| 学校 | 权重原始战队；同名框架衍生仓库（GKD-RM-Lab / SHM-white 等）的复制件已并入原始战队 |",
        "",
        "## 存疑项（建议人工复核）",
        "",
        "- `同济大学_YOLO11_Pose_Armor_yolo11_int8`：IR 含 `anchor_points`，Pose 属性由「同济使用四点模型」推定。",
        "- `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last`：同济以 `yolov8.xml` 命名，与北科 `mobilenetv3_last` 同源同字节；Pose 属性按 RM 惯例推定。",
        "- `武汉科技大学_YOLOX_Pose_Armor_opt1208/opt0527`：无元数据，按锚点数（3549=52²+26²+13²）判为 YOLOX 系、按通道数推为四点。",
        "- `talos战队`：`Blackjack200/talos_26` 未标明学校，暂以战队名标注。",
        "- `江南大学_YOLOv5_*`：ONNX 无 `task` 元数据，按输出锚点数（1280 输入 100800=3×33600）判为 YOLOv5 三尺度；类别名 `B1..R5` 判为装甲。",
        "- `深圳职业技术大学_YOLO_Pose_Rune_能量单元6dof.pt`：无元数据，据帖子标题「6dof 位姿检测」推为 Pose。",
        "- `东南大学_*_Compress_*`：为低码率图传的神经网络压缩权重（QVRF/MSSSIM/Real-ESRGAN），非识别任务，仅作留存。",
        "- `华北理工大学_YOLOv8_Det_*_雷达_*`：与 `武汉科技大学_YOLOv8_Det_*_雷达_*` 同源（武科大转载），此处为原始 `.pt`。",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正移动文件（默认仅预览）")
    ap.add_argument("--backfill", action="store_true",
                    help="仅把分类字段回填进已有 organized.json（不改动文件）")
    args = ap.parse_args()

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        records = json.load(fh)

    org_path = os.path.join(OUT_DIR, "organized.json")
    existing: list[dict] = []
    if os.path.exists(org_path):
        with open(org_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    known_sha = {e["sha256"] for e in existing if e.get("sha256")}
    used: dict[str, int] = {os.path.basename(e["path"]): 1 for e in existing}

    if args.backfill:
        by_sha: dict[str, tuple] = {}
        by_md5: dict[str, tuple] = {}
        for r in records:
            if r["status"] != "ok":
                continue
            cls = C.get(f"{os.path.basename(r['dest'])}|{r.get('bytes')}")
            if cls:
                by_sha[r["sha256"]] = cls
                if r.get("md5"):
                    by_md5[r["md5"]] = cls
        filled = 0
        for e in existing:
            if "task" in e:
                continue
            cls = by_sha.get(e.get("sha256")) or by_md5.get(e.get("md5"))
            if not cls:
                continue
            _, school, arch, pose, task, detail = cls
            e.update(school=school, arch=arch, pose=pose, task=task, detail=detail)
            filled += 1
        with open(org_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False, indent=2)
        print(f"已回填 {filled} / {len(existing)} 条分类字段 → {os.path.relpath(org_path, ROOT)}")
        return 0

    plans = build_plan(records, known_sha, used)
    bad = [p for p in plans if "error" in p]
    ok = [p for p in plans if "error" not in p]

    if bad:
        print(f"!! {len(bad)} 份内容未在分类表中：")
        for p in bad[:20]:
            print("   ", os.path.basename(p["sources"][0]["dest"]), "|", p["error"])
        return 1

    print(f"清单 {len(records)} 个文件，已归位 {len(existing)} 份，"
          f"本次新增 {len(ok)} 份内容\n")
    for p in sorted(ok, key=lambda x: (x["year"], x["name"])):
        tag = "MOVE" if p["src"] != p["dst"] else "KEEP"
        print(f"  [{tag}] {p['dst']}   <- {p['src']}  ({len(p['sources'])} 来源)")

    if not args.apply:
        print("\n（预览模式，未改动；加 --apply 执行）")
        return 0

    for p in ok:
        src = os.path.join(ROOT, p["src"])
        dst = os.path.join(OUT_DIR, p["dst"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
        for extra in p["sources"]:
            ep = os.path.join(ROOT, extra["dest"])
            if os.path.exists(ep) and os.path.abspath(ep) != os.path.abspath(dst):
                os.remove(ep)

    # 清理与既有内容重复的原始副本
    removed = 0
    for r in records:
        if r["status"] != "ok" or r["sha256"] not in known_sha:
            continue
        p = os.path.join(ROOT, r["dest"])
        if os.path.exists(p):
            os.remove(p)
            removed += 1

    for root, _dirs, files in os.walk(OUT_DIR, topdown=False):
        if root != OUT_DIR and not os.listdir(root):
            os.rmdir(root)

    # OpenVINO IR 必须成对：.xml（图）缺同名 .bin（权重）时，从同年同组复制一份
    paired: list[dict] = []
    for year in sorted({p["year"] for p in ok}):
        ydir = os.path.join(OUT_DIR, str(year))
        if not os.path.isdir(ydir):
            continue
        files = os.listdir(ydir)
        bins = [f for f in files if f.endswith(".bin")]
        for f in list(files):
            if not f.endswith(".xml"):
                continue
            stem = f[:-4]
            if stem + ".bin" in files:
                continue
            prefix = "_".join(stem.split("_")[:4])
            cand = [b for b in bins if b.startswith(prefix)]
            if not cand:
                continue
            # 取与 xml 名公共前缀最长者，避免配到同组但不同版本的 bin
            cand.sort(key=lambda b: len(os.path.commonprefix([stem, b[:-4]])), reverse=True)
            dst = os.path.join(ydir, stem + ".bin")
            shutil.copyfile(os.path.join(ydir, cand[0]), dst)
            paired.append({"year": year, "path": f"{year}/{stem}.bin",
                           "bytes": os.path.getsize(dst), "md5": "", "sha256": "",
                           "sources": [f"本地复制自 {year}/{cand[0]}"]})
            print(f"  [PAIR] {year}/{stem}.bin  <-  {year}/{cand[0]}")

    organized = existing + [{
        "year": p["year"], "path": p["dst"].replace(os.sep, "/"),
        "bytes": p["sources"][0].get("bytes"), "md5": p["sources"][0].get("md5"),
        "sha256": p["sha"],
        "sources": [f"{s['host']}/{s['repo']}/{s['path']}" for s in p["sources"]],
        "school": p["school"], "arch": p["arch"], "pose": p["pose"],
        "task": p["task"], "detail": p["detail"],
    } for p in ok] + paired
    organized.sort(key=lambda r: r["path"])
    with open(org_path, "w", encoding="utf-8") as fh:
        json.dump(organized, fh, ensure_ascii=False, indent=2)
    write_organized_md(organized, os.path.join(OUT_DIR, "ORGANIZED.md"))
    print(f"\n完成：本次新增 {len(ok)} 份；累计 {len(organized)} 份（organized.json / ORGANIZED.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
