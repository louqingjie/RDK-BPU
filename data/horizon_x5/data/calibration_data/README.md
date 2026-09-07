# RDK X5 PTQ 校准集 v1（2026-09-07）

由 `scripts/prepare_calibration_data.py` 生成，来源 `RM2025-Armor-Public-Dataset`（3504 张）。

## 生成链路

```
BGR 原图 -> letterbox(等比缩放 + 填充 114) -> BGR→RGB -> /255.0 -> float32 -> NCHW -> .bin
```

与 `data/AT_NN_Detector/README.md` 描述的输入格式一致，且**与部署侧预处理逐字节对齐**——这是校准集有效的必要条件。

## 筛选与抽样

| 步骤 | 结果 |
|---|---|
| 原始 | 3504 张 |
| 剔除 `min(w,h) < 360` 低质量小图 | −1077 |
| aHash 近似重复帧去重 | −234 |
| 可用池 | 2193 张 |
| 按亮度分桶分层抽样 | **200 张** |

亮度分桶配额（刻意保留暗光与强光两端）：

| 亮度区间 | 池 | 选 |
|---|---|---|
| [0, 15) | 140 | 15 |
| [15, 30) | 107 | 20 |
| [30, 50) | 467 | 55 |
| [50, 80) | 857 | 65 |
| [80, 120) | 527 | 35 |
| [120, 180) | 90 | 8 |
| [180, 256) | 5 | 2 |

## 目录结构

```
calibration_data/
├── images/          # 校准数据本体，200 个 float32 NCHW .bin
│   └── images_0000.bin ... images_0199.bin
├── source/          # 预处理后的可视化图，仅人工检查用，勿喂给工具链
├── manifest.csv     # 抽样清单（来源图、原始尺寸、亮度、scale、pad）
└── README.md
```

单样本规格：`1×3×576×768` float32，共 1,327,104 元素 / 5,308,416 字节，值域 [0,1]。
已校验：200 个文件大小一致、数值范围 [0,1]、padding 像素 = 114/255 = 0.4471。

## 在 hb_mapper yaml 中使用

`images` 为 ONNX 输入节点名，因此推荐把 `cal_data_dir` 指向上一级，让工具链按子目录名匹配输入：

```yaml
model_parameters:
  onnx_model: 'praysky_coord_noe2e_0331_576x768.onnx'
  march: 'bayes-e'
  input_parameters:
    input_name: 'images'
    input_shape: '1x3x576x768'
    input_layout_train: 'NCHW'
    input_type_train: 'rgb'
  norm_type: 'no_preprocess'      # 数据已在预处理阶段完成 /255 与 BGR2RGB

calibration_parameters:
  cal_data_dir: '/data/horizon_x5/data/calibration_data'
  cal_data_type: 'float32'
  calibration_type: 'default'     # 掉点明显时依次试 mix / percentile / kl
```

> ⚠️ 容器内路径为 `/data/...`（`data/` 挂载到容器 `/data`）。
> ⚠️ 以上参数名需在容器内对照 `oe_sdk/*/samples/ai_toolchain/model_zoo` 的模板复核后再跑。
> 若后续重新导出的 ONNX 改为 NHWC 输入，用 `--layout nhwc` 重跑脚本即可。

## 已知不足

1. **无负样本**：源数据集 3504 张全部含目标，0 张空帧。缺少背景/空场景会削弱背景激活的统计，可能推高误检 → 建议实拍补 20~40 张空帧（空场地、对天、对地）后重跑。
2. **非本机相机采集**：分辨率与光照特性与 CS016-10UC 不完全一致，属于最接近的替代方案。
3. **未做人工目视复核**：`source/` 下 200 张图建议抽看一遍，剔除明显异常帧。

详见 [docs/数据集使用规范.md](../../../docs/数据集使用规范.md)。
