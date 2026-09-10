# RoboMaster 开源自瞄 / 视觉权重库

从 **GitHub / Gitee 开源仓库** 与 **RoboMaster 官方论坛（经 RM Search 索引）** 两轮检索采集，
按 **年份 / 学校 / 网络版本 / 是否 Pose / 是否 Armor** 组织。仅供技术交流与量化研究，
供本仓库 RDK X5（BPU bayes-e）量化工作取用。

- 逐文件来源与校验：`ORGANIZED.md`（人读）/ `organized.json`（机读，含全部 sources）
- 原始抓取记录：`manifest.json`（223 条，含来源 URL 与 sha256）
- 论坛检索记录：`BBS_SOURCES.md`

## 一、目录结构

```
data/rm_weights/
├── 2020/    10 份   深大早期 Caffe/Darknet、装甲数字识别
├── 2021/     1 份   野狼战队 MNIST
├── 2023/    10 份   rm_vision 系 MLP/LeNet、杭电打符 YOLOX
├── 2024/    38 份   深大 0708、北科 Reborn、华科狼牙、厦理工 PFA、中南 FYT/打符
├── 2025/    24 份   深大 0526、同济 sp25、上科大 SKD、国科大 RM4P、RPS、浙大
├── 2026/    63 份   深大 26 部署库、武科大、华理 Horizon、复旦、江南、国科大、RPS、talos…
├── ORGANIZED.md  organized.json  manifest.json  BBS_SOURCES.md  README.md
└── fetch.log
```

合计 **146 个权重文件 / 约 1044 MB**（对应 239 条抓取记录，内容重复的已合并）。

## 二、命名规则

```
<学校>_<网络版本>_<Pose|Det>_<用途>_<细节>.<ext>
```

| 字段 | 取值 | 含义 |
|---|---|---|
| 学校 | 深圳大学 / 同济大学 / 上海科技大学 / 北京科技大学 / 华北理工大学 / 武汉科技大学 / 中国科学院大学 / 复旦大学 / 江南大学 / 中南大学 / 浙江大学 / 常州大学 / 浙江师范大学 / 杭州电子科技大学 / 合肥工业大学 / 华东交通大学 / 西南石油大学 / 华南师范大学 / 深圳职业技术大学 / 西北工业大学 / 东南大学 / RPS战队 / talos战队 / 野狼战队 / 通用 / 个人_* | 权重原始战队 |
| 网络版本 | YOLOv5 / YOLOv8 / YOLOv10 / YOLO11 / YOLOv12 / YOLO26 / YOLOX / tiny-YOLOv2 / EfficientNet / RepVGG / ResNet / MobileNetV2 / MobileNetV3 / LeNet / MLP / FC / CNN / QVRF / MSSSIM / Real-ESRGAN / PredNet | 主干架构 |
| Pose / Det | `Pose` = 回归关键点（RM 常见「四点模型」）；`Det` = 仅输出检测框 | 是否 Pose |
| 用途 | `Armor`=装甲板；`Rune`=打符；`Radar`=雷达；`Car`=车辆；`Cls`=数字/兵种分类；`Solver`=弹道预测；`Compress`=图传压缩；`Backbone`=特征主干；`AntiDrone`=反无人机；`General`=通用检测 | 是否 Armor |

示例：

- `2024/深圳大学_YOLOv5_Pose_Armor_0708.onnx` —— 深大 2024 赛季四点装甲模型（YOLOv5+MobileNetV3）
- `2025/同济大学_YOLOv8_Pose_Rune_5点_best2-sim.onnx` —— 同济 `best2-sim`，实为 **5 点打符**模型（元数据 `task=pose, kpt_shape=[5,2], names={0:'b'}`），**不是装甲模型**
- `2026/深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_480x640.onnx` —— 深大 RM2026 部署库的 V8 四点装甲（7 兵种类别）
- `2026/复旦大学_YOLOv12_Det_Armor_雷达_3类.onnx` —— 复旦雷达站 YOLOv12 装甲检测（仅框）
- `2026/深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_IR.xml` —— 深大能量机关 **五点** 模型（OpenVINO IR）

## 三、两轮来源

| 轮次 | 来源 | 说明 |
|---|---|---|
| 第一轮 | GitHub / Gitee 29 个开源仓库 | 见 `manifest.json`；同名框架衍生仓库（GKD-RM-Lab / SHM-white 等）的复制件并入原始战队 |
| 第二轮 | 官方论坛 1265 帖 → 87 帖命中 → 14 帖提供可下载权重 | 见 `BBS_SOURCES.md`；新增 `SZURPVision/26_NNDeployment_*`、`BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025` 等仓库与 Release |
| 第三轮 | 修正 `markdownContent` 字段漏读后重扫 → 36 帖含 65 个仓库 | 新增 `HUSTLYRM/HUST_HeroAim_2024`（华科狼牙，11 个文件）、gitee `CarryzhangZKY/pfa_vision_radar`（厦理工 PFA）、`SZURPVision/RP-26Rune` |

## 四、去重说明

内容完全相同的文件（sha256 一致）只保留一份，全部来源记录在 `organized.json` 的 `sources`。
两轮累计合并 81 份冗余副本，其中典型：

| 重复内容 | 原始份数 | 归属 |
|---|---|---|
| `mlp.onnx` / `lenet.onnx` | 13 / 8 | 华南师范大学（rm_vision 系）/ 通用 |
| `tiny_resnet.onnx` | 7 | 同济大学 |
| 深大 `0708` / `0526` 检测模型及 IR | 5 / 4（多轮累计更多） | 深圳大学 |
| `yolox_rune*` 打符系列 | 多份 | 中南大学 |
| 同济 `yolov5` / `yolov8` / `yolo11` IR | 3～4 | 同济框架衍生仓库 |

## 五、属性判定依据

| 属性 | 依据 |
|---|---|
| 网络版本 | ONNX 内嵌 `description`（如 `Ultralytics YOLOv8n-pose` / `YOLOv12m` / `YOLOv10n`）；OpenVINO IR 输出节点命名（`/model.N/` = YOLOv8 系，`/m/model.N/` = 深大 RP 系）；输出锚点数（`25200`/`100800` = YOLOv5 三尺度，`3549` = YOLOX，`300/5040` = 端到端） |
| Pose | Ultralytics 元数据 `task=pose` + `kpt_shape`；深大 RP 系据本仓库文档「四点检测模型」；输出通道含 4×2 关键点分量 |
| Armor / Rune | 元数据 `names` 类别名（`B1/B3/BS/R1/R3/RS`→装甲；`b`/`r_target`/`buff`/`b_inactive`→打符） |
| 年份 | 模型版本赛季（`0708`→2024、`0526`→2025）或元数据 `date`；无标记时取发布仓库赛季 |
| 学校 | 权重原始战队；命名规范见第二节 |

**存疑项**（建议人工复核，完整列表见 `ORGANIZED.md`）：

- 同济 `yolo11_int8`、北科 `MobileNetV3_last`、武科大 `opt1208/opt0527` 的 Pose 属性为推定。
- 江南大学 `YOLOv5_Det_*` 无 `task` 元数据，按输出锚点数与类别名判定。
- 深圳职业技术大学 `能量单元6dof` 的 Pose 属性据帖子标题推定。
- 华中科技大学 `YOLOX_Pose_Armor_英雄自瞄_12类_416`：无元数据，按输出通道 25 = 4 bbox + conf + color + 4×2 关键点 + num 拆解推为四点。
- 华中科技大学 `YOLOv10_Det_Armor_416`、`YOLOv8_Det_Armor_前哨站_green` 按 `task=detect` 与模型名判定为仅框。
- `talos战队`、`个人_*` 为未标明学校时的兜底命名。

## 六、与本仓库量化工作的关系

| 权重 | 用途 |
|---|---|
| `2024/深圳大学_YOLOv5_*_0708*`、`2025/深圳大学_YOLOv5_*_0526*` | 对应 `data/onnx/rp_0526.onnx` 与 rp0526 量化系列 |
| `2025/同济大学_YOLO11_*`、`2024/北京科技大学_YOLOv8_*_MobileNetV3_last` | 对应 `data/onnx/sp_vision_25/` 与 sp25 量化系列 |
| `2025/上海科技大学_YOLOv5_*_SKD250526*` | 对应 `docs/SHtech_SKD_NV12量化实录.md` |
| `2026/武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_*` | 与 `data/AT_NN_Detector` 的 CoordAtt 576×768 同源 |
| `2026/深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_*` | 深大最新五点打符，量化候选 |
| `2026/华北理工大学_YOLOv8_Det_*_雷达_*` | 与武科大转载的同源模型，此处为原始 `.pt` |
| `2025/同济大学_ResNet_Det_Cls_数字_32x32_9类.onnx` | 对应已量化成功的 `sp25_tiny_resnet_32x32.bin` |
| `*_Pose_Armor_*` 的 `.onnx` | 后续 RDK X5 PTQ 的候选源模型 |

> `.engine`（TensorRT）与 `.axmodel`（AX650）为平台相关编译产物，不可直接用于 RDK X5，
> 保留仅作对照；`.onnx` / `.pt` / OpenVINO IR 才是可迁移的权重。

## 七、许可与合规

- 各权重归属其原始战队/作者，多数在开源帖中声明 **CC BY-NC-SA 4.0 / 仅限技术交流、禁止商用**；
  部分仓库（如 `SHtech_auto_aim`）在 `NOTICE` 中明确模型不属于代码的 MIT 许可。
- 论坛权重同样受各战队声明约束；本轮仅采集公开发布的下载链接，未绕过任何访问控制。
- 本目录仅做**汇总留存**，不主张任何权利，不改变原始许可。

## 八、复现

```bash
# 1) 抓取/增量抓取（幂等，写入原始布局 + manifest.json）
python3 scripts/fetch_rm_weights.py --list
python3 scripts/fetch_rm_weights.py                       # 全量
python3 scripts/fetch_rm_weights.py --only wust,SZURPVision  # 只补某几个仓库

# 2) 按年份/学校/网络/Pose/Armor 重组（可增量执行，--apply 才真正移动）
python3 scripts/organize_rm_weights.py
python3 scripts/organize_rm_weights.py --apply
```

## 九、已知缺口

- 论坛中**有帖子但未取到权重**的战队：河北科技大学 Actor&Thinker（YOLO26 端到端 ONNX）、
  辽宁科技大学 COD、沈阳航空航天大学 TUP、华南理工大学华南虎、天津大学 OpenRM、
  佛山大学醒狮——正文/图片内的下载方式需登录论坛人工查看，详见 `BBS_SOURCES.md` 第四节。
- 年份对 `华东交通大学 fc.onnx`、`个人_hxfhxy Zenet` 等无赛季标记的权重为估算值。
- 第二轮新增的雷达（复旦/江南/中南）与图传压缩（东南大学）权重严格来说不属于「自瞄」，
  按「论坛全面检索」的要求一并保留；如需瘦身可删除 `*_Radar_*`、`*_Compress_*`、`*_Car_*`。
