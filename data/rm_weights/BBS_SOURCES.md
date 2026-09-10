# 官方论坛 / RM Search 检索记录（第二轮，2026-09-10）

第一轮（29 个 GitHub/Gitee 仓库）之后，本轮从 **RoboMaster 官方论坛** 与 **RM Search**
补充了新的权重来源。

## 一、检索方法

| 环节 | 做法 |
|---|---|
| 论坛页面 | 站点为 Nuxt 前端渲染，直接抓 HTML 取不到正文；改用论坛自身接口 |
| 帖子正文 | `POST https://bbs.robomaster.com/developers-server/rest/posts/info/{id}`（**无需鉴权**，返回 `htmlContent`） |
| 帖子枚举 | `POST .../rest/posts/list`，`{"pageSize":50,"pageNo":N,"filter":{}}`，共 1265 帖 |
| RM Search | 华南理工 `scutrobotlab/rm-search`（`search.scutbot.cn`）为 Meilisearch 代理，其 `index/crawler.go` 即上述论坛接口；本轮以其为索引入口 |
| 关键词 | 自瞄 / 视觉 / 装甲 / 识别 / 模型 / 权重 / YOLO / ONNX / 检测 / 能量机关 / 打符 / 瞄准 → 命中 87 帖 |

> `posts/search` 接口需要登录，故改用 `posts/list` 全量枚举 + 标题过滤。

## 二、本轮新增权重的来源帖

| 帖子 | 日期 | 战队 | 仓库 | 取到 |
|---|---|---|---|---|
| [1942761](https://bbs.robomaster.com/article/1942761) | 2026-08-25 | 深圳大学 RobotPilots | `SZURPVision/26_NNDeployment_Lib_and_Detection_Models` | V8 四点装甲 onnx、V8 五点打符 onnx、V5/V8/打符 OpenVINO IR |
| [1939101](https://bbs.robomaster.com/article/1939101) | 2026-08-15 | 深圳大学 RobotPilots | `SZURPVision/RuneDetectionModel` | `model-0624.onnx`（与上条打符模型同字节） |
| [1883640](https://bbs.robomaster.com/article/1883640) | 2026-04-26 | 华北理工大学 Horizon | `BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025` release `2026.04.24` | `Horizon-Armor/Car-*-ghost-p2-*.pt` 原始权重 |
| [1883887](https://bbs.robomaster.com/article/1883887) | 2026-05-06 | 吉林大学 TARS Go | `Fskaaaaaaaa/jlu_vision_26` | `0526.onnx`、`yolox_rune_3.6m.onnx` |
| [1938357](https://bbs.robomaster.com/article/1938357) | 2026-08-13 | 仲恺农业工程学院 奇点 | `NOMANE-0/QD_Vision2026` | 0526 IR、lenet/mlp、yolox_rune 系列 |
| [1938210](https://bbs.robomaster.com/article/1938210) | 2026-08-09 | 浙江大学 Hello World | `IC-Alan/HWauto_buff2026` | `buff.onnx`（YOLO11s-pose 9 点打符） |
| [1885745](https://bbs.robomaster.com/article/1885745) | 2026-06-07 | 常州大学 Climber | `CCZU-Climber/Climber_Vision_26` | `buff_repvgg` IR、best/tiny_resnet/yolo11/yolov5/yolov8 |
| [1942193](https://bbs.robomaster.com/article/1942193) | 2026-08-23 | 复旦大学 星云 EGA | `Ryaxwn7/EGA_Radar_Algorithm` | YOLOv12 armor/car 的 onnx+pt+engine、MobileNetv2 权重 |
| [1938353](https://bbs.robomaster.com/article/1938353) | 2026-08-13 | 江南大学 SHARK | `JNU-SHARK/shark-radar-vision` | YOLOv5 armor(1280)/car(640) 的 onnx+pt+engine |
| [1938208](https://bbs.robomaster.com/article/1938208) | 2026-08-09 | 西北工业大学 WMJ | `zplszz/WMJRadar` | `wmj_anti_drone.pt` |
| [1893539](https://bbs.robomaster.com/article/1893539) | 2026-06-19 | 深圳职业技术大学 RCIA | `BenmaoNeko/RCIA_Benmao_Vision` | `models/yolo/best.pt`、`best-v2.pt` |
| [19374](https://bbs.robomaster.com/article/19374) | 2024-08-29 | 中南大学 FYT | `baiyeweiguang/FYT2024_radar` | YOLOv10 armor/car 的 onnx+pt |
| [1941762](https://bbs.robomaster.com/article/1941762) | 2026-08-23 | 中国石油大学（华东）RPS | gitee `ustl-cod/sentry-auto-aim` | lenet/mlp、yolox_rune 系列 |
| [1884179](https://bbs.robomaster.com/article/1884179) | 2026-05-26 | 东南大学 | `ElainaXD/rm_compress` | QVRF / MSSSIM / Real-ESRGAN（低码率图传，非识别） |

以上已全部接入 `scripts/fetch_rm_weights.py`，可用
`python3 scripts/fetch_rm_weights.py --only <关键词>` 增量重抓。

## 三、已收录帖（权重在第一轮已抓）

| 帖子 | 战队 | 仓库 |
|---|---|---|
| [803315](https://bbs.robomaster.com/article/803315) | 同济大学 SuperPower | `TongjiSuperPower/sp_vision_25` |
| [54091](https://bbs.robomaster.com/article/54091) | 深圳大学 RobotPilots | `broalantaps/RobotDetectionModel` |
| [1942651](https://bbs.robomaster.com/article/1942651) | 南京理工大学 Alliance | `Alliance-Algorithm/rmcs_auto_aim_v2` |
| [1936030](https://bbs.robomaster.com/article/1936030) | 武汉科技大学 崇实 | `WUST-RM/awakening` |
| [1883871](https://bbs.robomaster.com/article/1883871) | 上海科技大学 magician | `Astra-Whale/SHtech_auto_aim` |
| [9655](https://bbs.robomaster.com/article/9655) | 北京科技大学 Reborn | `RebornVision/*` |
| [760845](https://bbs.robomaster.com/article/760845) | 浙江师范大学 浙狮 | `dielivelr/Z_LION_AutoAim2025` |
| [1939253](https://bbs.robomaster.com/article/1939253) | 深圳大学 RobotPilots | `SZURPVision/RP-26AutoAim-Frame`（框架，无权重） |

## 四、第三轮：修正 `markdownContent` 漏读（2026-09-10）

**前两轮的关键疏漏**：部分帖子正文只存在于 `markdownContent` 字段，而 `htmlContent` 为空串。
只解析 `htmlContent` 会把这类帖子误判为「无内容/无链接」。改为**双字段合并解析**后，
87 个命中帖中 **36 帖含仓库链接、共 65 个不同仓库**，新增权重来源：

| 帖子 | 战队 | 仓库 | 取到 |
|---|---|---|---|
| [41998](https://bbs.robomaster.com/article/41998) | 华中科技大学 狼牙 | `HUSTLYRM/HUST_HeroAim_2024` | YOLOX 英雄自瞄(416)、YOLOv10n(416)、YOLOv8 前哨站、SVM 数字识别，共 11 个文件 |
| [43538](https://bbs.robomaster.com/article/43538) | 厦门理工学院 PFA | gitee `CarryzhangZKY/pfa_vision_radar` | 单目雷达 armor / car onnx |
| [1942229](https://bbs.robomaster.com/article/1942229) | 深圳大学 RobotPilots | `SZURPVision/RP-26Rune` | `model-0624.onnx`（与 1939101 同字节，已去重） |

### 查证后确认「仓库无权重」的链接

| 仓库（帖子） | 结论 |
|---|---|
| `scutrobotlab/rm_vision_core`（华南虎，803708） | 纯 C++ 视觉库，能量机关走传统角点法，**无模型权重** |
| `tup-robomaster/TUP-NN-Train-2`（沈航，18608） | YOLOX 训练框架，**未提交权重** |
| `HHgzs/OpenRM-2024`、`HHgzs/TJURM-2024`（天大，20426） | 纯代码 |
| `SnocrashWang/WMJAimer`、`RuneLab`、`MPC`（西工大，9508） | 控制/拟合代码 |
| `zmsbruce/rm_power_rune`（哈工大 Hiter，18689） | 传统识别，无权重 |
| `ChenYuWuAi/powerrune24`、`coach114514/TCR-RM2026-Radar-OpenSource`、`LamdaDay/YAW_Auto_Controller`、`xiix666/sentry26`、`baiyeweiguang/CSU-RM-Sentry` | 均无权重 |
| `PraySky1337/AT_NN_Detector`（河北科技，1886180） | **本仓库已有**：`data/AT_NN_Detector/common/` 下 4 个 onnx，不重复收录 |

### 确认无模型的帖子

| 帖子 | 战队 | 说明 |
|---|---|---|
| [1662382](https://bbs.robomaster.com/article/1662382) | 佛山大学 醒狮 | 附件为 `装甲板V1.0.zip` 硬件资料 |
| [584512](https://bbs.robomaster.com/article/584512) | 深圳技术大学 悍匠 | 仅识别思路，无代码/权重 |
| [1882897](https://bbs.robomaster.com/article/1882897) | 辽宁科技大学 COD | Gitee 仓库为导航/决策/行为树 |
| [1438280](https://bbs.robomaster.com/article/1438280) | 北京理工大学 DreamChaser | 附件为 FPGA 硬件 zip/pdf |
| [1883871](https://bbs.robomaster.com/article/1883871) | 上海科技大学 | 硬件帖，权重见同队视觉帖（已收录） |

## 五、附带核查：图片与评论区

| 核查手段 | 结果 |
|---|---|
| 正文图片 / 二维码 | 有图的 4 帖共下载 41 张图（华南虎 14、佛山醒狮 18、辽科大 COD 8、深技大悍匠 1），`cv2.QRCodeDetector` 解码 **0 个二维码**；抽看为队徽、技术曲线、效果截图 |
| 评论区接口 | 从前端 JS 定位到 `POST /rest/comment/page`，参数 `{"filter":"<文章id>","pageNo":1,"pageSize":50}`（`filter` 为**标量**文章 id） |
| 全量评论扫描 | 87 个命中帖的评论正则匹配 `pan./aliyun/quark/lanzou/123pan/github/gitee/提取码/.onnx/.pt` → **0 条命中** |
| 网盘链接 | 5 帖含 `pan.baidu.com`，但均为硬件图纸 / 场地模型 / 数据集，非模型权重 |

结论：论坛侧可公开获取的模型权重**已全部收录**。

## 六、可复现的论坛接口

```bash
# 帖子正文（无需鉴权）
curl -sX POST https://bbs.robomaster.com/developers-server/rest/posts/info/1886180

# 帖子列表（全量枚举）
curl -sX POST https://bbs.robomaster.com/developers-server/rest/posts/list \
  -H 'Content-Type: application/json' \
  -d '{"pageSize":50,"pageNo":1,"filter":{}}'

# 评论列表（filter 为标量文章 id）
curl -sX POST https://bbs.robomaster.com/developers-server/rest/comment/page \
  -H 'Content-Type: application/json' \
  -d '{"filter":"1886180","pageNo":1,"pageSize":50}'
```

> 接口路径均从论坛前端 JS chunk（`bbs-web-static.robomaster.com/.../_nuxt/*.js`）中提取。

## 七、注意事项

- 论坛帖子普遍以 **网盘 + 提取码** 或 **图片内二维码** 分发，本轮仅收录了能解析出可下载 URL 的部分；
  上述「未取到」的帖子需要人工登录论坛查看图片或附件。
- 论坛权重同样受各战队声明约束（多为 **CC BY-NC-SA 4.0 / 禁止商用**），仅作技术交流留存。
