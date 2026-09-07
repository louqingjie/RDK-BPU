# AT_NN_Detector 模型包 RDK X5 部署可行性评估报告

- **日期**：2026-09-04
- **评估对象**：`/workspace/data/AT_NN_Detector` 模型包（河北科技大学 Actor&Thinker 战队 YOLO26 装甲板姿态检测模型）
- **评估问题**：现有权重能否直接导出为 RDK X5（地瓜旭日5）可用的 `.bin` 模型

## 一、模型包现状

| 文件 | 注意力 | 内置后处理 | 输出形状 | 用途 |
|------|--------|-----------|----------|------|
| `common/praysky_c2psa_e2e_0228_640x640.onnx` | C2PSA | TopK（端到端） | (1,30,18) | Orin NX 等通用 ONNX Runtime |
| `common/praysky_c2psa_e2e_0228_576x768.onnx` | C2PSA | TopK（端到端） | (1,30,18) | 同上 |
| `common/praysky_coord_noe2e_0331_640x640.onnx` | CoordAtt | NMS | (1,30,18) | 同上 |
| `common/praysky_coord_noe2e_0331_576x768.onnx` | CoordAtt | NMS | (1,30,18) | 同上 |
| `axera/praysky_coord_noe2e_qat_u16_all.axmodel` | CoordAtt | — | 已编译 | Axera650 专用（Pulsar2 工具链） |

两组 ONNX 的网络差异：

- **C2PSA（e2e）**：P5 层为多头自注意力，mAP50-95 = 0.8801，模型更小（~4.3 MB）；但 SoftMax 产生大 outlier，uint8/uint16 量化精度受损。
- **CoordAtt（noe2e）**：P5 层为通道+空间注意力，全部为卷积 + Sigmoid，量化友好；mAP50-95 = 0.8717，召回更高（0.9695）。

## 二、RDK X5 平台约束

- BPU（Bayes-e 架构）**仅支持 INT8 定点运算**，必须经 Open Explorer（OE）工具链 `hb_mapper` 量化编译。
- **数据依赖的动态算子（NMS、TopK 等）无法在 BPU 上编译**，会转换失败或整块回落 CPU，导致严重性能损失。
- BPU 偏好 **NHWC** 张量布局；标准做法是只导出裸检测分支（box/cls/kpt），解码 + NMS 全部放在 CPU（8 核 A55）执行。
- 社区实测教训：YOLOv11 部署 X5 时 Softmax 不支持 int8 被调度到 CPU，CPU↔BPU 频繁搬运导致端到端仅 7 FPS（优化后 47 FPS）。

## 三、逐项分析

| 文件 | 能否直接转 .bin | 原因 |
|------|----------------|------|
| `c2psa_e2e` ×2 | ❌ | 图内 TopK 为动态算子，BPU 不支持；Softmax 存在 int8 量化精度风险 |
| `coord_noe2e` ×2 | ❌ | 图内 NMS 为动态算子（输出形状依赖输入数据），hb_mapper 编译失败或回落 CPU |
| `axera/*.axmodel` | ❌ | Pulsar2 针对 Axera650 编译的专属格式，与地平线工具链不互通 |

## 四、结论

**现有 6 个模型文件无一可直接导出为 RDK X5 的 .bin。**

但需要明确：不适合的是**导出方式**（后处理已烧进计算图），而非网络本身。CoordAtt 主干对量化依然友好，其权重适合 RDK X5，只是必须换一种导出姿势。

## 五、建议的转换路径

```text
1. 获取 best.pt 及训练代码（需联系模型作者）
2. 修改 forward：
   - 去掉 NMS / TopK / 解码逻辑，仅输出裸 box/cls/kpt 分支
   - 输出转 NHWC 布局（permute(0,2,3,1)）
   - 导出无后处理的 ONNX
3. 准备校准数据集（建议 100+ 张真实装甲板图，覆盖不同光照场景）
   > ⚠️ 数据集来源红线见 [数据集使用规范.md](./数据集使用规范.md)：
   > `data/AT_NN_Detector/video/` 下 3 个 mp4 **禁止用于校准集与验证集**；
   > `/workspace/video.avi` 仅限暗光补充（≤20%）。
4. OE 工具链 Docker 内运行 hb_mapper checker，确认无算子回落 CPU
5. hb_mapper makertbin 完成 PTQ 量化，生成 .bin
6. 若 mAP 掉点明显，使用 horizon_plugin_pytorch 做 QAT
```

分辨率建议选 **576×768**：适配 CS016-10UC 相机（1440×1080）只需一次 resize、无 letterbox padding；BPU 上输入尺寸基本不影响算力消耗。

## 六、风险与注意事项

1. **Outpost 类别异常**：README 已注明 w8a16 量化下识别 Outpost 存在偶发异常（疑似数据集脏数据）。RDK 的 int8 量化更激进，上车前务必重点验证该类别。
2. **mAP 预期**：CoordAtt 的 mAP50-95（0.8717）略低于 C2PSA（0.8801），但召回更高，符合自瞄"宁可多报不漏检"的取向。
3. **CPU 后处理压力**：解码 + NMS 移至 CPU 后约占 12~14 ms/帧（社区数据，C++ 实现可接受）；关键点坐标映射也需在 CPU 侧完成。
4. **颜色分支**：模型含 4 类颜色分支（B/R/G/P），转换后需确认该分支输出节点在 YAML 中正确配置。

## 七、参考

- 数据集使用规范（禁用/限用清单）：[数据集使用规范.md](./数据集使用规范.md)
- 模型包自带 `README.md`（两组模型区别、量化说明、命名规则）
- RDK 社区：YOLOv8 / YOLOv5 在 X5 上的模型转换与部署实测帖
- CSDN：RDK X5 部署 YOLOv11n 从 7 FPS 到 47 FPS 的性能优化实录（Softmax int8 回落 CPU 问题）
