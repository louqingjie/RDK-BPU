# 量化工作总结与模型 bin 清单

更新：2026-09-09。环境基线：OE 1.2.8（hb_mapper 1.24.3 / hbdk 3.49.15），march bayes-e，板端 RDK OS 3.5.0（DNN runtime 1.24.5 / HBRT 3.15.55.0）。除特别说明外均为 PTQ、latency 模式、O2 优化。

本文回答两个问题：**导出的所有 bin 各是什么、哪些成功哪些失败**；并给出 20 项踩坑教训索引与遗留问题。

---

## 一、总体战况

| 系列 | 模型 | 编译 bin 数 | 最终结论 |
|---|---|---|---|
| rp0526（深大 RobotPilots，魔改 YOLOv5+MobileNetV3，640×640） | rp_0526.onnx 系列 | 9 | **全部不可用**（检出失败），探针实验完成损伤定位 |
| sp25（同济 sp_vision_25） | yolov5 检测器（rp24_0708.onnx）、tiny_resnet 分类器 | 4 | 检测器 INT8/int16_head 失败、all_int16 部分成功；**tiny_resnet 成功** |
| coord（AT_NN_Detector） | praysky_coord_noe2e_0331_576x768 | 1 | **链路/一致性/性能达标，精度待正式验收** |
| verifier（早期 C2PSA 试验） | praysky_c2psa_e2e_640x640 | 1 | 无文档记录，状态不明 |
| **合计** | | **15** | **仅 tiny_resnet 可直接使用；ORT FP32 兜底为当前唯一确定可用检测方案** |

---

## 二、编译模型 bin 清单（15 个）

### rp0526 系列（全部为 640×640）

| bin（quant/*/model_output/） | 大小 | 状态 | 关键证据 |
|---|---|---|---|
| `rp0526_640x640.bin`（baseline INT8） | 2.56MB | ❌ 失败 | 编译/加载/BPU 推理正常（BPU 4.95ms），但 conf 余弦 **0.0073（全零）**、阳性样本 0 检出（FP32 conf 0.978）；「编译成功 ≠ 能用」 |
| `rp0526_640x640_per_channel.bin` | 2.56MB | ❌ 失败 | per_channel + mix 校准后 armor_test 板端仍 0 检出 |
| `rp0526_640x640_probe.bin` | 2.56MB | 🔍 探针 | head 输入探针：20×20 分支 head 输入余弦 0.810，定位损伤起点 |
| `rp0526_640x640_probe_cv3.bin` | 2.56MB | 🔍 探针 | cv3 输入探针，支撑损伤累积结论 |
| `rp0526_640x640_probe_model10.bin` | 2.56MB | 🔍 探针 | model.10 三段探针：输入 0.833 → Conv 后 0.855 → SiLU 后 0.886，证实误差自深层主干累积 |
| `rp0526_640x640_probe_concat22.bin` | 2.56MB | 🔍 探针 | concat22（bypass/residual）探针，编译与板端 dump 均成功 |
| `rp0526_640x640_probe_concat23.bin` | 2.56MB | 🔍 探针 | concat23（bypass/residual/concat）探针 |
| `rp0526_640x640_int16_head.bin` | 2.88MB | ❌ 失败 | 仅 head 三尺度 Conv+Mul+Add INT16（node_info 混合量化），armor_test 仍 0 检出。注意：09-09 sp25 复核表明 head INT16 在同架构上有效，rp0526 失败不能反推「head 不可修」 |
| `rp0526_640x640_all_int16.bin` | 10.31MB | ❌ 失败 | 全局 Conv/Mul/Sigmoid INT16，armor_test 最大 conf 仅 0.012；bin 膨胀至 INT8 的 4 倍 |

### sp25 系列

| bin | 大小 | 状态 | 关键证据 |
|---|---|---|---|
| `sp25_yolov5_640x640.bin`（INT8） | 2.44MB | ❌ 失败 | 板端 36.3ms；conf 峰 0.9804→**0.6685**、候选 57→**1206**（×21）、top10 重合 0/10；整体余弦 0.997 系假象（P14） |
| `sp25_yolov5_640x640_int16_head.bin` | 2.69MB | ❌ 失败 | model.26 head 局部 INT16，conf 余弦 0.911 仍崩 |
| `sp25_yolov5_640x640_all_int16.bin` | 4.38MB | ⚠️ 部分成功 | 三个 head Conv 余弦 0.9134/0.9630/0.9767 → **0.9970/0.9987/0.9994**（特征量化误差可修）；但板端检出仍失败——**特征恢复 ≠ 检测恢复**，09-09 已撤回「INT16 无效/PTQ 无解」旧结论 |
| `sp25_tiny_resnet_32x32.bin` | 526KB | ✅ **成功** | 板端 **1.79ms/帧**；8/8 样本 argmax 与高精度参考一致；logits 余弦 0.998~1.000；前置手术：动态 batch 固化 + opset 17→11 声明改写；保留意见：8 样本 ≠ 分类准确率 100% |

### coord 系列

| bin | 大小 | 状态 | 关键证据 |
|---|---|---|---|
| `coord_trial_576x768.bin`（data/coord_trial_20260907/） | 2.3MB | ⚠️ 部分成功 | 12 输出 raw ONNX（去图内解码/NMS）+ 90 帧 KL 校准；板端 12 路输出与量化仿真**逐元素一致（最大绝对误差 0）**；单线程 1000 帧 **7.5618ms ≈ 131.9 FPS**；单正样本置信度 0.615→0.396、四角点误差 0.385~2.14px。README 自述「尚未通过正式精度验收」：校准视频含直播 UI，需真实相机原图独立验证集 |

### verifier（无文档记录）

| bin | 大小 | 状态 |
|---|---|---|
| `quant/verifier_output/sim_model/praysky_c2psa_e2e_640x640/praysky_c2psa_e2e_640x640.bin` | 4.51MB | ❓ 任何文档均未记录该实验及结论 |

---

## 三、验证与调试数据 bin（非模型产物）

| 位置 | 内容 | 用途 |
|---|---|---|
| `data/camera_test/board_out/`（43 个） | 5 张真实相机帧输入 + armor_test/calib_ref 系列各量化方案板端输出 + probe_concat22/23、probe_cv3 子目录中间层 dump + `armor_test_ort_fp32.bin`（FP32 正确输出参照） | 三重验证（逐通道余弦 / 板端 infer / 真实相机帧）的原始证据 |
| `data/sp25_requant/`（17 个） | `board_C_INT8_*.bin` ×5、`board_D_INT16_*.bin` ×5（FP32 conf top2 阳性+中位+最低 2 阴性共 5 张固定图）、`dump_int8/` 转换链路逐层 dump | 实验①逐阶段误差定位：C 确认 INT8 全坏（conf 峰恒 0.500、候选恒 1200、top10 全 0/10），D 证实 INT16 特征恢复但检测未恢复 |
| `data/coord_trial_20260907/` | `test_rgb_s8_chw.bin`、`positive_rgb_s8_chw.bin`（int8 = RGB−128 正确输入约定示例） | 板端复现与部署示例 |
| `data/onnx/sp_vision_25/yolov5.bin` | OpenVINO IR 数据文件（非量化产物） | 同济原始交付件 |

---

## 四、成败判定速查

**成功**
- ✅ `sp25_tiny_resnet_32x32.bin`：教科书式量化（文件减半、1.79ms、余弦 0.998+）
- ✅ 支撑链路：同源验证（余弦 1.0001）、tiny_resnet 前处理手术、ORT FP32 板端兜底（rp0526 conf 0.9785 / ~208ms，sp25 NUC 8.2ms）、相机采集-推理-标注链路

**部分成功**
- ⚠️ `sp25_yolov5_all_int16.bin`：特征恢复、检测未恢复
- ⚠️ `coord_trial_576x768.bin`：链路/一致性/性能达标，精度未正式验收

**失败**
- ❌ rp0526：baseline INT8、per_channel、int16_head、all_int16（4 个方案全部 0 检出/不可用）
- ❌ sp25_yolov5：INT8、int16_head
- ❓ 探针 bin ×5：诊断工具，成功达成了「损伤自深层主干累积」的定位目的，非可用模型
- ❓ `sp25_yolov5_o0`（O0 编译等级对照）：无独立产物与报告留存，`quant/rp_0526/rp_0526_config.yaml` 为 0 字节空文件

---

## 五、20 项踩坑教训索引（docs/量化踩坑记录.md）

1. 内嵌后处理（TopK/NMS/GatherND/NonZero）BPU 无法编译，须移到 CPU
2. FP16 导出的 ONNX 必须整图转 FP32，仅图首 Cast 不够
3. 相机 SDK 头文件/库链接：自包含头文件 + 实际存在的库全路径
4. yaml 缺 `input_layout_rt`（rgb 输入必须显式指定布局）
5. ONNX 图手术四坑：Slice 参数 int64、opset≥11 Clip 参数化、标量 initializer 形状 `[]`、图输出名严格一致
6. `hrt_model_exec` 子命令是 `perf`/`infer`，非 `model_perf`
7. 喂数据布局与 `input_layout_rt` 不一致 → conf 恰 0.50 的网格状垃圾检测；先怀疑喂错，用阳性对照判定
8. 偶发 `hbDNNInitializeFromFiles -6000006`：重试 3~5 次 + 合并单 SSH 会话
9. profiler 输出目录须先 `mkdir -p`
10. 逗号分隔 input_file 对应多输入节点，不是多帧批量
11. **（致命）** 输出混量纲被 per-tensor INT8 摧毁（66ch head Conv 混 0~640 坐标与 ±78 logit，conf 余弦 0.0073）；输出端四种手术均无效（位置在损伤点之后）
12. `[1,25200,1]` 简并形状输出被 runtime 清零——拆分输出避免末维为 1
13. runtime CPU 侧输出处理 ~25ms（反量化 32.7ms + 布局转换 27.5ms）——≥1MB 输出必须 profiler 实测
14. 逐层余弦会骗人（0.9994 但模型全坏）——验收必须含阳性对照
15. 仿真延迟准（4.2→4.95ms），精度与端到端性能必须实机验证
16. 负样本假阴性差点误判成功——测试集必须含已知置信度的阳性样本
17. 逐点实测修正根因：head Conv 阈值仅 17~24，损伤自深层主干累积；「INT16 无效」推论后被 09-09 sp25 复核撤回
18. 第三方 IR 先溯源原始 ONNX（同源数值验证余弦 >0.999），不做格式反推
19. opset 17 超工具链上限 11——语义未变时可声明改写 `opset_import` 兜底
20. 动态 batch 输入必须先固化 batch=1

---

## 六、遗留问题

1. rp0526 FP32 PTQ 后漏检根因未闭环；重点嫌疑：解码阶段量化尺度与部署链路一致性
2. sp25 检测器：head Conv 特征已恢复 0.997+，但 conf/color/num 检出通道仍崩——下一步应排查解码/拼接张量校准阈值达数千的共尺度问题
3. coord_trial：需真实相机原图独立验证集 + C++ 解码/NMS（不能用原 DFL 与 raw*2+grid 解码；板端网格解码为 raw+grid+0.5 后乘 stride）
4. `quant/coord_trial_576x768.yaml`、`quant/assessment_20260907/`（post.onnx、board_comparison 等）未随库存档，仍在当时的工作机
5. C2PSA verifier bin 与 sp25_o0 实验无结论记录
