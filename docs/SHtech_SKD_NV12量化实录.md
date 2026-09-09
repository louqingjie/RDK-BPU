# SHtech SKD250526 NV12 量化实录

> 2026-09-09，将上科大 [SHtech_auto_aim](https://github.com/Astra-Whale/SHtech_auto_aim) 主检测模型 SKD250526.onnx 以 **NV12 输入**量化部署到 RDK X5 的完整记录。
> 环境：OE v1.2.8（hb_mapper 1.24.3），板端 RDK OS 3.5.0（DNN 1.24.5 / HBRT 3.15.55.0）。
> 结论先行：**量化成功并已通过板端阳性对照验收**；BPU 3.74 ms（仿真 3.05 ms），NV12 色度损失可忽略。

## 1. 模型溯源与选择

| 候选 | 情况 | 判定 |
|---|---|---|
| `SKD250526.onnx` | 输入 image FP32 [1,3,512,640]，输出 [1,6720,21]，opset 11；checker 全图 BPU 通过（仅图尾 Concat+Transpose 回退 CPU） | ✅ **量化对象** |
| `SZU0526_fp32input_512x640_nopre_fixoutput.onnx` | FP16 导出（checker 失败于 `split_maxpool_into_multiples_pass`，P2 同款）；输出为像素坐标（-4179~1484）+ logit 混量纲，rp_0526 P3/P11 同款风险 | 备选，不量化。已转出 FP32 版存 `data/onnx/shtech/SZU0526_fp32.onnx` |

**axmodel 同源验证**：从 AX650 原厂 `SKD250526.axmodel` 二进制提取元数据
`outputs_info {"out": ["FP32", [1, 6720, 21]]}`，输入名 `image`，与 SKD250526.onnx 逐字一致
（含 pulsar2 的 `oeOnNX` 编译痕迹）——**上科大线上实战用的量化模型正是本 ONNX 的量化产物**，
本路线与其部署形态同构（pulsar2 uint8 输入 ≡ scale 1/255）。

## 2. 输出语义（量化友好的关键）

SKD 输出 [1,6720,21] **全通道概率域**（4 图 26880 行实测）：

| 通道 | 语义 | 实测范围 |
|---|---|---|
| 0~7 | 4 角点相对偏移（解码：`x*2*stride + 网格中心`，stride 8/16/32 按 64×80 + 32×40 + 16×20 遍历） | ±4 |
| 8 | conf（**已 sigmoid**，阈值 0.1） | 0~0.93 |
| 9~16 | tag（8 类，argmax） | 0~0.93 |
| 17~18 | color（2 类，argmax） | 0~0.93 |
| 19~20 | size（2 类，argmax） | 0~0.93 |

即作者在导出端已完成"fixoutput"式归一化——**不存在 rp_0526 混量纲崩坏的根因**。

## 3. 量化配置与校准集

- yaml：`quant/shtech_skd_config.yaml`。要点：`input_type_rt: 'nv12'`（不写 input_layout_rt，P2）、
  `input_type_train: 'rgb'`、`input_layout_train: 'NCHW'`、`norm_type: data_scale`、
  `scale_value: 1/255`（与 TRT 侧 `/255` 一致；AXCL 侧 uint8 图内归一化语义等价）。
- 校准集：`data/horizon_x5/data/calibration_data_512x640_u8/`，200 张 uint8 RGB NCHW。
  `scripts/prepare_calibration_data.py` 新增 `--resize-mode stretch`（直拉 resize，无 letterbox，
  与上科大 `detect/preprocess_submodule.cpp` 的 `cv::resize + BGR2RGB` 严格对齐）。
  源为 RM2025 公开数据集（min(w,h)≥360 过滤 + aHash 去重 + 亮度分桶分层抽样），合规依据 `docs/数据集使用规范.md`。
- 编译一次通过；`hb_model_info` 确认 `input source: HB_DNN_INPUT_FROM_PYRAMID`、
  `tensor type: HB_DNN_IMG_TYPE_NV12`、aligned byte size **491520 = 512×640×3/2**。
- bin：3.4 MB（AX650 原厂 axmodel 2.9 MB，同量级互证）。

## 4. host FP32 基准（P16：先基准后上板）

`scripts/shtech_fp32_baseline.py`：test.avi（1688 帧）均匀采样 120 帧，直拉 resize → RGB → /255 → ORT。
结果：**120/120 全部有检出**，max conf 0.912~0.925（top10 全 >0.91），每帧约 10 个候选@0.1，
top1 稳定为 tag=4 / color=1 / size=0。取 conf 最高的 3 帧为阳性样本
（`data/shtech_skd_positive/`：rgb/*.bin、nv12/*.bin、jpg/*.jpg + fp32_baseline.csv）。

## 5. 板端验收（P14：阳性对照，逐帧对比 host FP32）

板端 `~/rdk_trials/shtech_skd_512x640_20260909/`，`hrt_model_exec infer` 喂 NV12 bin，
输出 564480 字节 FP32 [1,6720,21]（无需反量化）。对比脚本 `scripts/shtech_board_compare.py`：

| 阳性帧 | conf host→板端 | 候选数 | 角点最大误差 | argmax (tag/color/size) | 判定 |
|---|---|---|---|---|---|
| 890 | 0.9247 → 0.8778 | 10 → 10 | **1.19 px** | 全一致 | ✅ PASS |
| 705 | 0.9222 → 0.8425 | 11 → 10 | **1.97 px** | 全一致 | ✅ PASS |
| 1160 | 0.9186 → 0.8649 | 10 → 10 | **1.34 px** | 全一致 | ✅ PASS |

conf 降幅 0.05~0.08（正常 PTQ 水平），3/3 PASS。**本次量化精度验收通过。**

## 6. 性能：仿真与实测拆分（P13 完整复现）

`hb_perf`：**FPS=328，latency 3.05 ms**，DDR 7.46 MB；NV12→YUV444 由 BPU 内部完成（虚拟节点 NV12TOYUV444）。

板端 `hrt_model_exec perf`（1000 帧，线程 1）：**平均 22.6 ms / 44.2 FPS**，min 15.0 ms。profiler 拆分：

| 项目 | 耗时 |
|---|---|
| **BPU 纯推理** | **3.74 ms**（与仿真吻合） |
| Transpose（输出 [1,6720,21]） | 10.9 ms |
| output_layout_convert | 7.4 ms |
| 各输出反量化 Dequantize ×8 | ~10.3 ms |
| Concat/Reshape 等 | ~1.8 ms |
| CPU 侧合计 | 30.4 ms |

结论与 P13 一致：**BPU 本体极快（3.7 ms），延迟全部耗在 runtime CPU 侧输出处理**（141k 元素的小张量花 28 ms，
同 rp_0526 的 554k→59 ms 同一实现问题）。缓解：C++ 部署时推理前申请 NHWC 输出（可省 layout convert + Transpose 约 18 ms），
反量化开销待向社区/runtime 版本确认。

## 7. NV12 色度域偏移评估（R1 风险解除）

`scripts/shtech_nv12_chroma_check.py`：host FP32 上对比"RGB 直接推理"vs"RGB 经 NV12（BT.601，4:2:0）往返后推理"，
test.avi 120 帧：

| 指标 | 结果 |
|---|---|
| top1 tag / color / size 一致率 | **120/120 = 100%** |
| conf 绝对差 | mean 0.0026 / p95 0.0131 / max 0.0394 |
| top1 角点误差 | mean 0.020 px / max 0.035 px |

NV12 的色度半分辨率采样对装甲板检测无可测影响，**确定最终输入格式为 nv12**。
（板端相机 BGR → NV12 转换约 1 ms CPU，或走 VPU/pyramid 硬件链路。）

## 8. 坑位与注意事项

0. **NV12 vs RGB 输入效率对照（2026-09-09 补充实验）**：同权重、同校准集，仅改 `input_type_rt`
   （`quant/shtech_skd_rgb_config.yaml`，`input_type_rt: rgb` + `input_layout_rt: NHWC`）编译对照模型实测：

   | 指标 | NV12 版 | RGB 版 | 说明 |
   |---|---|---|---|
   | BPU 仿真 latency | **3.05 ms** | 3.28 ms | NV12 的硬件 CSC 近乎免费，比 uint8 RGB 输入整理还快 |
   | 板端 BPU 纯推理（profiler） | **3.74 ms** | 3.92 ms | 转换开销不是问题 |
   | CPU 侧输出处理 | 30.4 ms | **15.7 ms** | 差距不在输入 CSC，而在输出端 Transpose/layout_convert/反量化路径（NV12 模型约为 RGB 版 2 倍） |
   | 板端端到端（perf 1000 帧） | 22.6 ms | **15.7 ms** | RGB 版当前更快，赢在 runtime CPU 侧；两者 min 均约 15 ms |
   | 输入数据量/帧 | 0.49 MB | 0.98 MB | 30 FPS 下带宽差异 ~15 MB/s，对 5.1 GB/s DDR 可忽略 |

   **结论**：① BPU 内部 NV12→RGB 硬件转换效率极高，无转换税；② 当前 runtime 版本下 NV12 模型的
   **输出端** CPU 处理路径更慢（P13 问题对 NV12 模型更严重），端到端 RGB 版暂时占优；
   ③ 按相机形态决策——USB BGR 相机（本项目大恒）短期直接用 RGB 版最省；MIPI/硬解码源原生 NV12
   零拷贝直通，省 CPU 转换 1~2 ms，但需先解决输出端开销（C++ 申请 NHWC 输出 / 反量化问题反馈社区）
   才能体现 NV12 端到端优势；④ 两个 bin 均已产出，按部署形态选择。

1. **校准预处理必须直拉**：上科大无 letterbox，角点按 src_roi 纯缩放还原；若沿用现有 letterbox 校准集会引入分布失配（最高优先坑 R2，已通过 `--resize-mode stretch` 解决）。
2. **NV12 不写 input_layout_rt**（P2）；板端喂数据大小 491520 字节（Y 平面 + UV 交错），可用 `bgr_to_nv12()`（`shtech_fp32_baseline.py`）生成，注意 OpenCV `BGR2YUV_I420` 输出需先 `reshape(-1)` 再切 U/V 平面。
3. **OPTEL 报告的逐层相似度不作验收依据**（P14）；本次以 host FP32 阳性对照为准。
4. `SZU0526` 的 FP16 checker 失败与 rp_0526 P2 同款；`convert_model_fp32.py` 可直接修复，但其混量纲输出仍需图手术才可安全量化。
5. 上科大 AXCL 后处理中 `color_id` 直接保留模型原始编号（0/1），MIGraphX(SZU) 路径才有红蓝交换——**对接后处理时不要混用两条路径的类别映射**。

## 9. 交付物清单

| 文件 | 说明 |
|---|---|
| `quant/shtech_skd_config.yaml` | NV12 量化配置 |
| `quant/shtech_skd/model_output/shtech_skd_512x640_nv12.bin` | 板端模型（3.4 MB） |
| `quant/shtech_skd/model_output/*_quantized_model.onnx` 等 | 量化中间产物 |
| `data/horizon_x5/data/calibration_data_512x640_u8/` | 200 张直拉校准集 |
| `data/onnx/shtech/SKD250526.onnx`、`SZU0526_fp32.onnx` | 量化对象与备选 |
| `data/shtech_skd_positive/` | host 基准 CSV + 3 张阳性样本（rgb/nv12/jpg）+ 板端输出 |
| `scripts/shtech_fp32_baseline.py` | host FP32 基准 + 阳性样本/NV12 bin 生成 |
| `scripts/shtech_board_compare.py` | 板端 vs host FP32 阳性对照 |
| `scripts/shtech_nv12_chroma_check.py` | NV12 色度域偏移统计 |
| 板端 | `sunrise@192.168.127.10:~/rdk_trials/shtech_skd_512x640_20260909/` |

## 10. 下一步

- C++ 部署：NHWC 输出申请省 ~18 ms（P13 缓解方案）；接入 CornerRefine（fitLine 精修）后量化误差（≤2 px）占精修容差 <10%。
- 扩充验收：真实相机实拍图 + 红方/灰方样本（test.avi 仅蓝方 tag=4 场景），color 分支需在多色样本上复核。
- 长时间稳定性与多线程帧率实测（当前未固定频率/温度）。
