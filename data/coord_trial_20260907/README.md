# CoordAtt 576×768 试验模型

2026-09-07。仅供链路验证与性能探索，尚未通过正式精度验收。

## 文件与部署

- 模型：`compiled/coord_trial_576x768.bin`，约 2.3 MiB。
- 配置：工作区 `quant/coord_trial_576x768.yaml`。
- 原模型：`AT_NN_Detector/common/praysky_coord_noe2e_0331_576x768.onnx`。
- 使用已经过浮点拆图等价检查的 12 输出 raw ONNX，移除解码/NMS，保留输出反量化。
- 工具：OE 1.2.8 / hb_mapper 1.24.3；板端 DNN 1.24.5 / HBRT 3.15.55.0，已实际加载运行。
- 板端目录：`sunrise@192.168.127.10:/home/sunrise/rdk_trials/coord_576x768_20260907/`。

## 输入输出约定（实测）

板端 `.bin` 输入是 RGB/NCHW `[1,3,576,768]`，共 1327104 字节。
传给 hrt_model_exec 的二进制内容必须是 RGB 像素减 128 后的 int8：

```python
rgb_chw = rgb_uint8.transpose(2, 0, 1)
board_input = (rgb_chw.astype(np.int16) - 128).astype(np.int8)
board_input.tofile('input.bin')
```

板端不要再除以 255，模型内置该归一化。先 letterbox 到 576×768，填充值 114，再转换颜色、布局及减 128。
虽然 model_info 显示 HB_DNN_IMG_TYPE_RGB，但裸 uint8 二进制在本次实测中不能得到正确输出。
`test_rgb_u8_chw.bin` / `positive_rgb_u8_chw.bin` 及旧 dump/infer/perf 文件是排查输入约定时的中间产物，不应作为部署示例。正确文件带 `s8` / `signed` 标记。

12 个输出均为 FP32/NCHW，无需用户再反量化；顺序按 stride=8/16/32，每个尺度为 box(4)、kpt(8)、color(4)、class(12)。完整名称、形状和 stride 见 model_info.log。
模型不直接输出 `[1,30,18]`，必须添加 CPU 解码和 NMS。PC 对照可使用 `quant/assessment_20260907/raw/praysky_coord_noe2e_0331_576x768_post.onnx`。

## 校准与验证结果

- 从三个随附视频等间隔抽取并做精确哈希去重，共 90 帧。
- 校准张量：RGB/CHW/float32，0–255；配置归一化 1/255；KL、O2、latency。
- manifest 与抽帧拼图见 calibration_manifest.json、calibration_contact_sheet.jpg。
- 视频含直播 UI、调试窗口和绘制框，仅能用于试验；不能据此评价正式 mAP/Recall。
- 两张输入（含一张有检出目标的样本），12 个板端输出与量化仿真逐元素一致，最大绝对误差 0。见 board_comparison.json、positive_board_comparison.json。
- 单正样本：浮点与量化均检出 1 个目标、类别 ID 均为 0；置信度从 0.61475 降至 0.39552。四角点相对浮点的欧氏误差约为 1.728、2.138、0.520、0.385 像素（网络坐标）。这不代表真实精度上限，样本也来自校准集。

## 板端性能

正确 signed 输入，单线程、1000 帧：平均 7.5618 ms，约 131.90 FPS；工具报告最小 4.844 ms、最大 10.190 ms。见 perf_signed_1000.log。
此值是 hrt_model_exec 推理循环表现（包含运行时输出处理），不是纯 BPU 核时间，也不包含相机、外部预处理、CPU 检测解码/NMS或 ROS 传输。未固定频率和温度，也未做长时间稳定性测试。

复现：

```bash
cd /home/sunrise/rdk_trials/coord_576x768_20260907
hrt_model_exec model_info --model_file coord_trial_576x768.bin
hrt_model_exec infer --model_file coord_trial_576x768.bin --input_file positive_rgb_s8_chw.bin --enable_dump true --dump_format bin --dump_path verify_dump
hrt_model_exec perf --model_file coord_trial_576x768.bin --input_file positive_rgb_s8_chw.bin --frame_count 1000 --thread_num 1
```

## 下一步

采集真实相机原图，建立独立验证集，重新校准并评价类别、颜色和角点/PnP。再接入 C++ 后处理，不能使用原 YOLOv8 DFL 和 raw*2+grid 解码。模型内网格解码为 raw+grid+0.5 后乘 stride。
