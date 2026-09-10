# 产物身份清单（artifact manifest）

> 生成：2026-09-09。目的：量化调试期间发生过产物覆盖事故（O2 all_int16 bin 被 O0 编译覆盖），
> 此清单固定所有关键产物的 md5，保证对照实验的可复现性与身份唯一性。
> 重新生成任何产物后必须更新本清单。

## 一、源模型与图手术产物

| 文件 | md5 | 说明 |
|---|---|---|
| `data/onnx/sp_vision_25/rp24_0708.onnx` | bc9f2c32da164e29596807682460d07f | 深大 RP24 官方原始 FP32 ONNX（同济 yolov5.xml 同源，余弦 1.0001） |
| `data/onnx/rp_0526.onnx` | 45b111e2ddde9e0b588fef5a38917af4 | 深大 RP24 2025 版（rp_0526） |
| `data/onnx/rp_0526_split.onnx` | 5713f3c6ec768b1dfd4b57e6ccae9f96 | rp_0526 输出归一化 + 拆分输出版（输出端手术） |
| `data/onnx/sp_vision_25/rp24_0708_pad128.onnx` | 见 `md5sum data/onnx/sp_vision_25/rp24_0708_pad128.onnx` | head Conv 通道 66→128 pad 版（pad 实验用） |

## 二、量化 bin（quant/）

| 文件 | md5 | 配置 | 状态 |
|---|---|---|---|
| `quant/sp25_yolov5/model_output/sp25_yolov5_640x640.bin` | 74f2ac0bea60eb8fe0b049fb7f0acb5c | sp25_yolov5_config.yaml（INT8, mix） | 板端 conf 峰 0.500 格点化 |
| `quant/sp25_yolov5_all_int16_rebuild/model_output/sp25_yolov5_640x640_all_int16.bin` | 14ea38f76adf6f05fc4fe153a55cdb01 | sp25_yolov5_all_int16_rebuild_config.yaml（O2 all_int16，独立目录重建） | 板端 conf 被抑制（0.99→0.25/0.38） |
| `quant/sp25_yolov5_all_int16/model_output/sp25_yolov5_o0_allint16.bin` | a58a909e8ff0075492ab7b3030c473e8 | sp25_yolov5_o0_config.yaml（O0 + all_int16） | 板端 conf 峰 0.591 / 22 候选 |
| `quant/sp25_yolov5_int16_head/model_output/sp25_yolov5_640x640_int16_head.bin` | 47b4af22757d2ec55684a37ec6907abb | sp25_yolov5_int16_head_config.yaml（O2 int16_head） | conf cos 0.911，仍崩 |
| `quant/sp25_pad128/model_output/sp25_pad128_640x640.bin` | 43879b570a1e53c28caeaf0e2c17606b | sp25_pad128_config.yaml（O2 all_int16 + head 通道 pad 128） | conf 峰 0.561，未恢复 |
| `quant/rp0526/model_output/rp0526_640x640.bin` | 2b1ab450cde3d5b4a76bcb8a2084e819 | rp0526_config.yaml（O2 mix，rp_0526 模型） | 板端 conf 通道 cos 0.0073（对 FP32） |

## 三、配置与编译日志

| 文件 | md5 |
|---|---|
| `quant/sp25_yolov5_config.yaml` | 4d53e6cc9446da8a33bd6e1973eb4d2f |
| `quant/sp25_yolov5_all_int16_config.yaml` | a6684cca5cd73bade5f0c019a35f33a3 |
| `quant/sp25_yolov5_all_int16_rebuild_config.yaml` | 66f121617e86179fb4c0fe3341009ac6 |
| `quant/sp25_yolov5_o0_config.yaml` | 6aa4ea1308d45fdc0ad81107d98ed10a |
| `quant/sp25_pad128_config.yaml` | 1987e35c221831d55144416f67cd584a |
| `quant/rp0526_config.yaml` | 1fdfc576d055f79f5b4404ad0abc760a |
| 编译日志 | `quant/archive/20260909_pre_restore/*.log`（mk3~mk11、mk_o0、mk_pad、mk_restore） |

## 四、板端输出 dump（对照实验原始数据）

| 文件 | md5 | 内容 |
|---|---|---|
| `data/sp25_requant/board_C_INT8_{0001,0026,0118,0129,0173}.bin` | 见 `md5sum data/sp25_requant/board_C_INT8_*.bin` | INT8 bin 板端输出（fp32 554400 元素） |
| `data/sp25_requant/board_D_INT16_{0001,0026,0118,0129,0173}.bin` | 见 `md5sum data/sp25_requant/board_D_INT16_*.bin` | all_int16 bin 板端输出 |
| `data/sp25_requant/o0_armor.bin` | 见 `md5sum data/sp25_requant/o0_armor.bin` | O0+all_int16 板端输出（armor_test） |
| `data/sp25_requant/dump_int8/` | 目录 | INT8 bin 的 `--dump_intermediate` 逐层 dump（含 12 字节头 + INT16 的 BPU 输出、y_scale 0.2337287217、最终 fp32 输出） |

## 五、校准集

- 目录：`data/horizon_x5/data/calibration_data_640_u8/images/`（200 张 uint8 RGB NCHW 640×640 bin）
- 集合整体哈希：`md5sum data/horizon_x5/data/calibration_data_640_u8/images/*.bin | awk '{print $1}' | md5sum`（生成时记录）
- 来源：复用 rp_0526 校准集（深大/同济共用 RP24 0708 同源架构）；**适用性复核未完成**（见踩坑记录遗留 #6）

## 六、覆盖事故记录

- **事故**：O0+all_int16 编译时 working_dir/prefix 沿用了 all_int16 配置，**覆盖了 O2 all_int16 的 bin**（`sp25_yolov5_640x640_all_int16.bin` 被覆盖后更名为 `sp25_yolov5_o0_allint16.bin` 保留）。
- **处置**：覆盖前无哈希可比对，O2 all_int16 bin 以 `sp25_yolov5_all_int16_rebuild_config.yaml` 在独立目录**按 O2 配置重建**（`sp25_yolov5_all_int16_rebuild/model_output/`），如实记录为"重建产物"而非"恢复原产物"。
- **预防**：后续所有编译使用独立 working_dir + 唯一 prefix；产物生成后立即记录 md5 到本清单。

## 七、工具链版本

| 组件 | 版本 |
|---|---|
| host hb_mapper / hbdk | 1.24.3 / 3.49.15 |
| 板端 runtime（hrt_model_exec） | 1.24.5 |
| 板端系统 | RDK OS 3.5.0（aarch64） |
| 板端 BPU 频率 | 996/1000 MHz（performance governor） |
