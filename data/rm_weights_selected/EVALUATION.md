# 装甲板 + 角点权重评估（第一批精选）

> 由 `scripts/select_armor_pose.py --apply` 生成。硬性要求：**① 装甲板权重**、**② 必须输出角点**。
> 从 `data/rm_weights/` 的 146 份中入选 **48 份**，合计约 258.9 MB；
> 排除 92 份（非装甲 / 仅检测框），另 6 份列「待验证」不入选。

## 一、置信度分档

| 档位 | 判据 |
|---|---|
| A 确认 | 元数据证明关键点回归（ONNX `task=pose` / PyTorch `train_args.task=pose` 且类别为装甲） |
| B 推定 | 无元数据，按输出通道拆出 4×2 角点分量 |
| C 继承 | IR / engine / axmodel，匹配到同源已确认模型 |

## 二、入选清单（按年份）

### 2024（11 份，75.8 MB）

| 文件 | 档位 | 证据 |
|---|---|---|
| `北京科技大学_YOLOv8_Pose_Armor_四点_train_best.pt` | A | PyTorch task=pose, names={0: 'B', 1: 'R'} |
| `北京科技大学_YOLOv8_Pose_Armor_四点_train_last.pt` | A | PyTorch task=pose, names={0: 'B', 1: 'R'} |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416.onnx` | B | 输出 [1, 3549, 25] = 4(bbox)+1(conf)+1(color)+4×2(角点)+11(num)，锚点 3549 |
| `深圳大学_YOLOv5_Pose_Armor_0708.onnx` | B | 输出 [1, 25200, 22] = 4(bbox)+1(conf)+1(color)+4×2(角点)+8(num)，锚点 25200 |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416_IR.bin` | C | 非可读权重，同源已确认（B 档）模型 `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416.onnx` |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416_IR.xml` | C | 非可读权重，同源已确认（B 档）模型 `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416.onnx` |
| `武汉科技大学_YOLOv5_Pose_Armor_0708_TensorRT.engine` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0708.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0708_SPR版.bin` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0708.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0708_SPR版.xml` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0708.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0708_同济版.bin` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0708.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0708_同济版.xml` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0708.onnx` |

### 2025（13 份，92.8 MB）

| 文件 | 档位 | 证据 |
|---|---|---|
| `RPS战队_EfficientNet_Pose_Armor_0717_eff2.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'Gb', 1: '1b', 2: '2b', 3: '3b', 4:  |
| `RPS战队_EfficientNet_Pose_Armor_0731_eff2.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'Gb', 1: '1b', 2: '2b', 3: '3b', 4:  |
| `上海科技大学_YOLOv5_Pose_Armor_SKD250526_512x640.onnx` | B | 输出 [1, 6720, 21] = 4(bbox)+1(conf)+1(color)+4×2(角点)+7(num)，锚点 6720 |
| `中国科学院大学_YOLOv5_Pose_Armor_RM4P_384x640.onnx` | B | 输出 [1, 15120, 20] = 4(bbox)+1(conf)+1(color)+4×2(角点)+6(num)，锚点 15120 |
| `浙江师范大学_YOLOv5_Pose_Armor_ZLion2025.onnx` | B | 输出 [1, 1008, 24] = 4(bbox)+1(conf)+1(color)+4×2(角点)+10(num)，锚点 1008 |
| `深圳大学_YOLOv5_Pose_Armor_0526.onnx` | B | 输出 [1, 25200, 22] = 4(bbox)+1(conf)+1(color)+4×2(角点)+8(num)，锚点 25200 |
| `深圳大学_YOLOv5_Pose_Armor_SZU0526_512x640.onnx` | B | 输出 [1, 20160, 22] = 4(bbox)+1(conf)+1(color)+4×2(角点)+8(num)，锚点 20160 |
| `上海科技大学_YOLOv5_Pose_Armor_SKD250526_AX650.axmodel` | C | 非可读权重，同源已确认（B 档）模型 `上海科技大学_YOLOv5_Pose_Armor_SKD250526_512x640.onnx` |
| `武汉科技大学_YOLOv5_Pose_Armor_0526_TensorRT.engine` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0526.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR.bin` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0526.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR.xml` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0526.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR_仲恺版.bin` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0526.onnx` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR_仲恺版.xml` | C | 非可读权重，同源已确认（B 档）模型 `深圳大学_YOLOv5_Pose_Armor_0526.onnx` |

### 2026（24 份，90.3 MB）

| 文件 | 档位 | 证据 |
|---|---|---|
| `RPS战队_EfficientNet_Pose_Armor_0516_eff4.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'Gb', 1: '1b', 2: '2b', 3: '3b', 4:  |
| `talos战队_YOLO26_Pose_Armor_best_640.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'Bs0', 1: 'Bs1', 2: 'Bs2', 3: 'Bs3', |
| `talos战队_YOLO26_Pose_Armor_best_dwconv_fp16.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'Bs0', 1: 'Bs1', 2: 'Bs2', 3: 'Bs3', |
| `中国科学院大学_YOLO11_Pose_Armor_6兵种_640.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'B1', 1: 'B3', 2: 'BS', 3: 'R1', 4:  |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_576x768.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 's0_o0', 1: 's0_o2', 2: 's0_o3', 3:  |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_640.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 's0_o0', 1: 's0_o2', 2: 's0_o3', 3:  |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_480x640.onnx` | A | Ultralytics task=pose, kpt_shape=[4, 2], names={0: 'class0', 1: 'class1', 2: 'class2',  |
| `武汉科技大学_YOLOX_Pose_Armor_opt0527_416.onnx` | B | 输出 [1, 3549, 21] = 4(bbox)+1(conf)+1(color)+4×2(角点)+7(num)，锚点 3549 |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` | B | 输出 [1, 3549, 21] = 4(bbox)+1(conf)+1(color)+4×2(角点)+7(num)，锚点 3549 |
| `talos战队_YOLO26_Pose_Armor_best_AX650.axmodel` | C | 非可读权重，同源已确认（A 档）模型 `talos战队_YOLO26_Pose_Armor_best_640.onnx` |
| `talos战队_YOLO26_Pose_Armor_best_QAT_u16_AX650.axmodel` | C | 非可读权重，同校同架构已确认（A 档）`talos战队_YOLO26_Pose_Armor_best_640.onnx`（同系列变体） |
| `talos战队_YOLO26_Pose_Armor_best_QAT_u8u16_AX650.axmodel` | C | 非可读权重，同校同架构已确认（A 档）`talos战队_YOLO26_Pose_Armor_best_640.onnx`（同系列变体） |
| `talos战队_YOLO26_Pose_Armor_best_final_AX650.axmodel` | C | 非可读权重，同校同架构已确认（A 档）`talos战队_YOLO26_Pose_Armor_best_640.onnx`（同系列变体） |
| `talos战队_YOLO26_Pose_Armor_best_sm87_TensorRT.engine` | C | 非可读权重，同校同架构已确认（A 档）`talos战队_YOLO26_Pose_Armor_best_640.onnx`（同系列变体） |
| `talos战队_YOLO26_Pose_Armor_best_u16attn_AX650.axmodel` | C | 非可读权重，同校同架构已确认（A 档）`talos战队_YOLO26_Pose_Armor_best_640.onnx`（同系列变体） |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_576x768_TensorRT.engine` | C | 非可读权重，同源已确认（A 档）模型 `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_576x768.onnx` |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_640_TensorRT.engine` | C | 非可读权重，同源已确认（A 档）模型 `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_640.onnx` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_TensorRT.engine` | C | 非可读权重，同源已确认（B 档）模型 `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn.bin` | C | 非可读权重，同源已确认（B 档）模型 `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn.param` | C | 非可读权重，同源已确认（B 档）模型 `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn_fp16.bin` | C | 非可读权重，同源已确认（B 档）模型 `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn_fp16.param` | C | 非可读权重，同源已确认（B 档）模型 `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_IR.bin` | C | 非可读权重，同源已确认（A 档）模型 `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_480x640.onnx` |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_IR.xml` | C | 非可读权重，同源已确认（A 档）模型 `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_480x640.onnx` |

## 三、可迁移性

| 格式 | 数量 | 说明 |
|---|---|---|
| `.axmodel` | 6 | 绑定平台，不可直接用于 RDK X5 |
| `.bin` | 8 | 可迁移，量化首选 |
| `.engine` | 6 | 绑定平台，不可直接用于 RDK X5 |
| `.onnx` | 18 | 可迁移，量化首选 |
| `.param` | 2 | 可迁移，量化首选 |
| `.pt` | 2 | 可迁移，量化首选 |
| `.xml` | 6 | 可迁移，量化首选 |

## 四、待验证（未入选）

| 文件 | 原因 |
|---|---|
| `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last.bin` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |
| `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last.xml` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |
| `同济大学_YOLO11_Pose_Armor_yolo11_int8.bin` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |
| `同济大学_YOLO11_Pose_Armor_yolo11_int8.xml` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |
| `深圳大学_YOLOv5_Pose_Armor_Infantry_v5n_20260725_IR.bin` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |
| `深圳大学_YOLOv5_Pose_Armor_Infantry_v5n_20260725_IR.xml` | 非 ONNX/PT，且匹配不到同源已确认模型（无同源 onnx） |

## 五、排除明细

| 文件 | 原因 |
|---|---|
| `RPS战队_MLP_Det_Cls_兵种_28x28_8类.onnx` | 非装甲板（task=Cls） |
| `东南大学_MSSSIM_Det_Compress_图传_生成器g_s_IR.bin` | 非装甲板（task=Compress） |
| `东南大学_MSSSIM_Det_Compress_图传_生成器g_s_IR.xml` | 非装甲板（task=Compress） |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_a_IR.bin` | 非装甲板（task=Compress） |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_a_IR.xml` | 非装甲板（task=Compress） |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_s_IR.bin` | 非装甲板（task=Compress） |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_s_IR.xml` | 非装甲板（task=Compress） |
| `东南大学_QVRF_Det_Compress_图传_192x384.onnx` | 非装甲板（task=Compress） |
| `东南大学_QVRF_Det_Compress_图传_448x896.onnx` | 非装甲板（task=Compress） |
| `东南大学_Real-ESRGAN_Det_Compress_图传_超分x4v3.pth` | 非装甲板（task=Compress） |
| `个人_hxfhxy_LeNet_Det_Cls_数字_28x28.onnx` | 非装甲板（task=Cls） |
| `个人_starrysky9959_CNN_Det_Cls_数字识别_state_dict.pt` | 非装甲板（task=Cls） |
| `个人_starrysky9959_CNN_Det_Cls_数字识别_权重.pt` | 非装甲板（task=Cls） |
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.bin` | 非装甲板（task=Rune） |
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.onnx` | 非装甲板（task=Rune） |
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.xml` | 非装甲板（task=Rune） |
| `中南大学_YOLOX_Pose_Rune_打符_480.bin` | 非装甲板（task=Rune） |
| `中南大学_YOLOX_Pose_Rune_打符_480.onnx` | 非装甲板（task=Rune） |
| `中南大学_YOLOX_Pose_Rune_打符_480.xml` | 非装甲板（task=Rune） |
| `中南大学_YOLOv10_Det_Armor_雷达_11类.onnx` | 仅输出检测框（Det），无角点 |
| `中南大学_YOLOv10_Det_Armor_雷达_11类.pt` | 仅输出检测框（Det），无角点 |
| `中南大学_YOLOv10_Det_Armor_雷达_11类_new2.pt` | 仅输出检测框（Det），无角点 |
| `中南大学_YOLOv10_Det_Car_雷达车辆.onnx` | 非装甲板（task=Car） |
| `中南大学_YOLOv10_Det_Car_雷达车辆.pt` | 非装甲板（task=Car） |
| `北京科技大学_CNN_Det_Cls_数字_20x28.onnx` | 非装甲板（task=Cls） |
| `北京科技大学_CNN_Det_Cls_数字_TensorRT.engine` | 非装甲板（task=Cls） |
| `北京科技大学_YOLOv8_Det_General_n_预训练_非装甲.pt` | 非装甲板（task=General） |
| `华东交通大学_FC_Det_Cls_数字_20x28_9类.onnx` | 非装甲板（task=Cls） |
| `华中科技大学_CNN_Det_Cls_数字_32x32_10类.onnx` | 非装甲板（task=Cls） |
| `华中科技大学_SVM_Det_Cls_数字_SVM.xml` | 非装甲板（task=Cls） |
| `华中科技大学_SVM_Det_Cls_数字_SVM_RBF.xml` | 非装甲板（task=Cls） |
| `华中科技大学_YOLOv10_Det_Armor_416_IR.bin` | 仅输出检测框（Det），无角点 |
| `华中科技大学_YOLOv10_Det_Armor_416_IR.xml` | 仅输出检测框（Det），无角点 |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green.onnx` | 仅输出检测框（Det），无角点 |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green_384_IR.bin` | 仅输出检测框（Det），无角点 |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green_384_IR.xml` | 仅输出检测框（Det），无角点 |
| `华北理工大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280_原始.pt` | 仅输出检测框（Det），无角点 |
| `华北理工大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280_原始.pt` | 非装甲板（task=Car） |
| `华南师范大学_MLP_Det_Cls_数字_20x28_9类.onnx` | 非装甲板（task=Cls） |
| `华南师范大学_MLP_Det_Cls_数字_20x28_9类_微调.onnx` | 非装甲板（task=Cls） |
| `厦门理工学院_YOLOv5_Det_Armor_雷达_640_11类.onnx` | 仅输出检测框（Det），无角点 |
| `厦门理工学院_YOLOv5_Det_Car_雷达_640_5类.onnx` | 非装甲板（task=Car） |
| `同济大学_ResNet_Det_Cls_数字_32x32_9类.onnx` | 非装甲板（task=Cls） |
| `同济大学_YOLO11_Pose_Rune_buff_int8.bin` | 非装甲板（task=Rune） |
| `同济大学_YOLO11_Pose_Rune_buff_int8.xml` | 非装甲板（task=Rune） |
| `同济大学_YOLOv8_Pose_Rune_5点_best2-sim.onnx` | 非装甲板（task=Rune） |
| `复旦大学_MobileNetV2_Det_Backbone_特征提取.pth` | 非装甲板（task=Backbone） |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类.onnx` | 仅输出检测框（Det），无角点 |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类.pt` | 仅输出检测框（Det），无角点 |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类_TensorRT.engine` | 仅输出检测框（Det），无角点 |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280.onnx` | 非装甲板（task=Car） |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280.pt` | 非装甲板（task=Car） |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280_TensorRT.engine` | 非装甲板（task=Car） |
| `常州大学_RepVGG_Pose_Rune_buff_repvgg_9点.bin` | 非装甲板（task=Rune） |
| `常州大学_RepVGG_Pose_Rune_buff_repvgg_9点.xml` | 非装甲板（task=Rune） |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.bin` | 非装甲板（task=Rune） |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.onnx` | 非装甲板（task=Rune） |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.xml` | 非装甲板（task=Rune） |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_416.onnx` | 非装甲板（task=Rune） |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_416_fp16.onnx` | 非装甲板（task=Rune） |
| `武汉科技大学_RepVGG_Pose_Rune_cb_rune_9点.onnx` | 非装甲板（task=Rune） |
| `武汉科技大学_RepVGG_Pose_Rune_cb_rune_9点_TensorRT.engine` | 非装甲板（task=Rune） |
| `武汉科技大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280.onnx` | 仅输出检测框（Det），无角点 |
| `武汉科技大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280_TensorRT.engine` | 仅输出检测框（Det），无角点 |
| `武汉科技大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280.onnx` | 非装甲板（task=Car） |
| `武汉科技大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280_TensorRT.engine` | 非装甲板（task=Car） |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类.onnx` | 仅输出检测框（Det），无角点 |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类.pt` | 仅输出检测框（Det），无角点 |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类_TensorRT.engine` | 仅输出检测框（Det），无角点 |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类_batch_TensorRT.engine` | 仅输出检测框（Det），无角点 |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类.onnx` | 非装甲板（task=Car） |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类.pt` | 非装甲板（task=Car） |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类_TensorRT.engine` | 非装甲板（task=Car） |
| `浙江大学_YOLO11_Pose_Rune_buff_9点.onnx` | 非装甲板（task=Rune） |
| `深圳大学_Caffe-ArmorNet_Det_Cls_装甲数字_200000.caffemodel` | 非装甲板（task=Cls） |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000.caffemodel` | 非装甲板（task=Cls） |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000_加负样本.caffemodel` | 非装甲板（task=Cls） |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000_旋转.caffemodel` | 非装甲板（task=Cls） |
| `深圳大学_Caffe-LeNet_Det_Cls_装甲数字_200000.caffemodel` | 非装甲板（task=Cls） |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_480x640.onnx` | 非装甲板（task=Rune） |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_IR.bin` | 非装甲板（task=Rune） |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_IR.xml` | 非装甲板（task=Rune） |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_235000.weights` | 非装甲板（task=Rune） |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_80000.weights` | 非装甲板（task=Rune） |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_80000_单通道.weights` | 非装甲板（task=Rune） |
| `深圳职业技术大学_YOLO11_Det_Armor_能量单元6dof.pt` | 仅输出检测框（Det），无角点 |
| `西北工业大学_YOLOv8_Pose_AntiDrone_反无人机.pt` | 非装甲板（task=AntiDrone） |
| `西南石油大学_PredNet_Det_Solver_弹道预测_IR.bin` | 非装甲板（task=Solver） |
| `西南石油大学_PredNet_Det_Solver_弹道预测_IR.xml` | 非装甲板（task=Solver） |
| `通用_LeNet_Det_Cls_数字_28x28_9类.onnx` | 非装甲板（task=Cls） |
| `通用_YOLOv8_Pose_General_n-pose_COCO人体姿态预训练.pt` | 非装甲板（task=General） |
| `野狼战队_LeNet_Det_Cls_MNIST_28x28.onnx` | 非装甲板（task=Cls） |
