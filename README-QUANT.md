# RDK X5 神经网络量化环境（Dev Container）

基于地平线 OE（Open Explorer）v1.2.8 算法工具链，通过 VS Code Dev Container 复刻官方量化容器环境。目标硬件 RDK X5（BPU bayes-e 架构）。

## 环境组成

| 组件 | 说明 |
|---|---|
| 镜像 | `openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8`（约 11.2 GB，含 CUDA） |
| SDK 交付包 | `oe_sdk/` 下，挂载到容器 `/open_explorer`（含 model_zoo PTQ yaml 模板） |
| 数据目录 | `data/`，挂载到容器 `/data`（onnx / calibration_data / output） |
| 容器工作区 | 本目录挂载到容器 `/workspace`，`--gpus=all --shm-size=15g` |

## 启动步骤

1. 拉取镜像（国内网络较慢，耐心等待；也可用离线包方式，见下文）：

   ```powershell
   docker pull openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
   ```

2. 下载 OE SDK 交付包（断点续传，可重复执行）：

   ```powershell
   .\scripts\download_oe_sdk.ps1
   ```

3. VS Code 打开本目录，安装 Dev Containers 扩展后执行 **Dev Containers: Reopen in Container**。
   创建完成后自动运行 `.devcontainer/post_create.sh` 自检（hb_mapper 版本、GPU 透传、目录骨架）。

## 常用量化命令（容器内）

```bash
# 1. 模型兼容性检查
hb_mapper checker --model-type onnx --march bayes-e --model data/onnx/your_model.onnx

# 2. PTQ 量化编译（yaml 参考 oe_sdk/*/samples/ai_toolchain/model_zoo 中的模板）
hb_mapper makertbin --model-type onnx --config your_config.yaml

# 3. 性能分析
hb_perf data/output/model_output/your_model.bin
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/量化工作总结与bin清单.md](docs/量化工作总结与bin清单.md) | **全部量化工作总览：15 个编译 bin 成败判定、验证数据清单、20 项踩坑索引、遗留问题** |
| [docs/RDK_X5_部署评估报告.md](docs/RDK_X5_部署评估报告.md) | AT_NN_Detector 模型包初评 |
| [docs/RDK_X5_量化可行性评估_AT_NN_Detector.md](docs/RDK_X5_量化可行性评估_AT_NN_Detector.md) | 图结构/算子/动态范围实测 + 实机测试结果（14 章） |
| [docs/量化踩坑记录.md](docs/量化踩坑记录.md) | **量化全过程问题实录（FP16、输出混量纲、runtime 开销等 16 项）** |
| [docs/SHtech_SKD_NV12量化实录.md](docs/SHtech_SKD_NV12量化实录.md) | 上科大 SHtech SKD250526 NV12 量化全记录（3.74 ms BPU、3/3 阳性对照 PASS、NV12 色度影响评估） |
| [data/coord_trial_20260907/README.md](data/coord_trial_20260907/README.md) | CoordAtt 576×768 板端部署试验（int8 输入约定、12 路输出布局、板端-仿真逐元素一致、131.9 FPS @1000 帧） |
| [docs/同济sp_vision_25量化实录_通俗版.md](docs/同济sp_vision_25量化实录_通俗版.md) | 同济开源模型量化全复盘（零基础向：溯源、成功/失败案例与启示） |
| [docs/传统精修与PnP解算分析_上科大与同济方案.md](docs/传统精修与PnP解算分析_上科大与同济方案.md) | 强队传统精修方案解读 |
| [docs/数据集使用规范.md](docs/数据集使用规范.md) | 校准/验证数据集禁用与合规清单 |

## 实用脚本（scripts/）

| 脚本 | 用途 |
|---|---|
| `inspect_onnx.py` | ONNX 算子统计与 BPU 阻断项识别 |
| `analyze_cutpoint.py` | 主干/后处理分界与算力统计 |
| `probe_output_range.py` | 裁剪点张量动态范围与量化步长 |
| `convert_model_fp32.py` | FP16 模型全图转 FP32（makertbin 前置） |
| `convert_input_fp32.py` | 仅输入节点 FP16→FP32（图首 Cast 方案） |
| `normalize_rp_output.py` | 深大模型输出归一化图手术（拆分输出） |
| `prepare_calibration_data.py` | 校准集生成（分层抽样 + letterbox 预处理） |
| `verify_sp25_models.py` | 第三方 IR vs 原始 ONNX 双引擎同源验证（余弦 >0.999） |
| `prepare_sp25_models.py` | tiny_resnet 量化前处理（动态 batch 固化 + opset 17→11 降级） |
| `prepare_sp25_calibration.py` | sp25 分类器 32×32 灰度校准集（对齐 classifier.cpp 预处理） |
| `capture.cpp` | 大恒相机最小采集程序（板上 GxIAPI） |
| `real_camera_test.py` | 真实相机帧板端推理 + 后处理 + 标注 |
| `pnp_error_sim.py` | 角点量化误差 → PnP 解算误差蒙特卡洛仿真 |
| `shtech_fp32_baseline.py` | SHtech SKD host FP32 基准 + 阳性样本/NV12 bin 生成 |
| `shtech_board_compare.py` | SHtech SKD 板端 vs host FP32 阳性对照 |
| `shtech_nv12_chroma_check.py` | NV12 色度域偏移影响统计（BGR↔NV12 往返模拟） |

## 切换 CPU 镜像 / 离线兜底

- **CPU 版**：将 `.devcontainer/devcontainer.json` 中 image 改为
  `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`，并删除 `runArgs` 中的 `--gpus=all`。
- **Docker Hub 拉取过慢时**，下载离线包导入后重打标签：

  ```powershell
  curl.exe -L -C - -o docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
  docker load -i docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
  docker tag <load出的镜像名> openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
  ```
