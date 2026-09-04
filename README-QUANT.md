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

## 切换 CPU 镜像 / 离线兜底

- **CPU 版**：将 `.devcontainer/devcontainer.json` 中 image 改为
  `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`，并删除 `runArgs` 中的 `--gpus=all`。
- **Docker Hub 拉取过慢时**，下载离线包导入后重打标签：

  ```powershell
  curl.exe -L -C - -o docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/oe_x5/1.2.8/docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
  docker load -i docker_openexplorer_ubuntu_20_x5_gpu_v1.2.8.tar.gz
  docker tag <load出的镜像名> openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
  ```
