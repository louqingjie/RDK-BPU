#!/usr/bin/env bash
# RDK X5 量化环境自检（Dev Container 创建后自动执行）
# 警告不中断：GPU 不可用仅影响量化校准速度，功能不受影响

echo "==== [1/4] Python ===="
python3 --version

echo "==== [2/4] hb_mapper ===="
hb_mapper --version || echo "[WARN] hb_mapper 不可用，请检查镜像"

echo "==== [3/4] GPU 透传检查 ===="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || echo "[WARN] nvidia-smi 执行失败，GPU 量化将回退 CPU"
else
    echo "[WARN] 容器内无 nvidia-smi，GPU 未透传；仅影响量化校准速度"
fi

echo "==== [4/5] 目录骨架 ===="
mkdir -p /workspace/data/onnx /workspace/data/calibration_data /workspace/data/output /workspace/data/horizon_x5/data
echo "data 目录骨架已就绪: onnx / calibration_data / output / horizon_x5/data（官方示例校准数据路径）"

# 修复 Windows 解压时无法创建的 6 个符号链接（指向均来自 SDK tar 清单）
echo "==== [5/5] 修复 SDK 符号链接 ===="
SDK=/open_explorer/D-Robotics_x5_open_explorer_v1.2.8-py310_20240926
if [ -d "$SDK/samples" ]; then
    ln -sfn ai_toolchain/model_zoo "$SDK/samples/model_zoo"
    ln -sfn ../../../model_zoo "$SDK/samples/ai_toolchain/horizon_model_convert_sample/01_common/model_zoo"
    ln -sfn /data/horizon_x5/data "$SDK/samples/ai_toolchain/horizon_model_convert_sample/01_common/data"
    ln -sfn ../../../model_zoo/runtime/horizon_runtime_sample "$SDK/samples/ai_toolchain/horizon_runtime_sample/x5/model/runtime"
    ln -sfn ../../../../model_zoo/runtime/ai_benchmark/qat "$SDK/samples/ai_benchmark/x5/qat/model/runtime"
    ln -sfn ../../../../model_zoo/runtime/ai_benchmark/ptq "$SDK/samples/ai_benchmark/x5/ptq/model/runtime"
    echo "6 处符号链接已补建"
else
    echo "[提示] /open_explorer 无 SDK，请在宿主机执行 scripts/download_oe_sdk.ps1"
fi

if [ -d /open_explorer ] && [ -n "$(ls -A /open_explorer 2>/dev/null)" ]; then
    echo "==== OE SDK (/open_explorer) ===="
    ls /open_explorer
else
    echo "[提示] /open_explorer 为空：请在宿主机执行 scripts/download_oe_sdk.ps1 下载 OE SDK"
fi
