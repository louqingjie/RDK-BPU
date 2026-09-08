# tiny_resnet 32x32 灰度校准集

- 样本数: 400（200 张源图 x [整图缩放 + 随机裁剪]）
- 格式: uint8 [1,1,32,32] NCHW，0~255，BPU 内部 /255
- 预处理对齐 sp_vision_25 classifier.cpp（等比缩放贴左上角、黑底、灰度）

```yaml
calibration_parameters:
  cal_data_dir: '/workspace/workspace/data/horizon_x5/data/calibration_data_32gray/images'
  cal_data_type: 'uint8'
```
