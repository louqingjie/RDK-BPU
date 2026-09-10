# 实验E：量化仿真 vs 板端分通道对照

```
### 0001（FP32 conf 峰值 0.079，候选 0）
- A_FP32       conf峰 0.079  候选    0  | kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000
- E_INT8_sim   conf峰 0.500  候选 1200  | kpt 0.9987 conf 0.9538 color 0.7964 num 0.8558
- E_INT16_sim  conf峰 0.050  候选    0  | kpt 0.9991 conf 0.9839 color 0.8762 num 0.8679
- C_INT8       conf峰 0.500  候选 1200  | kpt 0.9982 conf 0.9459 color 0.4955 num 0.7642
- D_INT16      conf峰 0.089  候选    0  | kpt 0.9986 conf 0.9672 color 0.5179 num 0.8281

### 0026（FP32 conf 峰值 0.091，候选 0）
- A_FP32       conf峰 0.091  候选    0  | kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000
- E_INT8_sim   conf峰 0.500  候选 1200  | kpt 0.9971 conf 0.9373 color 0.7186 num 0.7216
- E_INT16_sim  conf峰 0.249  候选    0  | kpt 0.9937 conf 0.9597 color 0.7475 num 0.7068
- C_INT8       conf峰 0.500  候选 1200  | kpt 0.9977 conf 0.9463 color 0.5970 num 0.7568
- D_INT16      conf峰 0.006  候选    0  | kpt 0.9978 conf 0.9599 color 0.5558 num 0.7336

### 0118（FP32 conf 峰值 0.988，候选 35）
- A_FP32       conf峰 0.988  候选   35  | kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000
- E_INT8_sim   conf峰 0.500  候选 1201  | kpt 0.9973 conf 0.9218 color 0.6913 num 0.7813
- E_INT16_sim  conf峰 0.298  候选    2  | kpt 0.9973 conf 0.9537 color 0.7393 num 0.7951
- C_INT8       conf峰 0.500  候选 1200  | kpt 0.9976 conf 0.9205 color 0.6052 num 0.7741
- D_INT16      conf峰 0.111  候选    0  | kpt 0.9978 conf 0.9417 color 0.6433 num 0.7926

### 0129（FP32 conf 峰值 0.914，候选 19）
- A_FP32       conf峰 0.914  候选   19  | kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000
- E_INT8_sim   conf峰 0.500  候选 1202  | kpt 0.9992 conf 0.9507 color 0.6258 num 0.8028
- E_INT16_sim  conf峰 0.380  候选    3  | kpt 0.9993 conf 0.9665 color 0.6479 num 0.8182
- C_INT8       conf峰 0.500  候选 1200  | kpt 0.9987 conf 0.9168 color 0.5685 num 0.7851
- D_INT16      conf峰 0.380  候选    2  | kpt 0.9988 conf 0.9256 color 0.5839 num 0.8269

### 0173（FP32 conf 峰值 0.990，候选 68）
- A_FP32       conf峰 0.990  候选   68  | kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000
- E_INT8_sim   conf峰 0.500  候选 1203  | kpt 0.9986 conf 0.9538 color 0.7532 num 0.8521
- E_INT16_sim  conf峰 0.351  候选    1  | kpt 0.9989 conf 0.9755 color 0.7924 num 0.8962
- C_INT8       conf峰 0.500  候选 1200  | kpt 0.9977 conf 0.9065 color 0.4846 num 0.7580
- D_INT16      conf峰 0.249  候选    0  | kpt 0.9979 conf 0.9229 color 0.5402 num 0.8009

## 仿真 vs 板端（同配置同输入）
- 0001 [E_INT8_s vs C_INT8]: kpt 0.9988 conf 0.9681 color 0.4689 num 0.7899  ❌不一致
- 0026 [E_INT8_s vs C_INT8]: kpt 0.9984 conf 0.9683 color 0.6289 num 0.8104  ❌不一致
- 0118 [E_INT8_s vs C_INT8]: kpt 0.9983 conf 0.9452 color 0.6173 num 0.8480  ❌不一致
- 0129 [E_INT8_s vs C_INT8]: kpt 0.9992 conf 0.9198 color 0.6447 num 0.8472  ❌不一致
- 0173 [E_INT8_s vs C_INT8]: kpt 0.9979 conf 0.9295 color 0.5925 num 0.8475  ❌不一致
- 0001 [E_INT16_ vs D_INT16]: kpt 0.9990 conf 0.9766 color 0.5484 num 0.8177  ❌不一致
- 0026 [E_INT16_ vs D_INT16]: kpt 0.9944 conf 0.9696 color 0.5950 num 0.7844  ❌不一致
- 0118 [E_INT16_ vs D_INT16]: kpt 0.9982 conf 0.9546 color 0.6445 num 0.8534  ❌不一致
- 0129 [E_INT16_ vs D_INT16]: kpt 0.9994 conf 0.9226 color 0.6607 num 0.8446  ❌不一致
- 0173 [E_INT16_ vs D_INT16]: kpt 0.9978 conf 0.9331 color 0.6179 num 0.8676  ❌不一致
```
