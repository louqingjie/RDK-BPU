# 实验①：sp25_yolov5 逐阶段分通道误差报告

```
### 0173（FP32 conf 峰值 0.99）
- B_FLOAT: conf峰 0.990(ref 0.990) 候选 68(ref 68) top10重合 10/10 | cos: kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000 | top行 num 2(ref 2) color 0(ref 0)
- C_INT8: conf峰 0.500(ref 0.990) 候选 1200(ref 68) top10重合 0/10 | cos: kpt 0.9977 conf 0.9065 color 0.4846 num 0.7580 | top行 num 0(ref 4) color 0(ref 1)
- D_INT16: conf峰 0.249(ref 0.990) 候选 0(ref 68) top10重合 0/10 | cos: kpt 0.9979 conf 0.9229 color 0.5402 num 0.8009 | top行 num 4(ref 5) color 0(ref 2)

### 0118（FP32 conf 峰值 0.988）
- B_FLOAT: conf峰 0.988(ref 0.988) 候选 35(ref 35) top10重合 10/10 | cos: kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000 | top行 num 2(ref 2) color 0(ref 0)
- C_INT8: conf峰 0.500(ref 0.988) 候选 1200(ref 35) top10重合 0/10 | cos: kpt 0.9976 conf 0.9205 color 0.6052 num 0.7741 | top行 num 0(ref 4) color 0(ref 1)
- D_INT16: conf峰 0.111(ref 0.988) 候选 0(ref 35) top10重合 0/10 | cos: kpt 0.9978 conf 0.9417 color 0.6433 num 0.7926 | top行 num 4(ref 0) color 0(ref 2)

### 0129（FP32 conf 峰值 0.914）
- B_FLOAT: conf峰 0.914(ref 0.914) 候选 19(ref 19) top10重合 10/10 | cos: kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000 | top行 num 3(ref 3) color 1(ref 1)
- C_INT8: conf峰 0.500(ref 0.914) 候选 1200(ref 19) top10重合 0/10 | cos: kpt 0.9987 conf 0.9168 color 0.5685 num 0.7851 | top行 num 0(ref 0) color 0(ref 1)
- D_INT16: conf峰 0.380(ref 0.914) 候选 2(ref 19) top10重合 0/10 | cos: kpt 0.9988 conf 0.9256 color 0.5839 num 0.8269 | top行 num 5(ref 6) color 0(ref 1)

### 0026（FP32 conf 峰值 0.091）
- B_FLOAT: conf峰 0.091(ref 0.091) 候选 0(ref 0) top10重合 10/10 | cos: kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000 | top行 num 1(ref 1) color 1(ref 1)
- C_INT8: conf峰 0.500(ref 0.091) 候选 1200(ref 0) top10重合 0/10 | cos: kpt 0.9977 conf 0.9463 color 0.5970 num 0.7568 | top行 num 0(ref 3) color 0(ref 1)
- D_INT16: conf峰 0.006(ref 0.091) 候选 0(ref 0) top10重合 0/10 | cos: kpt 0.9978 conf 0.9599 color 0.5558 num 0.7336 | top行 num 8(ref 0) color 1(ref 2)

### 0001（FP32 conf 峰值 0.079）
- B_FLOAT: conf峰 0.079(ref 0.079) 候选 0(ref 0) top10重合 10/10 | cos: kpt 1.0000 conf 1.0000 color 1.0000 num 1.0000 | top行 num 3(ref 3) color 0(ref 0)
- C_INT8: conf峰 0.500(ref 0.079) 候选 1200(ref 0) top10重合 0/10 | cos: kpt 0.9982 conf 0.9459 color 0.4955 num 0.7642 | top行 num 0(ref 3) color 0(ref 1)
- D_INT16: conf峰 0.089(ref 0.079) 候选 0(ref 0) top10重合 0/10 | cos: kpt 0.9986 conf 0.9672 color 0.5179 num 0.8281 | top行 num 5(ref 4) color 0(ref 2)

```
