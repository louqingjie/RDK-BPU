# RM 权重重组清单（按年份 / 学校 / 网络 / Pose / Armor）

> 由 `scripts/organize_rm_weights.py --apply` 自动生成，请勿手工编辑。
> 去重后共 146 份内容；年份分档：2020, 2021, 2023, 2024, 2025, 2026。
> 命名：`<学校>_<网络版本>_<Pose|Det>_<Armor|Rune|Radar|Car|Cls|Solver|General>_<细节>.<ext>`

## 2020（10 份，约 14.8 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `个人_starrysky9959_CNN_Det_Cls_数字识别_state_dict.pt` | 1 | starrysky9959/Digital-recognition | `2191d7398e9e587da062d5ba173c55e9` |
| `个人_starrysky9959_CNN_Det_Cls_数字识别_权重.pt` | 1 | starrysky9959/Digital-recognition | `c01a6fcb23460be0f31528f1eb5054ef` |
| `深圳大学_Caffe-ArmorNet_Det_Cls_装甲数字_200000.caffemodel` | 1 | yarkable/RP_Infantry_Plus | `5e16081f57bff6077dd692c19b6ce783` |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000.caffemodel` | 1 | yarkable/RP_Infantry_Plus | `0743c26372e8ea6671ccd4150100d14a` |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000_加负样本.caffemodel` | 1 | yarkable/RP_Infantry_Plus | `815e0a331ac30851984054c9cb171fea` |
| `深圳大学_Caffe-LeNet_Det_Cls_打符数字_80000_旋转.caffemodel` | 1 | yarkable/RP_Infantry_Plus | `d2d2f3853b57f4826df4995b06871a71` |
| `深圳大学_Caffe-LeNet_Det_Cls_装甲数字_200000.caffemodel` | 1 | yarkable/RP_Infantry_Plus | `a6a7fb1d6b1ea6f6e1f9a317b9466982` |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_235000.weights` | 1 | yarkable/RP_Infantry_Plus | `fb95d50cd9908fdefea814487257e15d` |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_80000.weights` | 1 | yarkable/RP_Infantry_Plus | `db5fe9984bbc7d7ed233c3e2aea9bd21` |
| `深圳大学_tiny-YOLOv2_Det_Rune_打符_80000_单通道.weights` | 1 | yarkable/RP_Infantry_Plus | `8ab91ca39ece2191a257f2febe5379d9` |

## 2021（1 份，约 0.0 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `野狼战队_LeNet_Det_Cls_MNIST_28x28.onnx` | 1 | wildwolf-team/WolfVision | `d7cd24a0a76cd492f31065301d468c3d` |

## 2023（10 份，约 14.3 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `个人_hxfhxy_LeNet_Det_Cls_数字_28x28.onnx` | 1 | hxfhxy/RoboMaster_vision_Armor | `d1092b760d0eb7fa63e26e05fa981576` |
| `华东交通大学_FC_Det_Cls_数字_20x28_9类.onnx` | 1 | ecjtu-cx/ECJTU_RM_Vision | `47b84781290fdec3193d2e31a355ac70` |
| `华南师范大学_MLP_Det_Cls_数字_20x28_9类.onnx` | 13 | 0X5A0X480X52/hfut_rm_auto_aim_ws、Blackjack200/talos_26、CSU-FYT-Vision/FYT2024_vision 等13个 | `5f3b50536bb7b731a78ffb41507a3396` |
| `华南师范大学_MLP_Det_Cls_数字_20x28_9类_微调.onnx` | 2 | Blackjack200/talos_26、WUST-RM/awakening | `aff7515bcb71ce9a2333a41735ccc58b` |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.bin` | 1 | HDU-PHOENIX/rm_auto_aim | `3b49cb46fbf00651eb1efa075faac57d` |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.onnx` | 1 | HDU-PHOENIX/rm_auto_aim | `2e75806c74656aa00110d70b57b1246e` |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_2023.xml` | 1 | HDU-PHOENIX/rm_auto_aim | `a7f75c56203544bee38dc789260aad73` |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_416.onnx` | 1 | HDU-PHOENIX/rm_auto_aim | `44f06579a64ab3d9638a7ec235486bc4` |
| `杭州电子科技大学_YOLOX_Pose_Rune_打符_416_fp16.onnx` | 1 | HDU-PHOENIX/rm_auto_aim | `c09a1c4ca0486fb41ec9a109436af26b` |
| `通用_LeNet_Det_Cls_数字_28x28_9类.onnx` | 6 | 0X5A0X480X52/hfut_rm_auto_aim_ws、Blackjack200/talos_26、CSU-FYT-Vision/FYT2024_vision 等6个 | `f8297789bb646fef48f4618f3f303d82` |

## 2024（38 份，约 214.1 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.bin` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `104f48fbcfba7e36a224c8c6a29983d6` |
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.onnx` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `7a0c6fc59093c9ac0b3111b113f5367b` |
| `中南大学_YOLOX_Pose_Rune_打符_3.6m_480.xml` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `83637a0f6ea02bd2898f3caa9971f8a3` |
| `中南大学_YOLOX_Pose_Rune_打符_480.bin` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `a81a50e20023882d270b2ed4d088587a` |
| `中南大学_YOLOX_Pose_Rune_打符_480.onnx` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `6205323336980f411ecac0120320ad70` |
| `中南大学_YOLOX_Pose_Rune_打符_480.xml` | 2 | CSU-FYT-Vision/FYT2024_vision、SPR-Algorithm/SPR-Vision-2025 | `ed887620b50eb7d73afa1158f89e4c7f` |
| `中南大学_YOLOv10_Det_Armor_雷达_11类.onnx` | 1 | baiyeweiguang/FYT2024_radar | `d1f4b37cc024023b96f4dc0980288120` |
| `中南大学_YOLOv10_Det_Armor_雷达_11类.pt` | 1 | baiyeweiguang/FYT2024_radar | `21caf5e99ab9eba53e45f81500490c12` |
| `中南大学_YOLOv10_Det_Armor_雷达_11类_new2.pt` | 1 | baiyeweiguang/FYT2024_radar | `5149933bbdfc8df2023465cfc2370b88` |
| `中南大学_YOLOv10_Det_Car_雷达车辆.onnx` | 1 | baiyeweiguang/FYT2024_radar | `51489c78564272ef2b857aa1aa94b1d9` |
| `中南大学_YOLOv10_Det_Car_雷达车辆.pt` | 1 | baiyeweiguang/FYT2024_radar | `3d74085d59f01e4379b5aaef9c8d0567` |
| `北京科技大学_CNN_Det_Cls_数字_20x28.onnx` | 2 | RebornVision/Reborn-Vision-2024-armor-Inference、WUST-RM/awakening | `722d8f148c33d913682373908da5b83c` |
| `北京科技大学_CNN_Det_Cls_数字_TensorRT.engine` | 1 | WUST-RM/awakening | `89216c26839d0836876a36497b57308f` |
| `北京科技大学_YOLOv8_Det_General_n_预训练_非装甲.pt` | 1 | gaoxinstudent/RM-Armor-Detect | `0305608151dd1725c9d7da6882ae52d5` |
| `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last.bin` | 4 | GKD-RM-Lab/gkd_vision_26、RebornVision/Reborn-Vision-2024-armor-Inference、SHM-white/RM2026-AutoAim 等4个 | `387d38640e603fc9ed756cf547cc6206` |
| `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last.xml` | 4 | GKD-RM-Lab/gkd_vision_26、RebornVision/Reborn-Vision-2024-armor-Inference、SHM-white/RM2026-AutoAim 等4个 | `dc817fa0cf06f00181caec4276d86aa5` |
| `北京科技大学_YOLOv8_Pose_Armor_n-pose_预训练.pt` | 1 | gaoxinstudent/RM-Armor-Detect | `0303686c8ec6aff7c7d13399de8cb667` |
| `北京科技大学_YOLOv8_Pose_Armor_四点_train_best.pt` | 1 | gaoxinstudent/RM-Armor-Detect | `e55bf9f8f98a97528f7e8d97cae0b96b` |
| `北京科技大学_YOLOv8_Pose_Armor_四点_train_last.pt` | 1 | gaoxinstudent/RM-Armor-Detect | `ed7b8d815f86376c2da434550bcbde9b` |
| `华中科技大学_CNN_Det_Cls_数字_32x32_10类.onnx` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `31ffa9fb58f4fcb0e97301ffa8b1bb8c` |
| `华中科技大学_SVM_Det_Cls_数字_SVM.xml` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `47fa0b165806dc7cf4c8cac061a061ba` |
| `华中科技大学_SVM_Det_Cls_数字_SVM_RBF.xml` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `de16943756f5d9471f230aa5b42ae3fe` |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416.onnx` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `eaf4aaf94b51cec67d9825ae9d9a7f7f` |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416_IR.bin` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `3d448f37e79396a3f67930fd7a87a480` |
| `华中科技大学_YOLOX_Pose_Armor_英雄自瞄_12类_416_IR.xml` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `a2193e389c2ed5d556477516e96780f5` |
| `华中科技大学_YOLOv10_Det_Armor_416_IR.bin` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `a588a5810a35132bfd789b0ff0c68b9e` |
| `华中科技大学_YOLOv10_Det_Armor_416_IR.xml` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `1f3edbfba07ce4644b9440d36bf587b7` |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green.onnx` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `9950db533c5a7dea1b6c083c58bf88e7` |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green_384_IR.bin` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `5ad819cd4aae96ea8834b00825ab6a56` |
| `华中科技大学_YOLOv8_Det_Armor_前哨站_green_384_IR.xml` | 1 | HUSTLYRM/HUST_HeroAim_2024 | `2bfd909ab792d179f6bd9c86b5ccc8a5` |
| `厦门理工学院_YOLOv5_Det_Armor_雷达_640_11类.onnx` | 1 | CarryzhangZKY/pfa_vision_radar | `0ba1594a67971d31da2c2af81bff94bd` |
| `厦门理工学院_YOLOv5_Det_Car_雷达_640_5类.onnx` | 1 | CarryzhangZKY/pfa_vision_radar | `5db3763286c97f99e750a885bec1c3fa` |
| `武汉科技大学_YOLOv5_Pose_Armor_0708_TensorRT.engine` | 1 | WUST-RM/awakening | `704e10a31b2cfed2fdde792b5c3b0152` |
| `深圳大学_YOLOv5_Pose_Armor_0708.onnx` | 5 | Alliance-Algorithm/rmcs_auto_aim_v2、SPR-Algorithm/SPR-Vision-2025、TOL0VE/RP24-DectionModel 等5个 | `bc9f2c32da164e29596807682460d07f` |
| `深圳大学_YOLOv5_Pose_Armor_0708_SPR版.bin` | 1 | 本地复制自 2024/深圳大学_YOLOv5_Pose_Armor_0708_同济版.bin | `-` |
| `深圳大学_YOLOv5_Pose_Armor_0708_SPR版.xml` | 1 | SPR-Algorithm/SPR-Vision-2025 | `5a815d6d2681bfb4caa7ac0bead11382` |
| `深圳大学_YOLOv5_Pose_Armor_0708_同济版.bin` | 5 | Alliance-Algorithm/rmcs_auto_aim_v2、GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim 等5个 | `e5e7a0b3ce1f134ecec20bc1071c8450` |
| `深圳大学_YOLOv5_Pose_Armor_0708_同济版.xml` | 4 | Alliance-Algorithm/rmcs_auto_aim_v2、GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim 等4个 | `9bbb1ec8c82681b5e4f1e0d8152cfc81` |

## 2025（24 份，约 168.2 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `RPS战队_EfficientNet_Pose_Armor_0717_eff2.onnx` | 1 | aiyo472/auto_aim_rps26 | `b9523e574d3458b541cac9fad7cbc037` |
| `RPS战队_EfficientNet_Pose_Armor_0731_eff2.onnx` | 1 | aiyo472/auto_aim_rps26 | `88baf224fc2c2e70adaf14001dd5d4f8` |
| `上海科技大学_YOLOv5_Pose_Armor_SKD250526_512x640.onnx` | 1 | Astra-Whale/SHtech_auto_aim | `8b35bdc332b0f7547bb170a479378605` |
| `上海科技大学_YOLOv5_Pose_Armor_SKD250526_AX650.axmodel` | 1 | Astra-Whale/SHtech_auto_aim | `f7ccb3ad393e7c4fdd1656e5fc4301d7` |
| `中国科学院大学_YOLOv5_Pose_Armor_RM4P_384x640.onnx` | 2 | ucas-sas-robot-team/RMVision-2026、ucas-sas-robot-team/RMVision-RM4P-2025 | `fd90aa0a05c041ca1eb154fd210f54c9` |
| `同济大学_ResNet_Det_Cls_数字_32x32_9类.onnx` | 6 | Blackjack200/talos_26、GKD-RM-Lab/gkd_vision_26、Justin186/vision_assessment_2026 等6个 | `9241dff88f59b4d176dcd8a4ab0287ff` |
| `同济大学_YOLO11_Pose_Armor_yolo11_int8.bin` | 3 | GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim、TongjiSuperPower/sp_vision_25 | `1650f8118afb4c66f03861b52d9a09eb` |
| `同济大学_YOLO11_Pose_Armor_yolo11_int8.xml` | 3 | GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim、TongjiSuperPower/sp_vision_25 | `e782601df538a8992ab199f381e2c773` |
| `同济大学_YOLO11_Pose_Rune_buff_int8.bin` | 3 | GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim、TongjiSuperPower/sp_vision_25 | `4182bd8379b606eebe7c41d4f66451f4` |
| `同济大学_YOLO11_Pose_Rune_buff_int8.xml` | 3 | GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim、TongjiSuperPower/sp_vision_25 | `e7612ebd531799a996d527d7e8049604` |
| `同济大学_YOLOv8_Pose_Rune_5点_best2-sim.onnx` | 3 | GKD-RM-Lab/gkd_vision_26、SHM-white/RM2026-AutoAim、TongjiSuperPower/sp_vision_25 | `ae75f0e662566f5ce2767dc2e822a2fe` |
| `武汉科技大学_YOLOv5_Pose_Armor_0526_TensorRT.engine` | 1 | WUST-RM/awakening | `b027dab3d4dc4005d3ac3d6aab84d8ce` |
| `浙江大学_YOLO11_Pose_Rune_buff_9点.onnx` | 1 | IC-Alan/HWauto_buff2026 | `7b7f35797cc28e602c3e9acfcf2425ed` |
| `浙江师范大学_YOLOv5_Pose_Armor_ZLion2025.onnx` | 1 | dielivelr/Z_LION_AutoAim2025 | `5e3a4336f0a185b0a26c2f57da4bb1a1` |
| `深圳大学_YOLOv5_Pose_Armor_0526.onnx` | 4 | Alliance-Algorithm/rmcs_auto_aim_v2、WUST-RM/awakening、broalantaps/RobotDetectionModel 等4个 | `45b111e2ddde9e0b588fef5a38917af4` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR.bin` | 1 | Justin186/vision_assessment_2026 | `5c2d9a0ea455245b3857b8005d8da8c1` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR.xml` | 1 | Justin186/vision_assessment_2026 | `59924218f3115a93c6d0fdc19e6956c5` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR_仲恺版.bin` | 1 | 本地复制自 2025/深圳大学_YOLOv5_Pose_Armor_0526_IR.bin | `5c2d9a0ea455245b3857b8005d8da8c1` |
| `深圳大学_YOLOv5_Pose_Armor_0526_IR_仲恺版.xml` | 1 | NOMANE-0/QD_Vision2026 | `4eefe6222d0c58bc6cacd36c67a3eed8` |
| `深圳大学_YOLOv5_Pose_Armor_Infantry_v5n_20260725_IR.bin` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `aff8745f9f359a82150bf6cf2d1556de` |
| `深圳大学_YOLOv5_Pose_Armor_Infantry_v5n_20260725_IR.xml` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `651e343caadbc1bfb468cfea3ff9313f` |
| `深圳大学_YOLOv5_Pose_Armor_SZU0526_512x640.onnx` | 1 | Astra-Whale/SHtech_auto_aim | `ce637d8cf12b9d185a12e801b2a2d9cb` |
| `西南石油大学_PredNet_Det_Solver_弹道预测_IR.bin` | 1 | soloplayl/Total3DAutoAim_md | `1b0c16c24c7f4459dd6960242e7a9eb9` |
| `西南石油大学_PredNet_Det_Solver_弹道预测_IR.xml` | 1 | soloplayl/Total3DAutoAim_md | `db1c1b2f7075c7035fe43b4e3de91986` |

## 2026（63 份，约 633.0 MB）

| 文件 | 来源数 | 来源仓库 | md5 |
|---|---|---|---|
| `RPS战队_EfficientNet_Pose_Armor_0516_eff4.onnx` | 1 | aiyo472/auto_aim_rps26 | `d7189fa9689d4a737bd06e4ec6851702` |
| `RPS战队_MLP_Det_Cls_兵种_28x28_8类.onnx` | 1 | aiyo472/auto_aim_rps26 | `beeead1711a6e5ec3dfa961d73d9612b` |
| `talos战队_YOLO26_Pose_Armor_best_640.onnx` | 1 | Blackjack200/talos_26 | `b27f8ab2190e121ae6b86960cd071df0` |
| `talos战队_YOLO26_Pose_Armor_best_AX650.axmodel` | 1 | Blackjack200/talos_26 | `9d9f125ec6139930aebda20c498ea182` |
| `talos战队_YOLO26_Pose_Armor_best_QAT_u16_AX650.axmodel` | 1 | Blackjack200/talos_26 | `a879e50a86d0f13c509cd3c53940fed3` |
| `talos战队_YOLO26_Pose_Armor_best_QAT_u8u16_AX650.axmodel` | 1 | Blackjack200/talos_26 | `0984c79de6e5d257ab71c13f9e59598c` |
| `talos战队_YOLO26_Pose_Armor_best_dwconv_fp16.onnx` | 1 | Blackjack200/talos_26 | `43d97b651288b5405c138d72f24b1241` |
| `talos战队_YOLO26_Pose_Armor_best_final_AX650.axmodel` | 1 | Blackjack200/talos_26 | `0bf1d94eef09a7a72d97e1f1ff544600` |
| `talos战队_YOLO26_Pose_Armor_best_sm87_TensorRT.engine` | 1 | Blackjack200/talos_26 | `e4697c1085e746dd262620974612ef9a` |
| `talos战队_YOLO26_Pose_Armor_best_u16attn_AX650.axmodel` | 1 | Blackjack200/talos_26 | `cbede5829d382ab786a10d92078209c1` |
| `东南大学_MSSSIM_Det_Compress_图传_生成器g_s_IR.bin` | 1 | ElainaXD/rm_compress | `95e29c3a8091dd4d1fbdc7f68b18f9cc` |
| `东南大学_MSSSIM_Det_Compress_图传_生成器g_s_IR.xml` | 1 | ElainaXD/rm_compress | `3daedad1591b1d26395f4d7dff4fd92a` |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_a_IR.bin` | 1 | ElainaXD/rm_compress | `41b6b514f0a352ea782cb0122e1763cf` |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_a_IR.xml` | 1 | ElainaXD/rm_compress | `b23f0ebbf449688ccc2af0a99f55b1d7` |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_s_IR.bin` | 1 | ElainaXD/rm_compress | `87e98861b85b6354f1c92d0ec1b11b70` |
| `东南大学_MSSSIM_Det_Compress_图传_超先验h_s_IR.xml` | 1 | ElainaXD/rm_compress | `7770170f1ade65e2dbadb56acc1ca429` |
| `东南大学_QVRF_Det_Compress_图传_192x384.onnx` | 1 | ElainaXD/rm_compress | `4a9e5210eda9fd9790ab968863891dc1` |
| `东南大学_QVRF_Det_Compress_图传_448x896.onnx` | 1 | ElainaXD/rm_compress | `81befd542ac436f70db6a061f74d69bf` |
| `东南大学_Real-ESRGAN_Det_Compress_图传_超分x4v3.pth` | 1 | ElainaXD/rm_compress | `91a7644643c884ee00737db24e478156` |
| `中国科学院大学_YOLO11_Pose_Armor_6兵种_640.onnx` | 1 | ucas-sas-robot-team/RMVision-2026 | `7a574d032b6e55f669d15f26d2193fe3` |
| `华北理工大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280_原始.pt` | 1 | github-release/BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025/Horizon-Armor-yolov8n-ghost-p2-2026-04-04.pt | `27669de4d0eda4bb6b360b81dede7cf3` |
| `华北理工大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280_原始.pt` | 1 | github-release/BreCaspian/ROBOMASTER-HORIZON-LiDAR-2025/Horizon-Car-yolov8s-ghost-p2-2026-04-06.pt | `45a4074622b1e8f0782f83666f21700b` |
| `复旦大学_MobileNetV2_Det_Backbone_特征提取.pth` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `2a138f4f0208ade5ef46d0820d33f243` |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类.onnx` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `37926d71191cfe15e12a2d29979cd9d6` |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类.pt` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `32fa0469bda05441ac8c61f0c78c5a01` |
| `复旦大学_YOLOv12_Det_Armor_雷达_3类_TensorRT.engine` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `953a81ac814bdc9a3785d59a87d72ce6` |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280.onnx` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `c275a9abb4d598c1b51958e0bc43d378` |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280.pt` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `05c20e973df4ed0f80c35cf87910ba1e` |
| `复旦大学_YOLOv12_Det_Car_雷达车辆_1280_TensorRT.engine` | 1 | Ryaxwn7/EGA_Radar_Algorithm | `92acce3d28f63123e9ef7a6303488bb2` |
| `常州大学_RepVGG_Pose_Rune_buff_repvgg_9点.bin` | 1 | CCZU-Climber/Climber_Vision_26 | `6f723cab8d53a4aab0ce5600a51c9ed8` |
| `常州大学_RepVGG_Pose_Rune_buff_repvgg_9点.xml` | 1 | CCZU-Climber/Climber_Vision_26 | `ce2e554c13ea1bbca1d49aaeba73d1c0` |
| `武汉科技大学_RepVGG_Pose_Rune_cb_rune_9点.onnx` | 1 | WUST-RM/awakening | `e9c1fa581f0bb9ecba00a9404c8ff69f` |
| `武汉科技大学_RepVGG_Pose_Rune_cb_rune_9点_TensorRT.engine` | 1 | WUST-RM/awakening | `778839e8000ae215015e21f9f2855820` |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_576x768.onnx` | 1 | WUST-RM/awakening | `535d4a84a4618e5ca01b522a1aa6577d` |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_576x768_TensorRT.engine` | 1 | WUST-RM/awakening | `9d355ff82836fb72c9ba750b3b73f3b6` |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_640.onnx` | 1 | WUST-RM/awakening | `714b8f36bc4438622bf0d17d31e0513b` |
| `武汉科技大学_YOLO26_Pose_Armor_praysky_C2PSA_640_TensorRT.engine` | 1 | WUST-RM/awakening | `5922ea7d9887d2227a09c689a091f355` |
| `武汉科技大学_YOLOX_Pose_Armor_opt0527_416.onnx` | 1 | WUST-RM/awakening | `4a58d9211a6914b8cad41bbd9c5f4a53` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416.onnx` | 2 | Blackjack200/talos_26、WUST-RM/awakening | `a2ef71420c6ca996150c32e3ae0cf423` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_TensorRT.engine` | 1 | WUST-RM/awakening | `2797f1ddaa6e9a45ab0c9cbee99b4051` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn.bin` | 1 | WUST-RM/awakening | `5708663a1ef7e475f7beb1e92ce20cb0` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn.param` | 1 | WUST-RM/awakening | `6eb5e7955c478f4abc59fa2291a470cd` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn_fp16.bin` | 1 | WUST-RM/awakening | `1cff71099c4c386af587150181de3e5b` |
| `武汉科技大学_YOLOX_Pose_Armor_opt1208_416_ncnn_fp16.param` | 1 | WUST-RM/awakening | `c8f7442960d29882a2701f4c43ed126f` |
| `武汉科技大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280.onnx` | 1 | WUST-RM/awakening | `538b19191e7dac19c409441a6190454a` |
| `武汉科技大学_YOLOv8_Det_Armor_雷达_ghost_p2_1280_TensorRT.engine` | 1 | WUST-RM/awakening | `06459e4ec49a7ef02bafa1a2d6c0780b` |
| `武汉科技大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280.onnx` | 1 | WUST-RM/awakening | `e5eab7c35a286383569f55234729ad94` |
| `武汉科技大学_YOLOv8_Det_Car_雷达车辆_ghost_p2_1280_TensorRT.engine` | 1 | WUST-RM/awakening | `e45f27aedf8eb7fa8612da5cec87f4a7` |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类.onnx` | 1 | JNU-SHARK/shark-radar-vision | `2761250f7615ac9c58cb0d6e16dee434` |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类.pt` | 1 | JNU-SHARK/shark-radar-vision | `ea570156d3e0f610a60822b4a7d0b82f` |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类_TensorRT.engine` | 1 | JNU-SHARK/shark-radar-vision | `ea6d2712a35a9d17d82d101e1025ed89` |
| `江南大学_YOLOv5_Det_Armor_雷达_1280_13类_batch_TensorRT.engine` | 1 | JNU-SHARK/shark-radar-vision | `22a921e015b3af9fdd09694c363846bc` |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类.onnx` | 1 | JNU-SHARK/shark-radar-vision | `4d4b002d1325cfc58511b3d3d3df28a3` |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类.pt` | 1 | JNU-SHARK/shark-radar-vision | `f0479e2aac05a72d5a67ea4dbc92f80d` |
| `江南大学_YOLOv5_Det_Car_雷达_640_5类_TensorRT.engine` | 1 | JNU-SHARK/shark-radar-vision | `cb6f4caad5bdddb32ab886c3f3827046` |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_480x640.onnx` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `d63490302c494961099c56ae33c58f66` |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_IR.bin` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `cfac31c06230db5a5573a263498acd5d` |
| `深圳大学_YOLOv8_Pose_Armor_Infantry_v8n_7类_IR.xml` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `0e715fdd3e33b8d909f9e6b1eadf72fb` |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_480x640.onnx` | 2 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models、SZURPVision/RuneDetectionModel | `48b8fbad24ae101817e23ebb4db9514e` |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_IR.bin` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `64f32f3781db934d1f9357031fd4a732` |
| `深圳大学_YOLOv8_Pose_Rune_Rune_v8n_5点_IR.xml` | 1 | SZURPVision/26_NNDeployment_Lib_and_Detection_Models | `75fb8370a0fd6ff4b0cf0eb139998216` |
| `深圳职业技术大学_YOLO_Pose_Rune_能量单元6dof.pt` | 1 | BenmaoNeko/RCIA_Benmao_Vision | `fad91a9f99604176cef2d2286d0869c0` |
| `西北工业大学_YOLO_Det_AntiDrone_反无人机.pt` | 1 | zplszz/WMJRadar | `9359799dc66f0b85d66586249886ad6e` |

## 判定依据

| 属性 | 依据 |
|---|---|
| 网络版本 | ONNX 内嵌 `description`（如 `YOLOv8n-pose` / `YOLO26n-pose`）；IR 输出节点命名（`/model.N/` 为 YOLOv8 系，`/m/model.N/` 为深大 RP 系） |
| Pose | Ultralytics 元数据 `task=pose` + `kpt_shape`；深大 RP 系据本仓库文档「四点检测模型」；输出通道含 4×2 关键点分量 |
| Armor/Rune | `names` 类别名（如 `B1/B3/BS/R1/R3/RS` 为装甲；`b`/`r_target`/`buff` 为打符） |
| 年份 | 模型版本赛季（`0708`→2024、`0526`→2025）或元数据 `date`；无标记时取发布仓库赛季 |
| 学校 | 权重原始战队；同名框架衍生仓库（GKD-RM-Lab / SHM-white 等）的复制件已并入原始战队 |

## 存疑项（建议人工复核）

- `同济大学_YOLO11_Pose_Armor_yolo11_int8`：IR 含 `anchor_points`，Pose 属性由「同济使用四点模型」推定。
- `北京科技大学_YOLOv8_Pose_Armor_MobileNetV3_last`：同济以 `yolov8.xml` 命名，与北科 `mobilenetv3_last` 同源同字节；Pose 属性按 RM 惯例推定。
- `武汉科技大学_YOLOX_Pose_Armor_opt1208/opt0527`：无元数据，按锚点数（3549=52²+26²+13²）判为 YOLOX 系、按通道数推为四点。
- `talos战队`：`Blackjack200/talos_26` 未标明学校，暂以战队名标注。
- `江南大学_YOLOv5_*`：ONNX 无 `task` 元数据，按输出锚点数（1280 输入 100800=3×33600）判为 YOLOv5 三尺度；类别名 `B1..R5` 判为装甲。
- `深圳职业技术大学_YOLO_Pose_Rune_能量单元6dof.pt`：无元数据，据帖子标题「6dof 位姿检测」推为 Pose。
- `东南大学_*_Compress_*`：为低码率图传的神经网络压缩权重（QVRF/MSSSIM/Real-ESRGAN），非识别任务，仅作留存。
- `华北理工大学_YOLOv8_Det_*_雷达_*`：与 `武汉科技大学_YOLOv8_Det_*_雷达_*` 同源（武科大转载），此处为原始 `.pt`。