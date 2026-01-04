# JDFNet
This is the official version of paper **"Simultaneous Surgical Stereo Depth and Motion Estimation via Brightness-aware Self-supervised Learning"**

## Proposed Method
This paper proposes a novel method, namely JDFNet, for jointly fusing surgical stereo depth and flow maps via self-supervised learning to boost the overall performance by leveraging relationships between the two tasks. Based on deep convolutional neural networks, the proposed method is built upon three well-designed modules for cross-domain fusion, poseguided reconstruction, and brightness-aware correction.

## Datasets
Related datasets and preprocessing details can be found from the following three papers:
- [x] Stereo correspondence and reconstruction of endoscopic data challenge
- [x] Frsr: Framework for real-time scene reconstruction in robotassisted minimally invasive surgery
- [x] Self-supervised siamese learning on stereo image pairs for depth estimation in robotic surgery
- [x] Serv-ct: A disparity dataset from cone-beam ct for validation of endoscopic 3d reconstruction


## Implementation Details
- [x] Training network for stereo depth estimation
- [x] Training network for consecutive motion estimation
- [x] Multitask learning
- [ ] More details and validation methods will be available step by step in the future
- [ ] Pretrained Models 

## Citation
Please cite our work if you find this work useful for your research.
```latex
@article{Liujdfnet2026,
author = {Yuxuan Liu, Xinyao Zhou, Yating Luo, Yunfei Luan, Yao Guo and Guang-Zhong Yang},
title = {Simultaneous Surgical Stereo Depth and Motion Estimation via Brightness-aware Self-supervised Learning},
journal = {},
pages = {},
year = {2026},
 } 
  
```
