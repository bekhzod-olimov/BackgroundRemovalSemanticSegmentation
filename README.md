# Background Removal using Semantic Segmentation Project
- Changing traditional image processing technique to deep learning-based technique in the decoder part of original DISNET network and reduce information loss, which results in creating side outputs that are similar to GT and decrease loss value;
- 보다 나은 독해를 위해 용어 정리 해 두었다.
- Original convolution transpose 2D is substituted with nearest-neighbor upsampling and bilinear upsampling.

|Input Image&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ground Truth&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Proposed Method&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DISNET|
|------------|
![model_performance_comparison](https://github.com/user-attachments/assets/865b44c2-5efe-4183-a51e-d351111fd970)


# References
- DISNET: [xuebinqin/DIS](https://github.com/xuebinqin/DIS)
- U2NET: [xuebinqin/U-2-NET](https://github.com/xuebinqin/U-2-Net)
- EGNET: [JXingZhao/EGNet](https://github.com/JXingZhao/EGNet)
