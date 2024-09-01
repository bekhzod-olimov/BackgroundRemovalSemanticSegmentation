# Background Removal using Semantic Segmentation Project
- Changing traditional image processing technique to deep learning-based technique in the decoder part of original DISNET network and reduce information loss, which results in creating side outputs that are similar to GT and decrease loss value;
- 보다 나은 독해를 위해 용어 정리 해 두었다.
- Original convolution transpose 2D is substituted with nearest-neighbor upsampling and bilinear upsampling.

|ORIGINAL_IMAGE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;GT&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DECONV_TO_D2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ORIGINAL_ISNET|
|------------|
|![models_performance_comparision](![model_performance_comparison](https://github.com/user-attachments/assets/4f3f8e58-5735-410f-8fca-5aa6148083e1))

# References
- DISNET: [xuebinqin/DIS](https://github.com/xuebinqin/DIS)
- U2NET: [xuebinqin/U-2-NET](https://github.com/xuebinqin/U-2-Net)
- EGNET: [JXingZhao/EGNet](https://github.com/JXingZhao/EGNet)
