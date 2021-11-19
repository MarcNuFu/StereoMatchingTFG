# StereoMatchingTFG

Dataset - KITTI2015
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip
```
Autoencoder info
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 128, 256]             448
       BatchNorm2d-2         [-1, 16, 128, 256]              32
              ReLU-3         [-1, 16, 128, 256]               0
            Conv2d-4          [-1, 32, 64, 128]           4,640
       BatchNorm2d-5          [-1, 32, 64, 128]              64
              ReLU-6          [-1, 32, 64, 128]               0
            Conv2d-7           [-1, 64, 32, 64]          18,496
       BatchNorm2d-8           [-1, 64, 32, 64]             128
              ReLU-9           [-1, 64, 32, 64]               0
           Conv2d-10          [-1, 128, 16, 32]          73,856
      BatchNorm2d-11          [-1, 128, 16, 32]             256
             ReLU-12          [-1, 128, 16, 32]               0
           Conv2d-13           [-1, 256, 8, 16]         295,168
      BatchNorm2d-14           [-1, 256, 8, 16]             512
  ConvTranspose2d-15          [-1, 128, 16, 32]         295,040
      BatchNorm2d-16          [-1, 128, 16, 32]             256
             ReLU-17          [-1, 128, 16, 32]               0
  ConvTranspose2d-18           [-1, 64, 32, 64]          73,792
      BatchNorm2d-19           [-1, 64, 32, 64]             128
             ReLU-20           [-1, 64, 32, 64]               0
  ConvTranspose2d-21          [-1, 32, 64, 128]          18,464
      BatchNorm2d-22          [-1, 32, 64, 128]              64
             ReLU-23          [-1, 32, 64, 128]               0
  ConvTranspose2d-24         [-1, 16, 128, 256]           4,624
      BatchNorm2d-25         [-1, 16, 128, 256]              32
             ReLU-26         [-1, 16, 128, 256]               0
  ConvTranspose2d-27          [-1, 3, 256, 512]             435
      BatchNorm2d-28          [-1, 3, 256, 512]               6
          Sigmoid-29          [-1, 3, 256, 512]               0
================================================================
Total params: 786,441
Trainable params: 786,441
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.50
Forward/backward pass size (MB): 54.50
Params size (MB): 3.00
Estimated Total Size (MB): 59.00
----------------------------------------------------------------
```
