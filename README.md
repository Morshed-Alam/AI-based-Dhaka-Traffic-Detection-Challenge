# AI-based-Dhaka-Traffic-Detection-Challenge
## Links
- [Preprocessing notebook](https://colab.research.google.com/drive/1OKGjFEYO4CPL2kzQcZdESdDWiczDpYA8?usp=sharing)
- [Training notebook](https://colab.research.google.com/drive/1u12CToHKw4iTxR2JysvFdYWO-FFgKtCg?usp=sharing)
- [Inference notebook](https://colab.research.google.com/drive/1e1dJOkvzOXpuwXKqS9aotpfP0moDForY?usp=sharing)
- [Dataset1](https://drive.google.com/file/d/1RNL2AT0UIrmQl7j0Ul01wVO1tKbBO1lG/view?usp=sharing)
- [Dataset2](https://drive.google.com/file/d/18PXVNsaUK4AveaawKFAMQ_HeQRfnOCxT/view?usp=sharing)
- [Weights trained on dataset1](https://drive.google.com/file/d/1KigPzWHrQrb8YsB7QxATeVpY5yFnGs8i/view?usp=sharing)
- [Weights trained on dataset2](https://drive.google.com/file/d/1IHxNroBc6uPJCdk8ZQ0gh4z9IUNWcZ_d/view?usp=sharing)
## Environment
- Google Colab Notebooks with free GPU.
## Preprocessing
### Dataset1
#### Sources of data:
   - 3002 images from dataset released by organizer
   - 272 images from [Dhaka-Traffic repository](https://github.com/Morshed-Alam/Dhaka-Traffic.git)
   - 6382 images generated using augmentation
   - 499 images from first round (Annotated manually)
#### Process:
- Download 3002 images released by organizer and 272 images from  [Dhaka-Traffic repository](https://github.com/Morshed-Alam/Dhaka-Traffic.git)
- Generate 6382 images using augmentation (Horizontal flip, Translation, Rotation, Shear, RandomHSV, Scale etc.) to images containing lower frequency classes (first 11 classes).
- Split data into train and valid set. Train set = 8656 & valid set = 1000.
- Add test1 annotated data to train set.
- Convert train and valid set to YOLOv5 data format.
- Resize train and valid set to 1024x1024.
- Finally add valid set data to train set to increase train data.

### Dataset2
#### Sources of data:
   - 3002 images from dataset released by organizer
   - 272 images from [Dhaka-Traffic repository](https://github.com/Morshed-Alam/Dhaka-Traffic.git)
   - 5001 images generated using augmentation
   - 499 images from first round (Annotated manually)
   - 537 images generated using dark and blur augmentation on [Roboflow](https://roboflow.com/). 
#### Process:
- Download 3002 images released by organizer and 272 images from  [Dhaka-Traffic repository](https://github.com/Morshed-Alam/Dhaka-Traffic.git)
- Add test1 annotated data to train set.
- Generate 5001 images using augmentation (Horizontal flip, Translation, Rotation, Shear, RandomHSV, Scale etc.) to images containing lower frequency classes (first 9 classes).
- Split data into train and valid set. Train set = 8774 & valid set = 1500.
- Convert train and valid set to YOLOv5 data format.
- Resize train and valid set to 1024x1024.
- Add 537 images generated using dark and blur augmentation on [Roboflow](https://roboflow.com/) from organizer released dataset selecting randomly.
- Finally add valid set data to train set to increase train data.
## Training
### Training setup

    ``` 
    img-size 1024
    batch-size 4
    epochs 40
    weights yolov5x.pt
       
### Model configuration & architecture
    ```
    # parameters

    nc: 21 # Number of classes
    depth_multiple: 1.33  # model depth multiple
    width_multiple: 1.25  # layer channel multiple

    # Anchors
    anchors:

    - [10,13, 16,30, 33,23] # P3/8
    - [30,61, 62,45, 59,119] # P4/16
    - [116,90, 156,198, 373,326] # P5/32
  
    # YOLOv5 backbone
    backbone:
    
    # [from, number, module, args]
    [[-1, 1, Focus, [64, 3]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, BottleneckCSP, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 9, BottleneckCSP, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, BottleneckCSP, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 1, SPP, [1024, [5, 9, 13]]],
    [-1, 3, BottleneckCSP, [1024, False]], # 9

    # YOLOv5 head
    head:
    
    [[-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, BottleneckCSP, [512, False]], # 13
    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, 'nearest']],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, BottleneckCSP, [256, False]], # 17 (P3/8-small)
    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, BottleneckCSP, [512, False]], # 20 (P4/16-medium)
    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, BottleneckCSP, [1024, False]], # 23 (P5/32-large)
    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)

### Hyper-parameters used in training
This hyper-parameters are fine tuned on PASCAL VOC dataset.


     ```
     lr0: 0.01
     lrf: 0.2
     momentum: 0.937
     weight_decay: 0.0005
     warmup_epochs: 3.0
     warmup_momentum: 0.8
     warmup_bias_lr: 0.1
     box: 0.05
     cls: 0.5
     cls_pw: 1.0
     obj: 1.0
     obj_pw: 1.0
     iou_t: 0.20
     anchor_t: 4.0
     #anchors: 3.63
     fl_gamma: 0.0
     hsv_h: 0.0138
     hsv_s: 0.664
     hsv_v: 0.464
     degrees: 0.373
     translate: 0.245
     scale: 0.898
     shear: 0.602
     perspective: 0.0
     flipud: 0.00856
     fliplr: 0.5
     mosaic: 1.0
     mixup: 0.243
     
## Inference
### Setup
- Weights from training on dataset1 and dataset2 (ensembling)
- TTA (Test time augmentation)
- Confidence threshold 0.5
- IoU threshold 0.5
