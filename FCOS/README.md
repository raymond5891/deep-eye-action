# FCOS Person Falling Detector

<img src="results/view.gif" alt="view" style="zoom:150%;" />

****
## Benchmarks

Resolution          |Mosaic|VOC-person mAP |Model Size |Epochs 
:-------------:|:--------:|:-------:|:--------------------:|:----------:
640*640      | - |   76.24%   | 107M      |36 
640*640      | √ |   **81.18% <font color='red'>(+4.94)</font>**   | 107M          |36 

Note:

* training data only using VOC-2012 person category (4087 training images)

* Test data using VOC-2007 test data (2007 testing images)

* in order to detect falling down person, rotation augmented was added to train VOC person

* Performance is measured on GeForce GTX 1060 

* you know... for the sake of poor GPU on my training server, only 4087 training images, if I got a better one (like 2080Ti), I will definitely add CrowdHuman

  

****
## Requirements

* Ubuntu 18.04

* CUDA >= 10.2

* Python >= 3.6

* Pytorch >= 1.6

  

## Pytorch Demo

First, install requirements following guide below. 

* Inference video 

```shell
#!/bin/bash

MODEL='./mosaic_training_dir/model_mosaic.pth'
VIDEO='./tests/v1.avi'
SAVE_PATH='./result.mp4'
VIDEO_WIDTH=672
VIDEO_HEIGHT=512

python detect_video.py \
	--model_path ${MODEL} \
	--test_video ${VIDEO} \
	--save_path ${SAVE_PATH} \
	--video_frame_width ${VIDEO_WIDTH} \
	--video_frame_height ${VIDEO_HEIGHT} \
```

+ model_path:  pytorch model to be tested
+ test_video: 
+ save_path: save path of result (empty by default, in which case the result will not be saved)
+ video_frame_width: width of frames in test video
+ video_frame_height: height of frames in test video

Note: 

+ if you want to save result, params **video_frame_width** and **video_frame_height** need to be checked before inference



## How to Train

### Prepare

Download Darknet-19 pretrained model *darknet19-deepBakSu-e1b3ec1e.pth*, and put it in ./weight

### Training

```shell
python train.py
```

Note: 

+ Remember to change your own training data directory in file "config/base.yaml"



## TODO

- [x] Mosaic Augment

- [ ] GIOU
- [ ] MultiScale Training
- [ ] Cosine Learing Rate Strategy
- [ ] IoU branch
