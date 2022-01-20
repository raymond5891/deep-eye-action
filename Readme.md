

[TOC]

## 1. 任务简述

主要任务为跌倒检测，推理和训练阶段使用的网络不同：

+ 训练阶段需要自己制作图卷积模型的输入数据，所以需要精度高的人体检测器和人体关键点检测配合
+ 推理阶段不需要人体检测
+ 训练阶段人体关键点模型选用精度高的 Alphapose
+ 推理阶段人体关键点模型选择速度快的 Openpose



### 1.1 推理阶段

分为三个模块：

1. 人体关键点检测（使用 Openpose）
2. 跟踪模块 （使用 SORT)
3. 根据多帧（暂定连续30帧）检测出来的关键点，通过图卷积模型识别动作（图卷积模型使用 st-gcn）



### 1.2 训练阶段

训练阶段需要人体检测器，所以训练阶段由三个模块组成：

1. 人体检测：使用 FCOS
   + 仅识别人体一个类别，但是要求可以检测到从站立到跌倒不同角度的人体
2. 人体关键点检测：使用 Alphapose
3. 动作识别（与推理阶段相同使用st-gcn）



## 2. 数据

### 2.1 人体检测数据集：

+ 第一阶段使用开源数据集 VOC2007 和 VOC2012

+ 后期提高精度会增加 Crowd Human 数据集

  

### 2.2 人体关键点数据集：

+ 使用 COCO 2017 开源数据

  

### 2.3 跌倒检测数据集：

+ 使用开源数据集 Le2i Fall detection

+ 共191个视频，暂定按照5:1的比例分割训练和测试数据



## 3. 指标

### 3.1 精度指标

#### 3.1.1 人体检测：

由于人体检测器 FCOS 在训练阶段生成图卷积模型使用，所以其精度直接影响动作识别效果，所以暂定 mAP >= 85%



#### 3.1.2 人体关键点检测：

训练阶段 Alphapose 网络  AP@0.5:0.95 >= 70% （直接使用预训练模型）



#### 3.1.3 动作识别

最终测试数据为 Le2i Fall detection 的视频

+ 精度指标定义（预测正确的帧数除以视频有标签的总帧数）：


<img src="imgs/ap.png" alt="ap" style="zoom:50%;" />

### 3.2 实时性指标

+ 最终运行的硬件环境为 GPU （具体型号有待确定）

+ 帧率 >= 25 FPS 
