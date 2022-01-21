# 直接执行此代码，定义网络的默认配置
class DefaultConfig():
    # backbone
    backbone="darknet19"
    mean = [0.5, 0.5, 0.5]
    std = [1., 1., 1.]

    #backbone="resnet50"
    #mean = [0.483, 0.452, 0.401]
    #std = [0.226, 0.221, 0.221]

    pretrained = True
    pretrained_backbone='/home/raymond/workspace/deep-eye/6th/detection/FCOS/pretrained/backbone/darknet19.pth'

    # fpn
    fpn_out_channels = 256
    use_p5 = True

    # head
    class_num = 20
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = True

    # training
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # inference
    score_threshold = 0.5
    nms_iou_threshold = 0.5
    max_detection_boxes_num = 150
