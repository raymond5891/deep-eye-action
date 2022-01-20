import torch
import cv2
import math, random
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = colorJitter(img, boxes)
        #if random.random() < 0.5:
        #    img, boxes = random_rotation(img, boxes)
        if random.random() <= 0.6:
            img, boxes = random_rotation(img, boxes, degree=45)
        if random.random() < 0.5:
            img, boxes = random_crop_resize(img, boxes)
        return img, boxes

def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.ColorJitter(brightness=brightness,
                contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes



def random_rotation(img, boxes, degree=10):
    d = np.clip(np.random.randn() * degree, -degree* 2, degree* 2)
    #d = random.uniform(-degree, degree)
    w, h = img.size
    rx0, ry0 = w / 2.0, h / 2.0
    img = img.rotate(d)
    a = -d / 180.0 * math.pi
    boxes = torch.from_numpy(boxes)
    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 1]
    new_boxes[:, 1] = boxes[:, 0]
    new_boxes[:, 2] = boxes[:, 3]
    new_boxes[:, 3] = boxes[:, 2]
    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
        tp = torch.zeros_like(z)
        tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
        tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
        ymax, xmax = torch.max(tp, dim=0)[0]
        ymin, xmin = torch.min(tp, dim=0)[0]
        new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
    new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
    new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
    boxes[:, 0] = new_boxes[:, 1]
    boxes[:, 1] = new_boxes[:, 0]
    boxes[:, 2] = new_boxes[:, 3]
    boxes[:, 3] = new_boxes[:, 2]
    boxes = boxes.numpy()
    return img, boxes



def _box_inter(box1, box2):
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    return inter

#def random_perspective(im, targets=(), segments=(), degrees=0, translate=.1, scale=.1, shear=10, perspective=0.0,
def random_perspective(im, targets=(), segments=(), degrees=0, translate=0., scale=0., shear=0., perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            #xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 0:4].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 0:4] = new[i]

    return im, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, attempt_max=10):
    success = False
    boxes = torch.from_numpy(boxes)
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = random.uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(math.sqrt(target_area * aspect_ratio_)))
        h = int(round(math.sqrt(target_area / aspect_ratio_)))
        if random.random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            # check
            crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
            inter = _box_inter(crop_box, boxes) # [1,N] N can be zero
            box_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) # [N]
            mask = inter>0.0001 # [1,N] N can be zero
            inter = inter[mask] # [1,S] S can be zero
            box_area = box_area[mask.view(-1)] # [S]
            box_remain = inter.view(-1) / box_area # [S]
            if box_remain.shape[0] != 0:
                if bool(torch.min(box_remain > remain_min)):
                    success = True
                    break
            else:
                success = True
                break
    if success:
        img = img.crop((x, y, x+w, y+h))
        boxes -= torch.Tensor([x,y,x,y])
        boxes[:,1::2].clamp_(min=0, max=h-1)
        boxes[:,0::2].clamp_(min=0, max=w-1)
        # ow, oh = (size, size)
        # sw = float(ow) / img.size[0]
        # sh = float(oh) / img.size[1]
        # img = img.resize((ow,oh), Image.BILINEAR)
        # boxes *= torch.FloatTensor([sw,sh,sw,sh])
    boxes = boxes.numpy()
    return img, boxes
