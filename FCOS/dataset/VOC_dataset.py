import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import  Image
import random

from dataset.augment import random_perspective

def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes

class VOCDataset(torch.utils.data.Dataset):
    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __init__(self,root_dir,resize_size=[640,800],split='trainval',use_difficult=False,is_train=True, augment=None, mean=[0.5,0.5,0.5], std=[1.,1.,1.]):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.img_ids=[x.split(' ')[0] for x in self.img_ids if x.split(' ')[-1]=='1']

        self.name2id=dict(zip(VOCDataset.CLASSES_NAME,range(len(VOCDataset.CLASSES_NAME))))
        self.id2name = {v:k for k,v in self.name2id.items()}
        self.resize_size=resize_size
        self.mean=mean
        self.std=std
        self.train = is_train
        self.augment = augment
        print("INFO=====>voc dataset init finished  ! !")

        ### mosaic 
        self.mosaic=False
        if is_train:
            self.mosaic=True
            print(f'training with Mosaic Augment')
        self.mosaic_border = [-resize_size[0]// 2, -resize_size[0]// 2]
        self.indices=range(len(self.img_ids))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self,index):
        if self.mosaic:
            img, boxes, classes = self.load_mosaic(index)
        else:
            img, boxes, classes = self.get_single_img(index)
            img, boxes=self.preprocess_img_boxes(img,boxes,self.resize_size)

        #self.visualize(img, boxes)
        img=transforms.ToTensor()(img)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)

        return img,boxes,classes

    def get_single_img(self,index):

        img_id=self.img_ids[index]
        img = Image.open(self._imgpath%img_id)

        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            name=obj.find("name").text.lower().strip()

            # make sure only person 
            if name != 'person': continue

            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            classes.append(self.name2id[name])

        if len(boxes):
            boxes=np.array(boxes,dtype=np.float32)
            if self.train:
                if random.random() < 0.5:
                    img, boxes = flip(img, boxes)
                if self.augment is not None:
                    img, boxes = self.augment(img, boxes)
            img = np.array(img)

        return img,boxes,classes

    def visualize(self, img, boxes):
        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img, pt1, pt2, (0,255,0))
        
        #img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def load_image(self, i):
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        im = self.imgs[i]
        if im is None:  # not cached in ram
            npy = self.img_npy[i]
            if npy and npy.exists():  # load npy
                im = np.load(npy)
            else:  # read image
                path = self.img_files[i]
                im = cv2.imread(path)  # BGR
                assert im is not None, f'Image Not Found {path}'
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                                interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        else:
            return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized

    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4 = []
        s = self.resize_size[0]
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            #img, _, (h, w) = load_image(self, index)
            img, labels, classes = self.get_single_img(index)
            img, labels = self.preprocess_img_boxes(img,labels,(self.resize_size[0], self.resize_size[0]))
            h, w, _ = img.shape
    
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
    
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
    
            # Labels
            labels = np.array(labels)

            '''
            if labels.size:
                y = np.copy(labels)

                y[:, 0] = labels[:, 0] + padw  # top left x
                y[:, 1] = labels[:, 1] + padh  # top left y
                y[:, 2] = labels[:, 2] + padw  # bottom right x
                y[:, 3] = labels[:, 3] + padh  # bottom right y

            labels4.append(y)
            '''

            for l in labels:
                y = [0] * 4
                if l[0] == l[2] or l[1] == l[3]:
                    continue
                y[0] = max(l[0] + padw, 0) # top left x
                y[1] = max(l[1] + padh, 0) # top left y
                y[2] = l[2] + padw  # bottom right x
                y[3] = l[3] + padh  # bottom right y
                
                labels4.append(np.array(y))
    
        # Concat/clip labels
        #labels4 = np.concatenate(labels4, 0)
        #for x in (labels4[:, :]):
        #    np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        
        labels4=np.array(labels4,dtype=np.float32)

        img4, labels4 = random_perspective(img4, labels4, border=self.mosaic_border)
        classes4 = np.array([15] * len(labels4))

        return img4, labels4, classes4

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes
        Returns
        image_paded: input_ksize
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape

        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w=32-nw%32
        pad_h=32-nh%32

        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def collate_fn(self,data):
        imgs_list,boxes_list,classes_list=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std,inplace=True)(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)))


        max_num=0
        for i in range(batch_size):
            n=boxes_list[i].shape[0]
            if n>max_num:max_num=n
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))


        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_imgs=torch.stack(pad_imgs_list)

        return batch_imgs,batch_boxes,batch_classes


if __name__=="__main__":
    pass
    eval_dataset = VOCDataset(root_dir='/Users/VOCdevkit/VOCdevkit/VOC0712', resize_size=[800, 1333],
                               split='test', use_difficult=False, is_train=False, augment=None)
    print(len(eval_dataset.CLASSES_NAME))
    #dataset=VOCDataset("/home/data/voc2007_2012/VOCdevkit/VOC2012",split='trainval')
    # for i in range(100):
    #     img,boxes,classes=dataset[i]
    #     img,boxes,classes=img.numpy().astype(np.uint8),boxes.numpy(),classes.numpy()
    #     img=np.transpose(img,(1,2,0))
    #     print(img.shape)
    #     print(boxes)
    #     print(classes)
    #     for box in boxes:
    #         pt1=(int(box[0]),int(box[1]))
    #         pt2=(int(box[2]),int(box[3]))
    #         img=cv2.rectangle(img,pt1,pt2,[0,255,0],3)
    #     cv2.imshow("test",img)
    #     if cv2.waitKey(0)==27:
    #         break
    #imgs,boxes,classes=eval_dataset.collate_fn([dataset[105],dataset[101],dataset[200]])
    # print(boxes,classes,"\n",imgs.shape,boxes.shape,classes.shape,boxes.dtype,classes.dtype,imgs.dtype)
    # for index,i in enumerate(imgs):
    #     i=i.numpy().astype(np.uint8)
    #     i=np.transpose(i,(1,2,0))
    #     i=cv2.cvtColor(i,cv2.COLOR_RGB2BGR)
    #     print(i.shape,type(i))
    #     cv2.imwrite(str(index)+".jpg",i)

