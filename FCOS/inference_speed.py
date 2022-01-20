import cv2,os
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
import matplotlib.patches as patches
import  matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from model.config import DefaultConfig

if __name__=="__main__":
    model=FCOSDetector(mode="inference",config=DefaultConfig).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./training_dir_fp16_nogn/model_24.pth",map_location=torch.device('cpu')))
    model=model.eval()
    print("===>success loading model")

    root="./test_images/"
    name=os.listdir(root)[0]

    img_bgr=cv2.imread(root+name)
    start_t = time.time()
    for i in range(500):
        if i%50 == 0:
            print('already tested %d images' % i)
        img_pad=preprocess_img(img_bgr,[640, 800])
        img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img1=transforms.ToTensor()(img)
        img1= transforms.Normalize([0.5,0.5,0.5], [1.,1.,1.],inplace=True)(img1)
        img1=img1
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
    end_t=time.time()
    cost_t=1000.*(end_t-start_t)
    print("===>success processing img, the average inference time for each image is %.2f ms"% (cost_t/500.))

