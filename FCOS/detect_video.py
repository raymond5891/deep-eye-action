import cv2,os
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataset.VOC_dataset import VOCDataset
import time
from model.config import DefaultConfig
import argparse

parser = argparse.ArgumentParser(description='detect video')
parser.add_argument('--lr', default=None, help='learning rate', type=float)
parser.add_argument('--video_frame_width', help='frame width of tested video', type=int)
parser.add_argument('--video_frame_height', help='frame height of tested video', type=int)
parser.add_argument('--model_path', default="./mosaic_training_dir/model_mosaic.pth", help="path to your test model")
parser.add_argument('--test_video', default="./tests/v1.avi", help="path to your test video")
parser.add_argument('--save_path', type=str, default="", help="path to your save path")

args = parser.parse_args()

def preprocess_img(image,input_ksize):
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
    return image_paded

def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output

if __name__=="__main__":

    cap = cv2.VideoCapture(args.test_video)
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #writer = cv2.VideoWriter(args.save_path, fourcc, 30, (args.video_frame_height, args.video_frame_width))
        writer = cv2.VideoWriter(args.save_path, fourcc, 30, (args.video_frame_width, args.video_frame_height))
        
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    model=FCOSDetector(mode="inference",config=DefaultConfig).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    
    model=model.eval()
    print("===>success loading model")

    while True:
        ret, img_bgr = cap.read()
        if ret:
            img_pad = preprocess_img(img_bgr, [640,640])
            img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
            img1 = transforms.ToTensor()(img)
            img1 = transforms.Normalize([0.5,0.5,0.5], [1.,1.,1.], inplace=True)(img1)
            start_t=time.time()
            with torch.no_grad():
                out = model(img1.unsqueeze_(dim=0))
            end_t = time.time()
            cost_t = 1000*(end_t-start_t)
            print("===>success processing img, cost time %.2f ms"%cost_t)
            scores, classes, boxes = out

            boxes = boxes[0].cpu().numpy().tolist()
            classes = classes[0].cpu().numpy().tolist()
            scores = scores[0].cpu().numpy().tolist()
            
            for i, box in enumerate(boxes):
                if scores[i] < 0.5: continue
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv2.rectangle(img_pad, pt1, pt2, (0,255,0))
                cv2.putText(img_pad, "%s %.3f"%(VOCDataset.CLASSES_NAME[int(classes[i])],scores[i]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(VOCDataset.CLASSES_NAME[int(classes[i])], scores[i])
            
            cv2.imshow('img', img_pad)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if args.save_path:
                writer.write(img_pad)
        else: break

    if args.save_path:
        writer.release()
    cv2.destroyAllWindows()
