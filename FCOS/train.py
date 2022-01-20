import yaml
from model.fcos import FCOSDetector
import torch
from dataset.VOC_dataset import VOCDataset
import math, time
from dataset.augment import Transforms
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse
from model.config import DefaultConfig

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./config/base.yaml', help='specify config file')
    args = parser.parse_args()
    print(args)
    with open(args.config_file, 'r') as stream:
        opts = yaml.safe_load(stream)
    return opts

def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['n_gpu']
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    transform = Transforms()
    config = DefaultConfig
    train_dataset = VOCDataset(root_dir=opt['data_root_dir'], resize_size=[640,640], split='person_trainval',use_difficult=False,is_train=True,augment=transform, mean=config.mean,std=config.std)

    model = FCOSDetector(mode="training").cuda()
    model = torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load(config.pre_model, map_location=torch.device('cuda')))
    output_dir = 'training_dir'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    BATCH_SIZE = opt['batch_size']
    EPOCHS = opt['epochs']
    #WARMPUP_STEPS_RATIO = 0.12
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,
                                            num_workers=opt['n_cpu'], worker_init_fn=np.random.seed(0))
    print("total_images : {}".format(len(train_dataset)))
    steps_per_epoch = len(train_dataset) // BATCH_SIZE
    TOTAL_STEPS = steps_per_epoch * EPOCHS
    WARMPUP_STEPS = 501

    GLOBAL_STEPS = 1
    LR_INIT = 1e-3
    optimizer = torch.optim.SGD(model.parameters(),lr=LR_INIT,momentum=0.9,weight_decay=1e-4)

    model.train()

    print(model)

    for epoch in range(EPOCHS):
        for epoch_step, data in enumerate(train_loader):

            batch_imgs, batch_boxes, batch_classes = data
            if batch_classes.shape[1] == 0: continue
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            #lr = lr_func()
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT)
            for param in optimizer.param_groups:
                param['lr'] = lr
            if GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
                lr = LR_INIT * 0.1
            for param in optimizer.param_groups:
                param['lr'] = lr
            if GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
                lr = LR_INIT * 0.01
            for param in optimizer.param_groups:
                param['lr'] = lr
            start_time = time.time()

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()

            optimizer.step()

            end_time = time.time()
            cost_time = int((end_time - start_time) * 1000)
            if GLOBAL_STEPS%50 == 0:
                print(
                    "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                    (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, losses[0].mean(), losses[1].mean(),
                    losses[2].mean(), cost_time, lr, loss.mean()))
            GLOBAL_STEPS += 1
        torch.save(model.state_dict(),
            os.path.join(output_dir, "model_{}.pth".format(epoch + 1)))
        
if __name__ == '__main__':
    opt = parse_config()
    main(opt)

