# -*- coding: utf-8 -*-

import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
from puzzle_utils import *
from puzzle_utils_2 import *
from elr_loss import *
# from model_pretrained import *

from dataset import RafDataset
from model_pre import *
from utils import *
from resnet import *
from loss import ACLoss
from nrce_loss import NRCELoss

import wandb 

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm  # tqdm 라이브러리 import 추가
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/eac_raf_puzzle9')

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='/home/jihyun/data/RAF', help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='/home/jihyun/code/eac/eac_puzzle/model/resnet50_ft_weight.pkl', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--lam', type=float, default=10.0, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--lam_ELR', type=int, default=7, help='ELR lambda 1, 3, 5, 7, 10')
parser.add_argument('--beta_ELR', type=float, default=0.5, help='ELR beta 0.5, 0.7, 0.9, 0.99')

# parser.add_argument('--learning_rate ', default=1e-2, type=float)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architectures', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=2, type=int)

# for sanity check 
parser.add_argument('--log_freq', type=int, default=50, help='log print frequency')
parser.add_argument('--wandb', type=str, default='', help='wandb project name')
parser.add_argument('--save_path', type=str, default='/home/jihyun/code/eac/eac_puzzle/', help='weight save path')

args = parser.parse_args()

NB_CLASSES = 7

def train(epoch, args, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.to(device)
    model.train()
    
    # num_examp = train_loader.dataset.__len__()
    # print(num_examp) # 12271

    total_loss = []
    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device)
        indexes = indexes.to(device)
        # loss1 = loss1.to(device)

        tiled_images = tile_features(imgs2, args.num_pieces) # (4 * N, 3, H//2, h//2)
        
        # output, hm1 = Model(imgs1) # (N, 7), (N, 7, H//16, W//16) 
        output, hm1 = model(imgs1)
        tiled_output, tiled_hm = model(tiled_images, with_cam=True) # (4 * N, 7), (4 * N, 7, H//32, W//32)
        # tiled output 8 7  tiled_hm 8 7 14 14
        re_features = merge_features(tiled_hm, args.num_pieces, args.batch_size) # (N, 7, H//16, W//16) 
        
        grid_l = generate_flip_grid(args.w, args.h, device)
        
        # classification loss 
        # loss1 = nn.CrossEntropyLoss()(output, labels)

        # elr loss
        loss1 = elr_loss(indexes, output, labels, args.beta_ELR, args.lam_ELR, device)
        # loss1 = NRCELoss(NB_CLASSES)(output, labels)
        # consistency loss (between original feature map and puzzle cam)
        flip_loss_l = ACLoss(hm1, re_features, grid_l, output)

        loss = loss1 + args.lam * flip_loss_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss
        
        if batch_i % args.log_freq == 0 or batch_i == len(train_loader) - 1:
            print(f"Epoch : [{epoch}/{args.epochs}], Iter : [{batch_i}/{len(train_loader)}], Loss : {loss:.4f}")

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    
    return acc, running_loss

def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0

        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs1, with_cam=True)
            
            # mask = labels.unsqueeze(2).unsqueeze(3)
            # cams = (make_cam(features) * mask)

            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += outputs.size(0)
        
        # # for visualization
        # if batch_i == 0:
        #     obj_cams = cams.max(dim=1)[0]
            
        #     for b in range(8):
        #         image = get_numpy_from_tensor(imgs1[b])
        #         cam = get_numpy_from_tensor(obj_cams[b])
                
        #         image = denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])[..., ::-1]
        #         h, w, c = image.shape
                
        #         cam = (cam * 255).astype(np.uint8)
        #         cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        #         cam = colormap(cam)
                
        #         image = cv2.addWeighted(imgs1, 0.5, cam, 0.5, 0)[..., ::-1]
        #         image = image.astype(np.float32) / 255.
                
        #         writer.add_image('CAM/{}'.format(b+1), image, iter_cnt, dataformats='HWC')
                
        # for batch_index in range(imgs1.size()[0]):
        #     # c, h, w -> h, w, c
        #     cam = get_numpy_from_tensor(cams[batch_index]).transpose((1, 2, 0))
        #     gt_mask = get_numpy_from_tensor(mask[batch_index])
            
        #     h, w, c = cam.shape
        #     gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        
            

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
    return test_acc, running_loss

def eval_per_class():
    setup_seed(0)
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    # model = model(imgs1)
    model = Model(args)
    model.load_state_dict(torch.load(os.path.join(args.save_path, "results", args.wandb, "best.pth")))
    model.eval()
    
    #####
    total = 0 
    correct = 0
    class_correct = list(0 for _ in range(NB_CLASSES)) 
    class_total = list(0 for _ in range(NB_CLASSES)) 
    correct_sum = 0
    data_num = 0
    #####
    
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
    
    with torch.no_grad():
        model.eval()

        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(tqdm(test_loader)):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)

            outputs, _ = model(imgs1)
            _, predicts = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicts == labels).sum().item()
            
            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            data_num += outputs.size(0)
            
            c = (predicts == labels).squeeze()
            for j in range(imgs1.size(0)):
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] += 1
                
    # print(class_correct, sum(class_correct))
    # print(class_total, sum(class_total))
    
    avg_accuracy = 100 * correct_sum.float() / float(data_num)
    
    for k in range(0, NB_CLASSES):
        print(f"Accuracy of class {k}: {(100 * class_correct[k] / class_total[k]):.4f}")
    print(f"Average Test Accuracy: {avg_accuracy:.4f}")
    
    return class_correct, class_total, avg_accuracy

def main():    
    setup_seed(0)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),])
        # transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    

    train_dataset = RafDataset(args, phase='train', transform=train_transforms)
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    # model = Model(, num_classes=7, mode = args.mode)
    model = Model(args, pretrained=True, num_classes=7, mode=args.mode)
    
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    wandb.init(project="IEEE_FER")
    wandb.run.name = args.wandb
    wandb.config.update(args)
    wandb.watch(model)
    
    # make result saving directory 
    os.makedirs(os.path.join(args.save_path, "results", args.wandb), exist_ok=True)
    
    best_acc = 0
    best_epoch = 0
    
    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(i, args, model, train_loader, optimizer, scheduler, device)

        test_acc, test_loss = test(model, test_loader, device)
        
        if test_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save_path, "results", args.wandb, "best.pth"))
            best_acc = test_acc
            best_epoch = i
        else:
            print(f"Still best accuracy is {best_acc} at epoch {best_epoch}\n")
        
        print(f"Epoch : [{i}/{args.epochs}] \n Train accuracy : {train_acc:.4f} \n Train loss : {train_loss:.4f} \n Test accuracy : {test_acc:.4f} \n Test loss : {test_loss:.4f}\n")
        
        with open(os.path.join(args.save_path, "results", 'rebuttal_50_noise_' + str(args.label_path) + f'{args.wandb}.txt'), 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')
            
        wandb.log({"Train accuracy" : train_acc,
                   "Train loss" : train_loss,
                   "Test accuracy" : test_acc,
                   "Test loss" : test_loss,
                   })
        
        if i == args.epochs:
            class_correct, class_total, avg_accuracy = eval_per_class()
            
            output_dict = {}
            for k in range(NB_CLASSES):
                output_dict[f"Test accuracy of class {i}"] = 100 * class_correct[k] / class_total[k]
            output_dict["Average Test accurcy"] = avg_accuracy
            
            wandb.log(output_dict)

  
if __name__ == '__main__':
    main()
