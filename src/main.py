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
# from model_pretrained import *

from dataset import RafDataset
from model_pre import *
from utils import *
from resnet_mi import *
from loss import ACLoss
import wandb 

from tqdm import tqdm  # tqdm 라이브러리 import 추가
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/eac_raf_puzzle9')

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
parser.add_argument('--lam', type=float, default=5.5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--image_size', default=224, type=int)
# parser.add_argument('--learning_rate ', default=1e-2, type=float)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architectures', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)

# for sanity check 
parser.add_argument('--log_freq', type=int, default=50, help='log print frequency')
parser.add_argument('--wandb', type=str, default='', help='wandb project name')
parser.add_argument('--save_path', type=str, default='/home/jihyun/code/eac/eac_puzzle/', help='weight save path')

args = parser.parse_args()


def train(epoch, args, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    
    model.to(device)
    model.train()

    total_loss = []
    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        labels = labels.to(device)

        tiled_images = tile_features(imgs2, args.num_pieces) # (4 * N, 3, H//2, h//2)
        
        # output, hm1 = Model(imgs1) # (N, 7), (N, 7, H//16, W//16) 
        output, hm1 = model(imgs1)
        tiled_output, tiled_hm = model(tiled_images, with_cam=True) # (4 * N, 7), (4 * N, 7, H//32, W//32)

        re_features = merge_features(tiled_hm, args.num_pieces, args.batch_size) # (N, 7, H//16, W//16) 
        
        grid_l = generate_flip_grid(args.w, args.h, device)
        
        # classification loss 
        loss1 = nn.CrossEntropyLoss()(output, labels)
        
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
    # with torch.no_grad():
    model.eval()

    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    data_num = 0

    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
        imgs1 = imgs1.to(device)
        labels = labels.to(device)

        outputs, _ = model(imgs1, with_cam=True)

        loss = nn.CrossEntropyLoss()(outputs, labels)

        iter_cnt += 1
        _, predicts = torch.max(outputs, 1)

        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num

        running_loss += loss
        data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
    return test_acc, running_loss
        
def main():    
    setup_seed(0)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
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

    wandb.init(project="ICCE-Asia2023-FER")
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

    # for i in tqdm(range(1, args.epochs + 1), desc='Epoch'):  
    #     train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
    #     test_acc, test_loss = test(model, test_loader, device)
    #     with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
    #         f.write(str(i)+'_'+str(test_acc)+'\n')

    #     writer.add_scalar('training loss', train_loss, i)
    #     writer.add_scalar('training accuracy', train_acc, i)
    #     writer.add_scalar('test loss', test_loss, i)
    #     writer.add_scalar('test accuracy', test_acc, i)

    #     print('Epoch: {} | train_loss: {:.4f} | train_acc: {:.4f} | test_loss: {:.4f} | test_acc: {:.4f}'.format(i, train_loss, train_acc, test_loss, test_acc))
    #     # print('Epoch: {} | train_loss: {:.4f} | train_acc: {:.4f} | test_loss: {:.4f} | test_acc: {:.4f}'.format(i, train_loss, train_acc, test_loss, test_acc))  
if __name__ == '__main__':
    main()
