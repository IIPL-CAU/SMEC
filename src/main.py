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

from dataset import RafDataset
from model import *
from utils import *
from resnet import *
from loss import ACLoss

from tqdm import tqdm  # tqdm 라이브러리 import 추가
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/eac_raf_puzzle9')


parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='/home/jihyun/data/RAF/', help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='/home/jihyun/code/Erasing-Attention-Consistency_1/model/resnet50_ft_weight.pkl', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='/home/jihyun/data/RAF/EmoLabel/list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--image_size', default=513, type=int)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architectures', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=9, type=int)

args = parser.parse_args()


def train(args, model, train_loader, optimizer, scheduler, device):
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
        # indexes = indexes.to(device)

        # len(indexes)

        criterion = nn.CrossEntropyLoss(reduction='none')

        output, hm1 = model(imgs1, with_cam=True)


        # puzzle module
        tiled_images = tile_features(imgs2, args.num_pieces)
        
        tiled_output, tiled_hm = model(tiled_images, with_cam=True)

        re_features = merge_features(tiled_hm, args.num_pieces, args.batch_size)
        
        grid_l = generate_flip_grid(args.w, args.h, device)
        
        loss1 = nn.CrossEntropyLoss()(output, labels)
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
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    
    
    
    model = Classifier(args.architectures, num_classes=7, mode = args.mode)
    
    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    
    
    # for i in tqdm(range(1, args.epochs + 1)):
    #     train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
    #     # args, model, train_loader, optimizer, scheduler, device
    #     test_acc, test_loss = test(model, test_loader, device)
    #     with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
    #         f.write(str(i)+'_'+str(test_acc)+'\n')

    for i in tqdm(range(1, args.epochs + 1), desc='Epoch'):  
        train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
        test_acc, test_loss = test(model, test_loader, device)
        with open('rebuttal_50_noise_'+str(args.label_path)+'.txt', 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')

        writer.add_scalar('training loss', train_loss, i)
        writer.add_scalar('training accuracy', train_acc, i)
        writer.add_scalar('test loss', test_loss, i)
        writer.add_scalar('test accuracy', test_acc, i)

        print('Epoch: {} | train_loss: {:.4f} | train_acc: {:.4f} | test_loss: {:.4f} | test_acc: {:.4f}'.format(i, train_loss, train_acc, test_loss, test_acc))
        # print('Epoch: {} | train_loss: {:.4f} | train_acc: {:.4f} | test_loss: {:.4f} | test_acc: {:.4f}'.format(i, train_loss, train_acc, test_loss, test_acc))  
if __name__ == '__main__':
    main()
