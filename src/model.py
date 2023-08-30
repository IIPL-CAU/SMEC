import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from abc_modules import ABC_Model
from resnet import *
from resnet_ori import *

# from torchsummary import summary as summary_

import torch
import pickle
# from torch.autograd import Variable

urls_dic = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ms_resnet50' : 'https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view?usp=sharing'
}

layers_dic = {
    'resnet18' : [2, 2, 2, 2],
    'resnet34' : [3, 4, 6, 3],
    'resnet50' : [3, 4, 6, 3],
    'resnet101' : [3, 4, 23, 3],
    'resnet152' : [3, 8, 36, 3],
    'ms_resnet50' : [3, 4, 6, 3],
}

#######################################################################
# Normalization
#######################################################################
# from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = ResNet(Bottleneck, layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            state_dict = model_zoo.load_url(urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            self.model.load_state_dict(state_dict)
            
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1, 
                                    self.model.bn1, 
                                    self.model.relu, 
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class Classifier(Backbone):
    def __init__(self, model_name, num_classes=7, mode='fix'):
        super().__init__(model_name, num_classes, mode)

        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False) # 2048 -> 7
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x) # [batch, num_classes, 7, 7]
            logits = self.global_average_pooling_2d(features)
            logits = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=- 1)
            return logits, features
        else:
            features = self.classifier(x) # [batch, num_classes, 7, 7]
            logits = self.global_average_pooling_2d(features)
            return logits, features