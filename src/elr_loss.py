import torch

import torch.nn.functional as F
from torch.autograd import Variable
# from torch import nn
# from main_ori import train_loader

def elr_loss(index, output, label, beta, lam, device):
    r"""Early Learning Regularization.
        Args
        * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
        * `output` Model's logits, same as PyTorch provided loss functions.
        * `label` Labels, same as PyTorch provided loss functions.
        """
    num_examp = 12271
    num_classes = 7
    # target = torch.device('cuda:1')
    target = torch.zeros(num_examp, num_classes).to(device)
    y_pred = F.softmax(output,dim=1)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
    y_pred_ = y_pred.data.detach()
    target[index] = beta * target[index] + (1-beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
    ce_loss = F.cross_entropy(output, label)
    elr_reg = ((1-(target[index] * y_pred).sum(dim=1)).log()).mean()
    final_loss = ce_loss +  lam *elr_reg
    return  final_loss