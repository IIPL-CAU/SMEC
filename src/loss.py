import torch

import torch.nn.functional as F
from torch.autograd import Variable

def ACLoss(att_map1, att_map2, grid_l, output):
    flip_grid_large = grid_l.expand(output.size(0), -1, -1, -1)
    flip_grid_large = Variable(flip_grid_large, requires_grad = False)
    flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
    att_map2_flip = F.grid_sample(att_map2, flip_grid_large, mode = 'bilinear', padding_mode = 'border', align_corners=True)
    # att_map1_resized = torch.nn.functional.interpolate(att_map1, size=(8, 7, 33, 33), mode='trilinear', align_corners=False)/
    att_map1_resized = torch.nn.functional.interpolate(att_map1, size=(7, 7), mode='bilinear', align_corners=False)

    flip_loss_l = F.mse_loss(att_map1_resized, att_map2_flip)
    return flip_loss_l
    