import math

import torch
import torch.nn.functional as F


def tile_features_2(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    # h_per_patch = h // num_pieces_per_line
    h_per_patch = h // 2
    w_per_patch = w // num_pieces_per_line
    # w_per_patch = w // 2 # if you want 2x1 puzzle
    
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    # features = torch.cat(patches, dim=0)
    return torch.cat(patches, dim=0)

def merge_features_2(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces)) 
    
    index = 0
    ext_h_list = []

    for _ in range(2):

        ext_w_list = []
        for _ in range(num_pieces_per_line): # 2개로 쪼갤땐 이걸루
            ext_w_list.append(features_list[index])
            index += 1
        # for _ in range(num_pieces_per_line): # 4, 16, 32...는 이걸루
        #     ext_w_list.append(features_list[index])
        #     index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features

def puzzle_module_2(x, func_list, num_pieces):
    tiled_x = tile_features_2(x, num_pieces)

    for func in func_list:
        tiled_x = func(tiled_x)
        
    merged_x = merge_features_2(tiled_x, num_pieces, x.size()[0])
    return merged_x
