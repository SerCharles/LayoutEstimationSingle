import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


def ordinal_regression_loss(predict_result, gt, useful_mask, ord_num, beta, gamma, discretization):
    ''' 
    description: get the ordinal regression loss, used only in training
    parameter: the prediction calculated by the network, the ground truth label of pictures, the useful masks, other hyperparameters
    return: the regrssion loss
    '''
    total_num = useful_mask.sum().float()
    N, C, H, W = gt.size()
    predict_result = predict_result.view(N, 2, ord_num, H, W)
    prob = F.log_softmax(predict_result, dim = 1)
    prob = prob.view(N, 2 * ord_num, H, W)

    gt = gt + gamma
    gt = gt * useful_mask + (~useful_mask)

    ord_c0 = torch.ones(N, ord_num, H, W).to(gt.device)
    if discretization == "SID":
        label = ord_num * torch.log(gt) / np.log(beta + gamma)
    else:
        label = ord_num * (gt - 1.0) / (beta + gamma - 1.0)
    label = label * useful_mask
    label = label.long()

    mask = torch.linspace(0, ord_num - 1, ord_num, requires_grad = False) \
        .view(1, ord_num, 1, 1).to(gt.device)
    mask = mask.repeat(N, 1, H, W).contiguous().long()
    mask = (mask >= label)
    ord_c0[mask] = 0
    ord_c1 = 1 - ord_c0
    
    ord_label = torch.cat((ord_c0, ord_c1), dim = 1)

    entropy = -prob * ord_label
    loss = torch.sum(entropy, dim = 1, keepdim = True)[useful_mask].mean()
    
    return loss



def get_norm_loss(norm, norm_gt, mask):
    ''' 
    description: get the normal loss 
    parameter: the norm got by us, the norm of the ground truth(both normalized), the mask
    return: normal loss
    '''
    batch_size = norm_gt.size(0)
    pixels = torch.sum(mask).float()
    norm_plain = (norm * mask).view(batch_size, -1)
    norm_gt_plain = (norm_gt * mask).view(batch_size, -1)
    loss = torch.sum((norm_plain - norm_gt_plain) ** 2)
    loss_per_pixel = loss / (pixels + torch.eq(pixels, 0))
    return loss_per_pixel


def get_segmentation_loss(output, init_label, epsilon, mask):
    ''' 
    description: get the segmentation accuracy and cross entropy loss
    parameter: the output, the ground truth segmentation, mask
    return: accuracy, loss, the prediction
    '''
    N, C, H, W = init_label.size()
    batch_size = init_label.size(0)
    total_num = torch.sum(mask)

    output = output * mask
    softmaxed_output = F.softmax(output, dim = 1) 

    mask_true = torch.ne(init_label, 0)
    mask_false = ~mask_true 

    mask_true = mask_true * mask 
    mask_false = mask_false * mask

    one_hot_gt = torch.cat((mask_true, mask_false), dim = 1).float()
    cross_entropy_loss = -torch.sum(one_hot_gt * torch.log(softmaxed_output + epsilon)) / total_num
    probability_true = softmaxed_output[:, 0:1, :, :]
    predict_true = probability_true > 0.5

    accuracy = float(((predict_true == mask_true) * mask).float().sum() / total_num)

    return accuracy, cross_entropy_loss, predict_true
