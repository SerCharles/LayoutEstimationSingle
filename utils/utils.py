''' 
functions used in mathmatic calculation
'''


import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
import torch.nn as nn
from torchvision import transforms

def normalize(norm, epsilon):
    ''' 
    description: normalize the normal vector 
    parameter: normal vector
    return: normalized normal vector
    '''
    batch_size = norm.size(0)
    nx = norm[:][0 : 1][:][:]
    ny = norm[:][1 : 2][:][:]
    nz = norm[:][2 : 3][:][:]
    length = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    norm = norm / (length + epsilon)
    return norm


def ordinal_regression_loss(predict_result, gt, useful_mask, beta, discretization):
    ''' 
    description: get the ordinal regression loss, used only in training
    parameter: the prediction calculated by the network, the ground truth label of pictures, the useful masks, other hyperparameters
    return: the regrssion loss
    '''
    N, C, H, W = gt.shape
    ord_num = C // 2
    predict_result = predict_result.view(N, 2, ord_num, H, W)
    prob = F.log_softmax(x, dim = 1).view(N, C, H, W)

    ord_c0 = torch.ones(N, ord_num, H, W).to(gt.device)
    if discretization == "SID":
        label = ord_num * torch.log(gt) / np.log(beta)
    else:
        label = ord_num * (gt - 1.0) / (beta - 1.0)

    label = label.long()
    mask = torch.linspace(0, ord_num - 1, ord_num, requires_grad = False) \
        .view(1, ord_num, 1, 1).to(gt.device)
    mask = mask.repeat(N, 1, H, W).contiguous().long()
    mask = (mask > label)
    ord_c0[mask] = 0
    ord_c1 = 1 - ord_c0
    ord_label = torch.cat((ord_c0, ord_c1), dim = 1)

    entropy = -probability * ord_label * useful_mask
    loss = torch.sum(entropy, dim = 1).mean()
    return loss

def get_predicted_depth(predicted_result, gt, beta, gamma, discretization):
    ''' 
    description: get depth based on the predicted result, used in validation
    parameter: predicted result, other hyperparameters
    return: predicted depth
    '''
    N, C, H, W = x.size()
    ord_num = C // 2

    ord_prob = F.softmax(x, dim = 1)[:, 0, :, :, :]
    ord_label = torch.sum((ord_prob > 0.5), dim = 1)

    if discretization == "SID":
        t0 = torch.exp(np.log(beta) * ord_label.float() / ord_num)
        t1 = torch.exp(np.log(beta) * (ord_label.float() + 1) / ord_num)
    else:
        t0 = 1.0 + (beta - 1.0) * ord_label.float() / ord_num
        t1 = 1.0 + (beta - 1.0) * (ord_label.float() + 1) / ord_num
    depth = (t0 + t1) / 2 - gamma
    return depth