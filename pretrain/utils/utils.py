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
import torch.nn.functional as F


def get_seg(output):
    ''' 
    description: get the segmentation
    parameter: the output
    return: the predicted seg
    '''
    N, C, H, W = output.size()
    total_num = N * H * W
    softmaxed_output = F.softmax(output, dim = 1) 
    probability_true = softmaxed_output[:, 0:1, :, :]
    predict_true = probability_true > 0.5

    return predict_true

def normalize(norm, epsilon):
    ''' 
    description: normalize the normal vector 
    parameter: normal vector
    return: normalized normal vector
    '''
    nx = norm[:, 0 : 1, :, :]
    ny = norm[:, 1 : 2, :, :]
    nz = norm[:, 2 : 3, :, :]
    length = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    norm = norm / (length + epsilon)
    return norm


def get_predicted_depth(predicted_result, beta, gamma, discretization):
    ''' 
    description: get depth based on the predicted result, used in validation
    parameter: predicted result, other hyperparameters
    return: predicted depth
    '''
    N, C, H, W = predicted_result.size()
    ord_num = C // 2
    predicted_result = predicted_result.view(N, 2, ord_num, H, W)
    ord_prob = F.softmax(predicted_result, dim = 1)[:, 0, :, :, :]
    ord_label = torch.sum((ord_prob > 0.5), dim = 1)

    if discretization == "SID":
        t0 = torch.exp(np.log(beta + gamma) * ord_label.float() / ord_num)
        t1 = torch.exp(np.log(beta + gamma) * (ord_label.float() + 1) / ord_num)
    else:
        t0 = 1.0 + (beta + gamma - 1.0) * ord_label.float() / ord_num
        t1 = 1.0 + (beta + gamma - 1.0) * (ord_label.float() + 1) / ord_num
    depth = (t0 + t1) / 2 - gamma
    depth = depth.view(N, 1, H, W)
    return depth

