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
    batch_size = norm.size(0)
    nx = norm[:][0 : 1][:][:]
    ny = norm[:][1 : 2][:][:]
    nz = norm[:][2 : 3][:][:]
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
        t0 = torch.exp(np.log(beta) * ord_label.float() / ord_num)
        t1 = torch.exp(np.log(beta) * (ord_label.float() + 1) / ord_num)
    else:
        t0 = 1.0 + (beta - 1.0) * ord_label.float() / ord_num
        t1 = 1.0 + (beta - 1.0) * (ord_label.float() + 1) / ord_num
    depth = (t0 + t1) / 2 - gamma
    depth = depth.view(N, 1, H, W)
    return depth




def get_plane_info_per_pixel(device, norm, depth, intrinsic):
    '''
    description: calculate the plane info based on the norm, depth and intrinsic we get
    parameter: device, the norm, depth, intrinsic
    return: plane info(A, B, C, D) per pixel
    '''
    N, C, H, W  = norm.size()
    xx, yy = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))
    yy = H - 1 - yy
    xx = torch.from_numpy(xx)
    yy = torch.from_numpy(yy)
    if device:
        xx = xx.cuda()
        yy = yy.cuda()
    
    xx = xx.view(1, 1, H, W).repeat(N, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(N, 1, 1, 1)
    fx = intrinsic[:, 0, 0].view(N, 1, 1, 1).repeat(1, 1, H, W)
    fy = intrinsic[:, 1, 1].view(N, 1, 1, 1).repeat(1, 1, H, W)
    x0 = intrinsic[:, 2, 0].view(N, 1, 1, 1).repeat(1, 1, H, W)
    y0 = intrinsic[:, 2, 1].view(N, 1, 1, 1).repeat(1, 1, H, W)
    x = ((xx - x0) / fx) * depth 
    y = ((yy - y0) / fy) * depth 

    A = norm[:, 0:1, :, :]
    B = norm[:, 1:2, :, :]
    C = norm[:, 2:3, :, :]
    D = - A * x - B * y - C * depth

    size = torch.sqrt(A ** 2 + B ** 2 + C ** 2 + D ** 2) 
    size_mask = torch.eq(size, 0)
    size_real = size + size_mask * 1e-8
    A = A / size_real
    B = B / size_real
    C = C / size_real
    D = D / size_real

    plane_info = torch.cat((A, B, C, D), dim = 1)
    return plane_info


def get_plane_max_num(plane_seg):
    '''
    description: get the plane ids
    parameter: plane seg map
    return: the max num of planes
    '''
    max_num = torch.max(plane_seg)
    max_num = max_num.detach()
    return max_num

def get_average_plane_info(device, plane_infos, plane_seg, max_num):
    '''
    description: get the average plane info 
    parameter: device, parameters per pixel, plane segmentation per pixel, the max segmentation num of planes
    return: average plane info
    '''
    batch_size = plane_seg.size(0)
    size_v = plane_seg.size(2)
    size_u = plane_seg.size(3)
    average_plane_infos = []
    
    for batch in range(batch_size):
        the_plane_info = plane_infos[batch]
        average_plane_infos.append([])
        for i in range(0, max_num + 1):
            the_mask = torch.eq(plane_seg[batch], i) #选择所有seg和i相等的像素
            the_mask = the_mask.detach()

            the_total = torch.sum(the_plane_info * the_mask, dim = [1, 2]) #对每个图符合条件的求和
            the_count = torch.sum(the_mask) #求和
            new_count = the_count + torch.eq(the_count, 0) #trick，如果count=0，mask=1，加上变成1(但是total=0，结果还是0)
            new_count = new_count.detach()


            
            average_plane_infos[batch].append((the_total / new_count).unsqueeze(0))
        average_plane_infos[batch] = torch.cat(average_plane_infos[batch], dim = 0).unsqueeze(0)
    average_plane_infos = torch.cat(average_plane_infos)
    return average_plane_infos

def set_average_plane_info(plane_seg, average_plane_info):
    '''
    description: set the per pixel plane info to the average
    parameter: the plane ids, plane_segs, the average plane infos, the shape of the depth map
    return: average per pixel plane info
    '''
    batch_size = len(plane_seg)
    size_v = len(plane_seg[0][0])
    size_u = len(plane_seg[0][0][0])
    new_plane_infos = []
    for i in range(batch_size):
        new_plane_infos.append([])
        for the_id in range(len(average_plane_info[i])):
            the_id = int(the_id) 
            mask = torch.eq(plane_seg[i][0], the_id)
            mask = mask.detach()
            a = average_plane_info[i][the_id][0]
            b = average_plane_info[i][the_id][1]
            c = average_plane_info[i][the_id][2]
            d = average_plane_info[i][the_id][3]
            masked_a = (mask * a).unsqueeze(0)
            masked_b = (mask * b).unsqueeze(0)
            masked_c = (mask * c).unsqueeze(0)
            masked_d = (mask * d).unsqueeze(0)
            the_plane_info = torch.cat([masked_a, masked_b, masked_c, masked_d]).unsqueeze(0)
            new_plane_infos[i].append(the_plane_info)

        new_plane_infos[i] = torch.cat(new_plane_infos[i])
        new_plane_infos[i] = torch.sum(new_plane_infos[i], dim = 0, keepdim = True)
    new_plane_infos = torch.cat(new_plane_infos)
    return new_plane_infos