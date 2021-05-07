''' 
Used in getting the depth metrics and the segmentation accuracy
'''

import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
from torchvision import transforms
from scipy.optimize import linear_sum_assignment

def log10(x):
    return torch.log(x) / log(10)

def depth_metrics(depth_map, depth_map_gt, mask):
    '''
    description: get the depth metrics of the got depth map and the ground truth
    parameter: depth map of mine and the ground truth, mask
    return: several metrics, rms, rel, log10, 1.25, 1.25^2, 1.25^3
    '''

    batch_size = depth_map_gt.size(0)
    number_by_batch = mask.view(batch_size, -1).sum(dim = 1).float()

    

    total_num = mask.float().sum()
    abs_diff = (depth_map - depth_map_gt).abs() * mask

    diff_square = torch.pow(abs_diff, 2)
    diff_square = diff_square.view(batch_size, -1)
    diff_square_avg = torch.sum(diff_square, dim = 1) / number_by_batch
    rms = float(torch.mean(torch.sqrt(diff_square_avg)))

    aa = log10(depth_map) * mask
    bb = log10(depth_map_gt) * mask
    cc = (aa - bb).abs()
    cc_avg = cc.view(batch_size, -1).sum(dim = 1) / number_by_batch
    rlog10 = float(torch.mean(cc_avg))

    rel_avg = torch.sum((abs_diff / depth_map_gt).view(batch_size, -1), dim = 1) / number_by_batch
    rel = float(torch.mean(rel_avg))

    max_ratio = torch.max(depth_map / depth_map_gt, depth_map_gt / depth_map)

    rate_1 = float(((max_ratio < 1.25) & mask).float().sum() / total_num)
    rate_2 = float(((max_ratio < (1.25 ** 2))  & mask).float().sum() / total_num)
    rate_3 = float(((max_ratio < (1.25 ** 3))  & mask).float().sum() / total_num)

    return rms, rel, rlog10, rate_1, rate_2, rate_3

def norm_metrics(norm, norm_gt, epsilon, mask):
    '''
    description: get the norm metrics of the got norm and the ground truth norm
    parameter: norm mine and the ground truth, epsilon, mask
    return: several metrics, mean, median, rmse, 11.25, 22.5, 30
    '''

    batch_size = norm_gt.size(0)
    number_by_batch = mask.view(batch_size, -1).sum(dim = 1).float()
    total_num = mask.float().sum()
    dot_product = torch.sum(norm * norm_gt, dim = 1, keepdim = True) #N * 1 * W * H
    dot = torch.clamp(dot_product, min = -1.0, max = 1.0)
    errors = torch.acos(dot) / np.pi * 180

    if int(torch.sum(mask)) == 0: 
        return inf, inf, inf, 0.0, 0.0, 0.0

    selected_error = torch.masked_select(errors, mask)

    mean = float(torch.mean(selected_error))
    median = float(torch.median(selected_error))

    error_square = torch.pow(errors, 2).view(batch_size, -1)
    total_size = batch_size * error_square.size(1)
    error_square_avg = torch.sum(error_square, dim = 1) / number_by_batch
    rmse = float(torch.mean(torch.sqrt(error_square_avg)))

    delta_1 = float(((errors < 11.25) & mask).float().sum() / total_num)
    delta_2 = float(((errors < 22.5) & mask).float().sum() / total_num)
    delta_3 = float(((errors < 30) & mask).float().sum() / total_num)

    return mean, median, rmse, delta_1, delta_2, delta_3


def seg_metrics(segs, segs_gt):
    ''' 
    description: get the segmentation accuracy
    parameter: our seg results, the gt seg
    return: mean accuracy of the batch
    '''
    batch_size = len(segs)
    total_pixel_num = batch_size * len(segs[0][0]) * len(segs[0][0][0])

    total_same_num = 0
    for batch in range(batch_size):
        my_seg = segs[batch][0]
        seg_gt = segs_gt[batch][0]
        my_id = np.unique(my_seg)
        id_gt = np.unique(seg_gt)

        cost_matrix = np.zeros((len(my_id), len(id_gt)))
        for i in range(len(my_id)):
            for j in range(len(id_gt)):
                ii = my_id[i]
                jj = id_gt[j]
                mask_my = np.equal(my_seg, ii)
                mask_gt = np.equal(seg_gt, jj)
                mask_same = mask_my & mask_gt
                same_num = np.sum(mask_same)
                cost_matrix[i][j] = same_num
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize = True)
        same_num = cost_matrix[row_ind, col_ind].sum()
        total_same_num += same_num
    accuracy = total_same_num / total_pixel_num
    return accuracy