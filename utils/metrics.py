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

def depth_metrics(depth_map, depth_map_gt):
    '''
    description: get the depth metrics of the got depth map and the ground truth
    parameter: depth map of mine and the ground truth
    return: several metrics, rms, rel, log10, 1.25, 1.25^2, 1.25^3
    '''
    batch_size = depth_map_gt.size(0)
    abs_diff = (depth_map - depth_map_gt).abs()

    diff_square = torch.pow(abs_diff, 2)
    diff_square = diff_square.view(batch_size, -1)
    diff_square_avg = torch.mean(diff_square, dim = 1)
    rmse = float(torch.mean(torch.sqrt(diff_square_avg)))

    aa = log10(depth_map)
    bb = log10(depth_map_gt)
    cc = (aa - bb).abs()
    rlog10 = float(cc.mean())
    rel = float((abs_diff / depth_map_gt).mean())

    max_ratio = torch.max(depth_map / depth_map_gt, depth_map_gt / depth_map)
    rate_1 = float(((max_ratio < 1.25) & (max_ratio >= 0)).float().mean())
    rate_2 = float(((max_ratio < (1.25 ** 2)) & (max_ratio >= 0)).float().mean())
    rate_3 = float(((max_ratio < (1.25 ** 3)) & (max_ratio >= 0)).float().mean())

    return rms, rel, rlog10, rate_1, rate_2, rate_3

def norm_metrics(norm, norm_gt, epsilon):
    '''
    description: get the norm metrics of the got norm and the ground truth norm
    parameter: norm mine and the ground truth, epsilon
    return: several metrics, mean, median, rmse, 11.25, 22.5, 30
    '''
    dot_product = torch.sum(norm * norm_gt, dim = 1)
    dot = torch.clamp(dot_product, min = -1.0, max = 1.0)
    errors = torch.acos(dot) / np.pi * 180

    mean = float(torch.mean(errors))
    median = float(torch.median(errors))

    batch_size = norm.size(0)
    error_square = torch.pow(errors, 2).view(batch_size, -1)
    total_size = batch_size * error_square.size(1)
    error_square_avg = torch.mean(error_square, 1)
    rmse = float(torch.mean(torch.sqrt(error_square_avg)))

    delta_1 = float((errors < 11.25).float().mean())
    delta_2 = float((errors < 22.5).float().mean())
    delta_3 = float((errors < 30).float().mean())

    return mean, median, rmse, delta_1, delta_2, delta_3