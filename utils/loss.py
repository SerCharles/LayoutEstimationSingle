import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


def ordinal_regression_loss(predict_result, gt, useful_mask, ord_num, beta, discretization):
    ''' 
    description: get the ordinal regression loss, used only in training
    parameter: the prediction calculated by the network, the ground truth label of pictures, the useful masks, other hyperparameters
    return: the regrssion loss
    '''
    N, C, H, W = gt.shape
    predict_result = predict_result.view(N, 2, ord_num, H, W)
    prob = F.log_softmax(predict_result, dim = 1).view(N, 2 * ord_num, H, W)

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

    entropy = -prob * ord_label * useful_mask
    loss = torch.sum(entropy, dim = 1).mean()
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


def get_segmentation_loss(output, init_label, epsilon):
    ''' 
    description: get the segmentation accuracy and cross entropy loss
    parameter: the output, the ground truth segmentation
    return: accuracy, loss, the prediction
    '''
    N, C, H, W = init_label.size()
    total_num = N * H * W
    softmaxed_output = F.softmax(output, dim = 1) 
    mask_true = torch.ne(init_label, 0)
    mask_false = ~mask_true 
    one_hot_gt = torch.cat((mask_true, mask_false), dim = 1).float()
    cross_entropy_loss = -torch.sum(one_hot_gt * torch.log(softmaxed_output + epsilon)) / total_num
    probability_true = softmaxed_output[:, 0:1, :, :]
    predict_true = probability_true > 0.5

    accuracy = float((predict_true == mask_true).float().sum() / total_num)
    return accuracy, cross_entropy_loss, predict_true

def discrimitive_loss(plane_infos, plane_seg_gt, average_plane_info, delta_v, delta_d):
    '''
    description: get the discrimitive loss 
    parameter: the parameter driven by our model, the ground truth segmentation, the ground truth plane id
        the average plane info, threshold of the same plane, threshold of different planes
    return: depth loss
    '''
    batch_size = len(plane_seg_gt)
    max_num = len(average_plane_info[0])
    lvar = [] 
    dvar = []

    for i in range(batch_size):
        a = plane_infos[i][0]
        b = plane_infos[i][1]
        c = plane_infos[i][2]
        d = plane_infos[i][3]
        useful_mask = []

        current_lvar = []
        for seg_id in range(1, max_num):
            mask = torch.eq(plane_seg_gt[i][0], seg_id) 
            mask = mask.detach()

            count = mask.sum()
            useful_mask.append(torch.ne(count, 0).unsqueeze(0))
            da_total = mask * torch.abs(a - average_plane_info[i][seg_id][0])
            db_total = mask * torch.abs(b - average_plane_info[i][seg_id][1])
            dc_total = mask * torch.abs(c - average_plane_info[i][seg_id][2])
            dd_total = mask * torch.abs(d - average_plane_info[i][seg_id][3])
            loss_total = torch.clamp(dp_total + dq_total + dr_total + ds_total - delta_v, min = 0)
            mask_auxiliary = torch.eq(count, 0) #trick
            new_count = count + mask_auxiliary
            new_count = new_count.detach()


            the_sum = loss_total.sum() / new_count 
            current_lvar.append(the_sum.unsqueeze(0))
        useful_mask = torch.cat(useful_mask)
        useful_mask = useful_mask.detach()
        current_lvar = torch.cat(current_lvar)
        C = useful_mask.sum()
        C = C.detach()
        
        lvar.append((current_lvar.sum() / C).unsqueeze(0))


        current_dvar = []
        for ii in range(1, max_num - 1):
            for jj in range(ii + 1, max_num):
                diff_param = torch.abs(average_plane_info[i][ii] - average_plane_info[i][jj])
                diff = diff_param.sum()
                the_sum = torch.clamp(delta_d - diff, min = 0)
                masked_sum = the_sum * useful_mask[ii] * useful_mask[jj]
                current_dvar.append(masked_sum.unsqueeze(0))
        current_dvar = torch.cat(current_dvar) 
        
        dvar_raw = current_dvar.sum() * 2
        #防止/1
        divided = C * (C - 1)
        divided_mask = torch.eq(divided, 0)
        divided_total = divided + divided_mask 
        divided_total = divided_total.detach()
        
        dvar_result = dvar_raw / divided_total
        dvar.append(dvar_result.unsqueeze(0))

    lvar = torch.cat(lvar)
    dvar = torch.cat(dvar)

    total_loss = (lvar + dvar).sum() / batch_size

    return total_loss