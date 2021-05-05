''' 
The function of training and validing one epoch of the network
'''

import numpy as np
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

from data.dataset import *
from models.dorn import * 
from train_valid_utils import *
from utils.utils import *
from utils.metrics import *
from utils.average_meters import *



def train(args, device, train_loader, model, optimizer, epoch):
    '''
    description: train one epoch of the model 
    parameter: args, device, train_loader, model, optimizer, epoch
    return: the model trained
    '''
    model.train()
    average_meter_seg = AverageMeterSeg()
    average_meter_normal_gt = AverageMeterNorm()
    average_meter_normal_mine = AverageMeterNorm()
    average_meter_depth_gt = AverageMeterDepth()
    average_meter_depth_mine = AverageMeterDepth()
    total_loss = 0.0
    total_num = 0
    for i, (image, layout_depth, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):
        start = time.time()
        if device:
            image = image.cuda()
            layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        
        batch_size = image.size(0)
        mask_gt = torch.ne(init_label, 0)

        the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()

        output = model(the_input)  
        seg_result = output[:, 0:2, :, :]
        norm_result = output[:, 2:5, :, :]
        depth_result = output[:, 5:, :, :]

        accuracy_seg, loss_seg, mask_mine = get_segmentation_loss(seg_result, init_label, args.epsilon)
        loss_seg = loss_seg *  args.weight_seg
        average_meter_seg.add_batch(batch_size, loss_seg.item(), accuracy_seg)

        norm_result = normalize(norm_result, args.epsilon)
        loss_norm_gt = get_norm_loss(norm_result, normal, mask_gt) * args.weight_norm_gt
        loss_norm_mine = get_norm_loss(norm_result, normal, mask_mine) * args.weight_norm_mine
        mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt = norm_metrics(norm_result, normal, args.epsilon, mask_gt)
        mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine = norm_metrics(norm_result, normal, args.epsilon, mask_mine)
        average_meter_normal_gt.add_batch(batch_size, loss_norm_gt.item(), mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt)
        average_meter_normal_mine.add_batch(batch_size, loss_norm_mine.item(), mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine)
        loss_norm = loss_norm_gt + loss_norm_mine

        loss_depth_gt = ordinal_regression_loss(depth_result, layout_depth, mask_gt, args.ord_num, args.ordinal_beta, args.discretization) * args.weight_depth_gt
        loss_depth_mine = ordinal_regression_loss(depth_result, layout_depth, mask_mine, args.ord_num, args.ordinal_beta, args.discretization) * args.weight_depth_mine
        my_depth = get_predicted_depth(depth_result, args.ordinal_beta, args.ordinal_gamma, args.discretization)
        rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt = depth_metrics(my_depth, layout_depth, mask_gt)
        rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine = depth_metrics(my_depth, layout_depth, mask_mine)
        average_meter_depth_gt.add_batch(batch_size, loss_depth_gt.item(), rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt)
        average_meter_depth_mine.add_batch(batch_size, loss_depth_mine.item(), rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine)
        loss_depth = loss_depth_gt + loss_depth_mine

        loss = loss_seg + loss_norm + loss_depth
        total_loss += loss.item() * batch_size
        total_num += batch_size
        loss.backward()
        optimizer.step()

        
        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('training', epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss.item()) + '\n'
        result_string += get_result_string_seg(loss_seg.item(), accuracy_seg) + '\n'
        result_string += get_result_string_norm(loss_norm_gt.item(), mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt) + '\n'
        result_string += get_result_string_norm(loss_norm_mine.item(), mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine) + '\n'
        result_string += get_result_string_depth(loss_depth_gt.item(), rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt) + '\n'
        result_string += get_result_string_depth(loss_depth_mine.item(), rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine)

        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
    avg_loss = total_loss / total_num
    avg_loss_seg, avg_acc = average_meter_seg.get_average()
    avg_loss_normal_gt, avg_mean_gt, avg_median_gt, avg_rmse_gt, avg_d_1125_gt, avg_d_2250_gt, avg_d_30_gt = average_meter_normal_gt.get_average()
    avg_loss_normal_mine, avg_mean_mine, avg_median_mine, avg_rmse_mine, avg_d_1125_mine, avg_d_2250_mine, avg_d_30_mine = average_meter_normal_mine.get_average()
    avg_loss_depth_gt, avg_rms_gt, avg_rel_gt, avg_log10_gt, avg_delta_1_gt, avg_delta_2_gt, avg_delta_3_gt = average_meter_depth_gt.get_average()
    avg_loss_depth_mine, avg_rms_mine, avg_rel_mine, avg_log10_mine, avg_delta_1_mine, avg_delta_2_mine, avg_delta_3_mine = average_meter_depth_mine.get_average()

    result_string = get_result_string_average('training', epoch + 1, args.epochs, avg_loss) + '\n'
    result_string += get_result_string_seg(avg_loss_seg, avg_acc) + '\n'
    result_string += get_result_string_norm(avg_loss_normal_gt, avg_mean_gt, avg_median_gt, avg_rmse_gt, avg_d_1125_gt, avg_d_2250_gt, avg_d_30_gt) + '\n'
    result_string += get_result_string_norm(avg_loss_normal_mine, avg_mean_mine, avg_median_mine, avg_rmse_mine, avg_d_1125_mine, avg_d_2250_mine, avg_d_30_mine) + '\n'
    result_string += get_result_string_depth(avg_loss_depth_gt, avg_rms_gt, avg_rel_gt, avg_log10_gt, avg_delta_1_gt, avg_delta_2_gt, avg_delta_3_gt) + '\n'
    result_string += get_result_string_depth(avg_loss_depth_mine, avg_rms_mine, avg_rel_mine, avg_log10_mine, avg_delta_1_mine, avg_delta_2_mine, avg_delta_3_mine)

    print(result_string)
    write_log(args, epoch, 1, 'training', result_string)

    return model
 

 
def valid(args, device, valid_loader, model, epoch):
    '''
    description: valid one epoch of the model 
    parameter: args, device, train_loader, modelr, epoch
    return: empty
    '''
    model.eval()
    average_meter_seg = AverageMeterSeg()
    average_meter_normal_gt = AverageMeterNorm()
    average_meter_normal_mine = AverageMeterNorm()
    average_meter_depth_gt = AverageMeterDepth()
    average_meter_depth_mine = AverageMeterDepth()
    total_loss = 0.0
    total_num = 0
    for i, (image, layout_depth, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
        start = time.time()

        if device:
            image = image.cuda()
            layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        with torch.no_grad():
            batch_size = image.size(0)
            mask_gt = torch.ne(init_label, 0)

            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)  
            seg_result = output[:, 0:2, :, :]
            norm_result = output[:, 2:5, :, :]
            depth_result = output[:, 5:, :, :]

            accuracy_seg, loss_seg, mask_mine = get_segmentation_loss(seg_result, init_label, args.epsilon)
            loss_seg = loss_seg *  args.weight_seg
            average_meter_seg.add_batch(batch_size, loss_seg.item(), accuracy_seg)

            norm_result = normalize(norm_result, args.epsilon)
            loss_norm_gt = get_norm_loss(norm_result, normal, mask_gt) * args.weight_norm_gt
            loss_norm_mine = get_norm_loss(norm_result, normal, mask_mine) * args.weight_norm_mine
            mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt = norm_metrics(norm_result, normal, args.epsilon, mask_gt)
            mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine = norm_metrics(norm_result, normal, args.epsilon, mask_mine)
            average_meter_normal_gt.add_batch(batch_size, loss_norm_gt.item(), mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt)
            average_meter_normal_mine.add_batch(batch_size, loss_norm_mine.item(), mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine)
            loss_norm = loss_norm_gt + loss_norm_mine

            loss_depth_gt = ordinal_regression_loss(depth_result, layout_depth, mask_gt, args.ord_num, args.ordinal_beta, args.discretization) * args.weight_depth_gt
            loss_depth_mine = ordinal_regression_loss(depth_result, layout_depth, mask_mine, args.ord_num, args.ordinal_beta, args.discretization) * args.weight_depth_mine
            my_depth = get_predicted_depth(depth_result, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt = depth_metrics(my_depth, layout_depth, mask_gt)
            rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine = depth_metrics(my_depth, layout_depth, mask_mine)
            average_meter_depth_gt.add_batch(batch_size, loss_depth_gt.item(), rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt)
            average_meter_depth_mine.add_batch(batch_size, loss_depth_mine.item(), rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine)
            loss_depth = loss_depth_gt + loss_depth_mine

            loss = loss_seg + loss_norm + loss_depth
            total_loss += loss.item() * batch_size
            total_num += batch_size


        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('validation', epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, loss.item()) + '\n'
        result_string += get_result_string_seg(loss_seg.item(), accuracy_seg) + '\n'
        result_string += get_result_string_norm(loss_norm_gt.item(), mean_gt, median_gt, rmse_gt, d_1125_gt, d_2250_gt, d_30_gt) + '\n'
        result_string += get_result_string_norm(loss_norm_mine.item(), mean_mine, median_mine, rmse_mine, d_1125_mine, d_2250_mine, d_30_mine) + '\n'
        result_string += get_result_string_depth(loss_depth_gt.item(), rms_gt, rel_gt, rlog10_gt, delta_1_gt, delta_2_gt, delta_3_gt) + '\n'
        result_string += get_result_string_depth(loss_depth_mine.item(), rms_mine, rel_mine, rlog10_mine, delta_1_mine, delta_2_mine, delta_3_mine)


        print(result_string)
        write_log(args, epoch, i, 'validation', result_string)

    avg_loss = total_loss / total_num
    avg_loss_seg, avg_acc = average_meter_seg.get_average()
    avg_loss_normal_gt, avg_mean_gt, avg_median_gt, avg_rmse_gt, avg_d_1125_gt, avg_d_2250_gt, avg_d_30_gt = average_meter_normal_gt.get_average()
    avg_loss_normal_mine, avg_mean_mine, avg_median_mine, avg_rmse_mine, avg_d_1125_mine, avg_d_2250_mine, avg_d_30_mine = average_meter_normal_mine.get_average()
    avg_loss_depth_gt, avg_rms_gt, avg_rel_gt, avg_log10_gt, avg_delta_1_gt, avg_delta_2_gt, avg_delta_3_gt = average_meter_depth_gt.get_average()
    avg_loss_depth_mine, avg_rms_mine, avg_rel_mine, avg_log10_mine, avg_delta_1_mine, avg_delta_2_mine, avg_delta_3_mine = average_meter_depth_mine.get_average()

    result_string = get_result_string_average('validation', epoch + 1, args.epochs, avg_loss) + '\n'
    result_string += get_result_string_seg(avg_loss_seg, avg_acc) + '\n'
    result_string += get_result_string_norm(avg_loss_normal_gt, avg_mean_gt, avg_median_gt, avg_rmse_gt, avg_d_1125_gt, avg_d_2250_gt, avg_d_30_gt) + '\n'
    result_string += get_result_string_norm(avg_loss_normal_mine, avg_mean_mine, avg_median_mine, avg_rmse_mine, avg_d_1125_mine, avg_d_2250_mine, avg_d_30_mine) + '\n'
    result_string += get_result_string_depth(avg_loss_depth_gt, avg_rms_gt, avg_rel_gt, avg_log10_gt, avg_delta_1_gt, avg_delta_2_gt, avg_delta_3_gt) + '\n'
    result_string += get_result_string_depth(avg_loss_depth_mine, avg_rms_mine, avg_rel_mine, avg_log10_mine, avg_delta_1_mine, avg_delta_2_mine, avg_delta_3_mine)

    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 