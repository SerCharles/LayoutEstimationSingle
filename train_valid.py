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
    average_meter_depth = AverageMeterDepth()
    average_meter_normal = AverageMeterNorm()
    average_meter_seg = AverageMeterSeg()
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
        mask = torch.ne(init_label, 0)

        the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()

        output = model(the_input)  
        seg_result = output[:, 0:2, :, :]
        norm_result = output[:, 2:5, :, :]
        depth_result = output[:, 5:, :, :]

        accuracy_seg, loss_seg = get_segmentation_loss(seg_result, init_label, args.epsilon)

        norm_result = normalize(norm_result, args.epsilon)
        loss_norm = get_norm_loss(norm_result, normal, mask)

        loss_depth = ordinal_regression_loss(depth_result, layout_depth, mask, args.ord_num, args.ordinal_beta, args.discretization)
        
        loss = loss_seg * args.weight_seg + loss_norm * args.weight_norm + loss_depth * args.weight_depth
        
        loss.backward()
        optimizer.step()

        
        mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(norm_result, normal, args.epsilon, mask)
        average_meter_normal.add_batch(batch_size, loss_norm.item(), mean, median, rmse, d_1125, d_2250, d_30)

        average_meter_seg.add_batch(batch_size, loss_seg.item(), accuracy_seg)
        
        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('training', epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss.item()) + '\n'
        result_string += get_result_string_seg(loss_seg.item(), accuracy_seg) + '\n'
        result_string += get_result_string_norm(loss_norm.item(), mean, median, rmse, d_1125, d_2250, d_30)
        
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)

    avg_loss_seg, avg_acc = average_meter_seg.get_average()
    avg_loss_normal, avg_mean, avg_median, avg_rmse, avg_d_1125, avg_d_2250, avg_d_30 = average_meter_normal.get_average()
    result_string = get_result_string_average('training', epoch + 1, args.epochs, None) + '\n'
    result_string += get_result_string_seg(avg_loss_seg, avg_acc) + '\n'
    result_string += get_result_string_norm(avg_loss_normal, avg_mean, avg_median, avg_rmse, avg_d_1125, avg_d_2250, avg_d_30)

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
    average_meter_depth = AverageMeterDepth()
    average_meter_normal = AverageMeterNorm()
    average_meter_seg = AverageMeterSeg()

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
            mask = torch.ne(init_label, 0)

            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)
            seg_result = output[:, 0:2, :, :]
            norm_result = output[:, 2:5, :, :]
            depth_result = output[:, 5:, :, :]



            accuracy_seg, loss_seg = get_segmentation_loss(seg_result, init_label, args.epsilon)
            average_meter_seg.add_batch(batch_size, loss_seg.item(), accuracy_seg)

            norm_result = normalize(norm_result, args.epsilon)
            loss_norm = get_norm_loss(norm_result, normal, mask)
            mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(norm_result, normal, args.epsilon, mask)
            average_meter_normal.add_batch(batch_size, loss_norm.item(), mean, median, rmse, d_1125, d_2250, d_30)

            my_depth = get_predicted_depth(depth_result, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            rms, rel, rlog10, delta_1, delta_2, delta_3 = depth_metrics(my_depth, layout_depth, mask)
            average_meter_depth.add_batch(batch_size, rms, rel, rlog10, delta_1, delta_2, delta_3)


        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('validation', epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, None) + '\n'
        result_string += get_result_string_seg(loss_seg.item(), accuracy_seg) + '\n'
        result_string += get_result_string_norm(loss_norm.item(), mean, median, rmse, d_1125, d_2250, d_30) + '\n'
        result_string += get_result_string_depth(rms, rel, rlog10, delta_1, delta_2, delta_3)

        print(result_string)
        write_log(args, epoch, i, 'validation', result_string)

    avg_loss_seg, avg_acc = average_meter_seg.get_average()
    avg_loss_normal, avg_mean, avg_median, avg_rmse, avg_d_1125, avg_d_2250, avg_d_30 = average_meter_normal.get_average()
    avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3 = average_meter_depth.get_average()
    result_string = get_result_string_average('validation', epoch + 1, args.epochs, None) + '\n'
    result_string += get_result_string_seg(avg_loss_seg, avg_acc) + '\n'
    result_string += get_result_string_norm(avg_loss_normal, avg_mean, avg_median, avg_rmse, avg_d_1125, avg_d_2250, avg_d_30) + '\n'
    result_string += get_result_string_depth(avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3)

    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 