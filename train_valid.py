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
    for i, (image, layout_depth, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):
        start = time.time()
        if device:
            image = image.cuda()
            #layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        
        batch_size = image.size(0)
        mask = torch.ne(init_label, 0)


        
        the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()

        '''
        output = model(image)
        loss = ordinal_regression_loss(output, layout_depth, mask, args.ord_num, args.ordinal_beta, args.discretization)
        '''
        output = model(the_input)
        output = normalize(output, args.epsilon)
        loss, loss_per_pixel = get_norm_loss(output, normal, mask)
        
        loss.backward()
        optimizer.step()

        
        mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon, mask)
        average_meter_normal.add_batch(batch_size, loss_per_pixel, mean, median, rmse, d_1125, d_2250, d_30)
        
        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss_per_pixel) + \
            'mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
            .format(mean, median, rmse, d_1125, d_2250, d_30)
        
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)

    avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30 = average_meter_normal.get_average()
    
    result_string = 'Average: Epoch: [{} / {}], Loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
        .format(epoch + 1, args.epochs, avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30)
    '''
    result_string = 'Average: Epoch: [{} / {}], rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' 
        .format(epoch + 1, args.epochs, avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3)
    '''
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
    for i, (image, layout_depth, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
        start = time.time()

        if device:
            image = image.cuda()
            #layout_depth = layout_depth.cuda()
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
            output = normalize(output, args.epsilon)
            loss, loss_per_pixel = get_norm_loss(output, normal, mask)

            mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon, mask)
            average_meter_normal.add_batch(batch_size, loss_per_pixel, mean, median, rmse, d_1125, d_2250, d_30)
            ''' 
            output = model(image)
            my_depth = get_predicted_depth(output, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            rms, rel, rlog10, delta_1, delta_2, delta_3 = depth_metrics(my_depth, layout_depth, mask)
            average_meter.add_batch(batch_size, rms, rel, rlog10, delta_1, delta_2, delta_3)
            '''
        end = time.time()
        the_time = end - start

        result_string = 'Valid: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, loss_per_pixel) + \
            'mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
            .format(mean, median, rmse, d_1125, d_2250, d_30)
        '''
        'rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' 
        .format(rms, rel, rlog10, delta_1, delta_2, delta_3)
        '''

        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)

    avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30 = average_meter_normal.get_average()
    #avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3 = average_meter.get_average()
    
    result_string = 'Average: Epoch: [{} / {}], Loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
        .format(epoch + 1, args.epochs, avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30)
    '''
    result_string = 'Average: Epoch: [{} / {}], rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' 
        .format(epoch + 1, args.epochs, avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3)
    '''
    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 