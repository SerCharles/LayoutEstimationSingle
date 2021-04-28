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
    #average_meter = AverageMeter()
    #for i, (image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):
    #for i, (image, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):
    for i, (image, depth, intrinsic) in enumerate(train_loader):
        start = time.time()
        if device:
            image = image.cuda()
            depth = depth.cuda()
            #layout_depth = layout_depth.cuda()
            #layout_seg = layout_seg.cuda()
            #normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            #mesh_x = mesh_x.cuda() 
            #mesh_y = mesh_y.cuda()
        
        batch_size = image.size(0)
        num_pixels = image.size(2) * image.size(3)

        
        #the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()
        loss = model(image, depth).mean()
        '''
        output = model(the_input)
        output = normalize(output, args.epsilon)
        output_plain = output.view(batch_size, -1)
        normal_plain = normal.view(batch_size, -1)
        loss = torch.sum((output_plain - normal_plain) ** 2)
        '''
        loss.backward()
        optimizer.step()

        '''
        mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon)
        the_avg_loss = loss.item() / batch_size / num_pixels
        average_meter.add_batch(batch_size, the_avg_loss, mean, median, rmse, d_1125, d_2250, d_30)
        '''
        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, loss.item())

        
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
    

    return model
 

 
def valid(args, device, valid_loader, model, epoch):
    '''
    description: valid one epoch of the model 
    parameter: args, device, train_loader, modelr, epoch
    return: empty
    '''
    model.eval()
    average_meter = AverageMeterDepth()
    #for i, (image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
    #for i, (image, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
    for i, (image, depth, intrinsic) in enumerate(valid_loader):

        start = time.time()

        if device:
            image = image.cuda()
            #layout_depth = layout_depth.cuda()
            #layout_seg = layout_seg.cuda()
            #normal = normal.cuda()
            depth = depth.cuda()
            intrinsic = intrinsic.cuda()
            #mesh_x = mesh_x.cuda() 
            #mesh_y = mesh_y.cuda()

        with torch.no_grad():
            batch_size = image.size(0)
            num_pixels = image.size(2) * image.size(3)

            '''
            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)
            output = normalize(output, args.epsilon)
            output_plain = output.view(batch_size, -1)
            normal_plain = normal.view(batch_size, -1)
            loss = torch.sum((output_plain - normal_plain) ** 2)
            mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon)
            the_avg_loss = loss.item() / num_pixels / batch_size
            average_meter.add_batch(batch_size, the_avg_loss, mean, median, rmse, d_1125, d_2250, d_30)
            ''' 
            my_depth = model(image, depth)
            rms, rel, rlog10, rate_1, rate_2, rate_3 = depth_metrics(my_depth, depth)
            average_meter.add_batch(batch_size, rms, rel, log10, delta_1, delta_2, delta_3)

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, the_avg_loss) + \
            'rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' \
            .format(rms, rel, rlog10, rate_1, rate_2, rate_3)
        '''
        'mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' 
        .format(mean, median, rmse, d_1125, d_2250, d_30)
        '''
        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)

    #avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30 = average_meter.get_average()
    avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3 = average_meter.get_average()
    '''
    result_string = 'Average: Epoch: [{} / {}], Loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' 
        .format(epoch + 1, args.epochs, avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30)
    '''
    result_string = 'Average: Epoch: [{} / {}], rms: {:.4f}, rel: {:.4f}, log10: {:.4f}, delta_1: {:.3f}, delta_2: {:.3f}, delta_3: {:.3f}' \
        .format(epoch + 1, args.epochs, avg_rms, avg_rel, avg_log10, avg_delta_1, avg_delta_2, avg_delta_3)
    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 