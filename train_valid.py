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
            #layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            #normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            #mesh_x = mesh_x.cuda() 
            #mesh_y = mesh_y.cuda()
        
        batch_size = image.size(0)
        pixel_num = image.size(2) * image.size(3)
        #mask = torch.ne(init_label, 0)


        
        #the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()

        output = model(image)
        accuracy, loss = get_segmentation_loss(output, init_label, args.epsilon)
        '''
        output = model(image)
        loss = ordinal_regression_loss(output, layout_depth, mask, args.ord_num, args.ordinal_beta, args.discretization)
        '''
        '''
        output = model(the_input)
        output = normalize(output, args.epsilon)
        loss, loss_per_pixel = get_norm_loss(output, normal, mask)
        '''
        loss.backward()
        optimizer.step()

        
        #mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon, mask)
        #average_meter_normal.add_batch(batch_size, loss_per_pixel, mean, median, rmse, d_1125, d_2250, d_30)
        average_meter_seg.add_batch(batch_size, loss.item() / batch_size / pixel_num, accuracy)
        
        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('training', epoch + 1, args.epochs, i + 1, len(train_loader), the_time, None) + '\n'
        result_string += get_result_string_seg(loss.item() / batch_size / pixel_num, accuracy)
        
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)

    avg_loss, avg_acc = average_meter_seg.get_average()
    result_string = get_result_string_average('training', epoch + 1, args.epochs, None) + '\n'
    result_string += get_result_string_seg(avg_loss, avg_acc)

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
            #layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            #normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            #mesh_x = mesh_x.cuda() 
            #mesh_y = mesh_y.cuda()
        with torch.no_grad():
            batch_size = image.size(0)
            pixel_num = image.size(2) * image.size(3)
            #mask = torch.ne(init_label, 0)

            '''
            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)
            output = normalize(output, args.epsilon)
            loss, loss_per_pixel = get_norm_loss(output, normal, mask)

            mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon, mask)
            average_meter_normal.add_batch(batch_size, loss_per_pixel, mean, median, rmse, d_1125, d_2250, d_30)
            ''' 
            '''
            output = model(image)
            my_depth = get_predicted_depth(output, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            rms, rel, rlog10, delta_1, delta_2, delta_3 = depth_metrics(my_depth, layout_depth, mask)
            average_meter.add_batch(batch_size, rms, rel, rlog10, delta_1, delta_2, delta_3)
            '''
            output = model(image)
            accuracy, loss = get_segmentation_loss(output, init_label, args.epsilon)
            average_meter_seg.add_batch(batch_size, loss.item() / batch_size / pixel_num, accuracy)


        end = time.time()
        the_time = end - start

        result_string = get_result_string_total('validation', epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, None) + '\n'
        result_string += get_result_string_seg(loss.item() / batch_size / pixel_num, accuracy)
        
        print(result_string)
        write_log(args, epoch, i, 'validation', result_string)

    avg_loss, avg_acc = average_meter_seg.get_average()
    result_string = get_result_string_average('validation', epoch + 1, args.epochs, None) + '\n'
    result_string += get_result_string_seg(avg_loss, avg_acc)

    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 