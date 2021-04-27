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

class AverageMeter(object):
    def __init__(self):
        self.total_num = 0
        self.total_loss = 0
        self.total_mean = 0
        self.total_median = 0
        self.total_rmse = 0
        self.total_1125 = 0
        self.total_2250 = 0
        self.total_30 = 0
    
    def add_batch(self, length, loss, mean, median, rmse, d_1125, d_2250, d_30):
        self.total_num += length 
        self.total_loss += length * loss 
        self.total_mean += length * mean 
        self.total_median += length * median 
        self.total_rmse += length * rmse 
        self.total_1125 += length * d_1125 
        self.total_2250 += length * d_2250 
        self.total_30 += length * d_30 
    
    def get_average(self):
        avg_loss = self.total_loss / self.total_num 
        avg_mean = self.total_mean / self.total_num
        avg_median = self.total_median / self.total_num
        avg_rmse = self.total_rmse / self.total_num
        avg_1125 = self.total_1125 / self.total_num
        avg_2250 = self.total_2250 / self.total_num
        avg_30 = self.total_30 / self.total_num
        return avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30

def train(args, device, train_loader, model, optimizer, epoch):
    '''
    description: train one epoch of the model 
    parameter: args, device, train_loader, model, optimizer, epoch
    return: the model trained
    '''
    model.train()
    average_meter = AverageMeter()
    #for i, (image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):
    for i, (image, normal, intrinsic, mesh_x, mesh_y) in enumerate(train_loader):

        start = time.time()
        if device:
            image = image.cuda()
            #layout_depth = layout_depth.cuda()
            #layout_seg = layout_seg.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        
        batch_size = image.size(0)
        the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
        optimizer.zero_grad()
        output = model(the_input)
        output = normalize(output, args.epsilon)
        output_plain = output.view(batch_size, -1)
        normal_plain = normal.view(batch_size, -1)
        loss = torch.sum((output_plain - normal_plain) ** 2)
        mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon)
        the_avg_loss = loss.item() / (batch_size * output_plain.size(1))
        average_meter.add_batch(batch_size, the_avg_loss, mean, median, rmse, d_1125, d_2250, d_30)

        loss.backward()
        optimizer.step()

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(train_loader), the_time, the_avg_loss) + \
            'mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
            .format(mean, median, rmse, d_1125, d_2250, d_30)
        
        print(result_string)
        write_log(args, epoch, i, 'training', result_string)
    
    
    avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30 = average_meter.get_average()
    result_string = 'Average: Epoch: [{} / {}], Loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
        .format(epoch + 1, args.epochs, avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30)
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
    average_meter = AverageMeter()
    #for i, (image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
    for i, (image, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):

        start = time.time()

        if device:
            image = image.cuda()
            #layout_depth = layout_depth.cuda()
            #layout_seg = layout_seg.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()

        with torch.no_grad():
            batch_size = image.size(0)
            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)
            output = normalize(output, args.epsilon)
            output_plain = output.view(batch_size, -1)
            normal_plain = normal.view(batch_size, -1)
            loss = torch.sum((output_plain - normal_plain) ** 2)
            mean, median, rmse, d_1125, d_2250, d_30 = norm_metrics(output, normal, args.epsilon)
            the_avg_loss = loss.item() / (batch_size * output_plain.size(1))
            average_meter.add_batch(batch_size, the_avg_loss, mean, median, rmse, d_1125, d_2250, d_30)

        end = time.time()
        the_time = end - start

        result_string = 'Train: Epoch: [{} / {}], Batch: [{} / {}], Time {:.3f}s, Loss {:.4f}\n' \
            .format(epoch + 1, args.epochs, i + 1, len(valid_loader), the_time, the_avg_loss) + \
            'mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
            .format(mean, median, rmse, d_1125, d_2250, d_30)
        write_log(args, epoch, i, 'validation', result_string)
        print(result_string)

    avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30 = average_meter.get_average()
    result_string = 'Average: Epoch: [{} / {}], Loss {:.4f}, mean: {:.4f}, median: {:.4f}, rmse: {:.4f}, 11.25: {:.3f}, 22.50: {:.3f}, 30: {:.3f}' \
        .format(epoch + 1, args.epochs, avg_loss, avg_mean, avg_median, avg_rmse, avg_1125, avg_2250, avg_30)
    print(result_string)
    write_log(args, epoch, 1, 'validation', result_string)
    save_checkpoint(args, model.state_dict(), epoch)
 