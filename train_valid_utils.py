''' 
Utils used in training and validing the network
'''


import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import os
import PIL 
import cv2
from PIL import Image
from collections import OrderedDict


from data.dataset import *
from models.dorn import *


def init_args():
    '''
    description: load train args
    parameter: empty
    return: args
    '''
    parser = argparse.ArgumentParser(description = 'PyTorch GeoLayout3D Training')
    parser.add_argument('--seed', default = 1453)
    parser.add_argument('--cuda',  type = int, default = 1, help = 'use GPU or not')
    parser.add_argument('--parallel',  type = int, default = 1, help = 'use multiple GPUs or not')
    parser.add_argument('--gpu_id', type = int, default = 0, help = 'GPU device id used')
    parser.add_argument('--epochs', default = 200, type = int)
    parser.add_argument('--start_epoch', default = 0, type = int,
                    help = 'manual epoch number (useful on restarts)')
    parser.add_argument('--learning_rate', '--lr', default = 1e-3, type = float)
    parser.add_argument('--weight_decay', '--wd',  default = 0, type = float)
    parser.add_argument('--epsilon', default = 1e-8, type = float)
    parser.add_argument('--batch_size', '--bs', default = 8, type = int)
    parser.add_argument('--data_dir', default = '/home/shenguanlin/geolayout', type = str)
    parser.add_argument('--save_dir', default = '/home/shenguanlin/geolayout_result', type = str)
    parser.add_argument('--cur_name', default = 'final', type = str)

    #weight
    parser.add_argument('--weight_seg', default = 1.0, type = float)
    parser.add_argument('--weight_norm_gt', default = 0.5, type = float)
    parser.add_argument('--weight_norm_mine', default = 0.5, type = float)
    parser.add_argument('--weight_depth_gt', default = 0.5, type = float)
    parser.add_argument('--weight_depth_mine', default = 0.5, type = float)
    parser.add_argument('--weight_discrimitive', default = 1.0, type = float)

    #depth
    parser.add_argument('--ord_num', default = 90, type = int)
    parser.add_argument('--ordinal_beta', default = 80.0, type = float)
    parser.add_argument('--ordinal_gamma', default = 1.0, type = float)
    parser.add_argument('--discretization', default = 'UD', type = str)

    #discrimitive_loss
    parser.add_argument('--delta_v', default = 0.1, type = float)
    parser.add_argument('--delta_d', default = 1.0, type = float)

    #post process
    parser.add_argument('--threshold', default = 0.05, type = float)

    args = parser.parse_args()
    return args


def save_checkpoint(args, state, epoch):
    '''
    description: save the checkpoint
    parameter: args, optimizer, epoch_num
    return: empty
    '''
    file_dir = os.path.join(args.save_dir, args.cur_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if (epoch + 1) % 50 == 0:
        filename = os.path.join(file_dir, 'checkpoint_' + str(epoch + 1) + '.pth')
        torch.save(state, filename)

def write_log(args, epoch, batch, the_type, info):
    '''
    description: write the log file
    parameter: args, epoch, type(training/validation/testing), the info you want to write
    return: empty
    '''
    file_dir = os.path.join(args.save_dir, args.cur_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    file_name = os.path.join(file_dir, 'log.txt')
    if epoch == 0 and batch == 0 and the_type == 'training':
        file = open(file_name, 'w')
        file.close()
    file = open(file_name, 'a')
    file.write(info + '\n')
    file.close()

def init_model(args):
    '''
    description: init the device, dataloader, model, optimizer of the model
    parameter: args
    return: device, dataloader_train, dataloader_valid, model, optimizer
    '''
    print(args)
    print('getting device...')
    torch.manual_seed(args.seed)
    if args.cuda == 1:
        device = True
        
        '''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) + ',' + str(args.gpu_id + 1) + ',' + str(args.gpu_id + 2) + ',' + str(args.gpu_id + 3) + \
        ',' + str(args.gpu_id + 4) + ',' + str(args.gpu_id + 5) + ',' + str(args.gpu_id + 6) + ',' + str(args.gpu_id + 7)+ ',' + str(args.gpu_id + 8)
        '''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) + ',' + str(args.gpu_id + 1) + ',' + str(args.gpu_id + 2) + ',' + str(args.gpu_id + 3)
    else:
        device = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #print(device)

    print('Initialize model')
    
    model = DORN(channel = 5, output_channel = args.ord_num * 2 + 5)


    if device:
        if args.parallel: 
            #model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
            model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3]).cuda()
        else: 
            model = model.cuda()

    if(args.start_epoch != 0):
        file_dir = os.path.join(args.save_dir, args.cur_name)
        filename = os.path.join(file_dir, 'checkpoint_' + str(args.start_epoch) + '.pth')
        model.load_state_dict(torch.load(filename))


    print('Getting optimizer')
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay = args.weight_decay)


    print('Getting dataset')
    dataset_training = MatterPortDataSet(args.data_dir, 'training')
    dataset_validation = MatterPortDataSet(args.data_dir, 'validation')
    dataloader_training = DataLoader(dataset_training, batch_size = args.batch_size, shuffle = True, num_workers = 2)
    dataloader_validation = DataLoader(dataset_validation, batch_size = args.batch_size, shuffle = False, num_workers = 2)
    print('Data got!')

    return device, dataloader_training, dataloader_validation, model, optimizer, args.start_epoch

def init_valid_model(args):
    '''
    description: init the device, dataloader, model, optimizer of the model for validation
    parameter: args
    return: device, dataloader_valid, model,
    '''
    print(args)
    print('getting device...')
    torch.manual_seed(args.seed)
    if args.cuda == 1:
        device = True
        
        '''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) + ',' + str(args.gpu_id + 1) + ',' + str(args.gpu_id + 2) + ',' + str(args.gpu_id + 3) + \
        ',' + str(args.gpu_id + 4) + ',' + str(args.gpu_id + 5) + ',' + str(args.gpu_id + 6) + ',' + str(args.gpu_id + 7)+ ',' + str(args.gpu_id + 8)
        '''
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) + ',' + str(args.gpu_id + 1) + ',' + str(args.gpu_id + 2) + ',' + str(args.gpu_id + 3)
    else:
        device = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    #print(device)

    print('Initialize model')
    
    model = DORN(channel = 5, output_channel = args.ord_num * 2 + 5)


    if device:
        if args.parallel: 
            #model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
            model = torch.nn.DataParallel(model, device_ids = [0, 1, 2, 3]).cuda()
        else: 
            model = model.cuda()

    file_dir = os.path.join(args.save_dir, args.cur_name)
    filename = os.path.join(file_dir, 'checkpoint_' + str(args.epochs) + '.pth')
    model.load_state_dict(torch.load(filename))



    print('Getting dataset')
    dataset_validation = MatterPortDataSet(args.data_dir, 'validation')
    dataloader_validation = DataLoader(dataset_validation, batch_size = args.batch_size, shuffle = False, num_workers = 2)
    print('Data got!')

    return device, dataset_validation, dataloader_validation, model

