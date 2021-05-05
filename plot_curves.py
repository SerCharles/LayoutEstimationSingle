'''
Plot the train and test curve
'''

import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboardX import SummaryWriter

from train_valid_utils import *

class PlotCurves:
    def get_info(self):
        '''
        description: get the train and test loss info from the log
        parameter: empty
        return: the average train and test loss info
        '''
        file_name = os.path.join(args.save_dir, args.cur_name, 'log.txt')
        #file_name = "C:\\Users\\Lenovo\\Desktop\\log.txt"

        train_loss = []
        train_loss_seg = []
        train_loss_norm_gt = []
        train_loss_norm_mine = []
        train_loss_depth_gt = []
        train_loss_depth_mine = []

        valid_loss = []
        valid_loss_seg = []
        valid_loss_norm_gt = []
        valid_loss_norm_mine = []
        valid_loss_depth_gt = []
        valid_loss_depth_mine = []

        with open(file_name) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                words = lines[i].split()
                if(len(words) <= 0):
                    continue
                if words[1] == 'Average:':
                    loss_seg = float(lines[i + 1].split()[1][:-1])
                    loss_norm_gt = float(lines[i + 2].split()[1][:-1])
                    loss_norm_mine = float(lines[i + 3].split()[1][:-1])
                    loss_depth_gt = float(lines[i + 4].split()[1][:-1])
                    loss_depth_mine = float(lines[i + 5].split()[1][:-1])
                    loss_all = loss_seg + loss_norm_gt + loss_norm_mine + loss_depth_gt + loss_depth_mine
                    if words[0] == 'training:':
                        train_loss.append(loss_all)
                        train_loss_seg.append(loss_seg)
                        train_loss_norm_gt.append(loss_norm_gt)
                        train_loss_norm_mine.append(loss_norm_mine)
                        train_loss_depth_gt.append(loss_depth_gt)
                        train_loss_depth_mine.append(loss_depth_mine)
                    else:
                        valid_loss.append(loss_all)
                        valid_loss_seg.append(loss_seg)
                        valid_loss_norm_gt.append(loss_norm_gt)
                        valid_loss_norm_mine.append(loss_norm_mine)
                        valid_loss_depth_gt.append(loss_depth_gt)
                        valid_loss_depth_mine.append(loss_depth_mine)


        return train_loss, train_loss_seg, train_loss_norm_gt, train_loss_norm_mine, train_loss_depth_gt, train_loss_depth_mine, \
            valid_loss, valid_loss_seg, valid_loss_norm_gt, valid_loss_norm_mine, valid_loss_depth_gt, valid_loss_depth_mine


    def plot_all(self, train_loss, train_loss_seg, train_loss_norm_gt, train_loss_norm_mine, train_loss_depth_gt, train_loss_depth_mine, \
            valid_loss, valid_loss_seg, valid_loss_norm_gt, valid_loss_norm_mine, valid_loss_depth_gt, valid_loss_depth_mine):
        writer = SummaryWriter()
        for epoch in range(len(train_loss)):
            writer.add_scalars('scalar/test', \
                {"train_all" : train_loss[epoch], "train_seg" : train_loss_seg[epoch], "train_norm_gt": train_loss_norm_gt[epoch], \
                    "train_norm_mine": train_loss_norm_mine[epoch], "train_depth_gt": train_loss_depth_gt[epoch], "train_depth_mine": train_loss_depth_mine[epoch], \
                    "valid_all" : valid_loss[epoch], "valid_seg" : valid_loss_seg[epoch], "valid_norm_gt": valid_loss_norm_gt[epoch], \
                    "valid_norm_mine": valid_loss_norm_mine[epoch], "valid_depth_gt": valid_loss_depth_gt[epoch], "valid_depth_mine": train_loss_depth_mine[epoch]}, epoch)
        writer.close()    
        
    def plot_one(self, train_loss, valid_loss):
        '''
        description: plot one kind of train/valid loss on the chart
        parameter: train and valid loss
        return: empty
        '''
        writer = SummaryWriter()
        for epoch in range(len(train_loss)):
            writer.add_scalars('scalar/test', {"train" : train_loss[epoch], "valid" : valid_loss[epoch]}, epoch)
        writer.close()



    def __init__(self, args):
        ''' 
        description: the main function of plotting
        parameter: args
        return: empty
        '''
        super(PlotCurves, self).__init__()
        self.args = args
        train_loss, train_loss_seg, train_loss_norm_gt, train_loss_norm_mine, train_loss_depth_gt, train_loss_depth_mine, \
            valid_loss, valid_loss_seg, valid_loss_norm_gt, valid_loss_norm_mine, valid_loss_depth_gt, valid_loss_depth_mine = self.get_info()
        while(True):
            print('Please input the type you want to plot')
            print('0: Plot all types of loss of training and testing')
            print('1: Plot only the total loss')
            print('2: Plot only the segmentation loss')
            print('3: Plot only the norm loss of gt seg')
            print('4: Plot only the norm loss of my seg')
            print('5: Plot only the depth loss of gt seg')
            print('6: Plot only the depth loss of my seg')

            try:
                the_type = int(input())
            except: 
                print('Invalid input, please try again!')
                continue 
            if the_type == 0: 
                self.plot_all(train_loss, train_loss_seg, train_loss_norm_gt, train_loss_norm_mine, train_loss_depth_gt, train_loss_depth_mine, \
                    valid_loss, valid_loss_seg, valid_loss_norm_gt, valid_loss_norm_mine, valid_loss_depth_gt, valid_loss_depth_mine)
                break 
            elif the_type == 1: 
                self.plot_one(train_loss, valid_loss)
                break
            elif the_type == 2: 
                self.plot_one(train_loss_seg, valid_loss_seg)
                break
            elif the_type == 3: 
                self.plot_one(train_loss_norm_gt, valid_loss_norm_gt)
                break
            elif the_type == 4: 
                self.plot_one(train_loss_norm_mine, valid_loss_norm_mine)
                break
            elif the_type == 5: 
                self.plot_one(train_loss_depth_gt, valid_loss_depth_gt)
                break
            elif the_type == 6: 
                self.plot_one(train_loss_depth_mine, valid_loss_depth_mine)
                break
            else: 
                print('Invalid input, please try again!')

if __name__ == "__main__":
    args = init_args()
    a = PlotCurves(args)