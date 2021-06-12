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
        #file_name = os.path.join(args.save_dir, args.cur_name, 'log.txt')
        file_name = "C:\\Users\\Lenovo\\Desktop\\log.txt"

        train_loss = []
        train_loss_seg = []
        train_loss_norm = []
        train_loss_depth= []

        valid_loss = []
        valid_loss_seg = []
        valid_loss_norm = []
        valid_loss_depth = []


        with open(file_name) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                words = lines[i].split()
                if(len(words) <= 0):
                    continue
                if words[1] == 'Average:':
                    loss_seg = float(lines[i + 1].split()[1][:-1])
                    loss_norm = float(lines[i + 2].split()[1][:-1])
                    loss_depth = float(lines[i + 3].split()[1][:-1])
 
                    loss_all = loss_seg + loss_norm + loss_depth
                    if words[0] == 'training:':
                        train_loss.append(loss_all)
                        train_loss_seg.append(loss_seg)
                        train_loss_norm.append(loss_norm)
                        train_loss_depth.append(loss_depth)

                    else:
                        valid_loss.append(loss_all)
                        valid_loss_seg.append(loss_seg)
                        valid_loss_norm.append(loss_norm)
                        valid_loss_depth.append(loss_depth)


        return train_loss, train_loss_seg, train_loss_norm, train_loss_depth, valid_loss, valid_loss_seg, valid_loss_norm, valid_loss_depth


    def plot_all(self, train_loss, train_loss_seg, train_loss_norm, train_loss_depth, valid_loss, valid_loss_seg, valid_loss_norm, valid_loss_depth):
        writer = SummaryWriter()
        for epoch in range(len(train_loss)):
            writer.add_scalars('scalar/test', \
                {"train_all" : train_loss[epoch], "train_seg" : train_loss_seg[epoch], "train_norm": train_loss_norm[epoch], "train_depth_gt": train_loss_depth[epoch], \
                    "valid_all" : valid_loss[epoch], "valid_seg" : valid_loss_seg[epoch], "valid_norm": valid_loss_norm[epoch], "valid_depth": valid_loss_depth[epoch]}, epoch)
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
        train_loss, train_loss_seg, train_loss_norm, train_loss_depth, valid_loss, valid_loss_seg, valid_loss_norm, valid_loss_depth = self.get_info()
        while(True):
            print('Please input the type you want to plot')
            print('0: Plot all types of loss of training and testing')
            print('1: Plot only the total loss')
            print('2: Plot only the segmentation loss')
            print('3: Plot only the norm loss')
            print('4: Plot only the depth loss')

            try:
                the_type = int(input())
            except: 
                print('Invalid input, please try again!')
                continue 
            if the_type == 0: 
                self.plot_all(train_loss, train_loss_seg, train_loss_norm, train_loss_depth, valid_loss, valid_loss_seg, valid_loss_norm, valid_loss_depth)
                break 
            elif the_type == 1: 
                self.plot_one(train_loss, valid_loss)
                break
            elif the_type == 2: 
                self.plot_one(train_loss_seg, valid_loss_seg)
                break
            elif the_type == 3: 
                self.plot_one(train_loss_norm, valid_loss_norm)
                break
            elif the_type == 4: 
                self.plot_one(train_loss_depth, valid_loss_depth)
                break
            else: 
                print('Invalid input, please try again!')

if __name__ == "__main__":
    args = init_args()
    a = PlotCurves(args)