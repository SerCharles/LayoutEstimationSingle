''' 
The main function of training and validing the network
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

from data.dataset import *
from models.dorn import *
from utils.utils import * 
from utils.metrics import *
from train_valid_utils import * 
from train_valid import * 

def main():
    args = init_args()
    device, dataset_training, dataset_validation, model, optimizer, start_epoch = init_model(args)

    for i in range(start_epoch, args.epochs):
        model = train(args, device, dataset_training, model, optimizer, i)
        valid(args, device, dataset_validation, model, i)

if __name__ == "__main__":
    main()