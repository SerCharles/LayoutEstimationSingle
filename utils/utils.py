''' 
functions used in mathmatic calculation
'''


import numpy as np
import os
from math import *
from PIL import Image
import PIL
import torch
import torch.nn as nn
from torchvision import transforms

def normalize(norm):
    ''' 
    description: normalize the normal vector 
    parameter: normal vector
    return: normalized normal vector
    '''
    batch_size = norm.size(0)
    nx = norm[:][0 : 1][:][:]
    ny = norm[:][1 : 2][:][:]
    nz = norm[:][2 : 3][:][:]
    length = torch.sqrt(torch.pow(nx, 2) + torch.pow(ny, 2) + torch.pow(nz, 2))
    norm = norm / length
    return norm

