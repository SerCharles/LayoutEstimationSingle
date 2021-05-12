'''
Used in clearing the pretrain dataset, in case it may have pictures in geolayout
'''
import h5py
import numpy as np
import os
import torch
import scipy.io as sio
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import PIL


base_dir = '/home/shenguanlin/geolayout_pretrain'
base_dir_geolayout = '/home/shenguanlin/geolayout'


def save_remove(name):
    '''
    description: remove a file safely
    parameter: file name
    return: empty
    '''
    try: 
        os.remove(name)
        print('removed', name)
    except: 
        pass

def clear_one(base_dir, base_dir_geolayout, type):
    ''' 
    description: clearing the data in geolayout dataset
    parameter: the base dir of pretrained data, the base_dir of geolayout data, the type of data
    return: empty
    '''
    for name in glob.glob(os.path.join(base_dir_geolayout, type, 'image', '*.jpg')):
        filename = name.split(os.sep)[-1]
        base_name = filename[:-9]
        group_name = filename[-7]
        ins_name = filename[-5]
        image_name = filename 
        depth_name = base_name + '_d' + group_name + '_' + ins_name + '.png'
        nx_name = base_name + '_d' + group_name + '_' + ins_name + '_nx.png'
        ny_name = base_name + '_d' + group_name + '_' + ins_name + '_ny.png'
        nz_name = base_name + '_d' + group_name + '_' + ins_name + '_nz.png'
        boundary_name = base_name + '_d' + group_name + '_' + ins_name + '_boundary.png'
        radius_name = base_name + '_d' + group_name + '_' + ins_name + '_radius.png'

        #intrinsic_name = base_name + '_intrinsics_' + group_name + '.txt'
        save_remove(os.path.join(base_dir, 'image', image_name))
        save_remove(os.path.join(base_dir, 'depth', depth_name))
        save_remove(os.path.join(base_dir, 'norm', nx_name))
        save_remove(os.path.join(base_dir, 'norm', ny_name))
        save_remove(os.path.join(base_dir, 'norm', nz_name))
        save_remove(os.path.join(base_dir, 'norm', boundary_name))
        save_remove(os.path.join(base_dir, 'norm', radius_name))

def clear_norm(base_dir):
    ''' 
    description: clearing the useless norm data
    parameter: the base dir of pretrained data
    return: empty
    '''
    for name in glob.glob(os.path.join(base_dir, 'image', '*.jpg')):
        filename = name.split(os.sep)[-1]
        base_name = filename[:-9]
        group_name = filename[-7]
        ins_name = filename[-5]
        boundary_name = base_name + '_d' + group_name + '_' + ins_name + '_boundary.png'
        radius_name = base_name + '_d' + group_name + '_' + ins_name + '_radius.png'
        save_remove(os.path.join(base_dir, 'norm', boundary_name))
        save_remove(os.path.join(base_dir, 'norm', radius_name))

def data_split(base_dir):
    ''' 
    description: split the train and valid data
    parameter: the base dir of pretrained data
    return: empty
    '''
    image_filenames = glob.glob(os.path.join(base_dir, 'image', '*.jpg'))
    length = len(image_filenames)
    threshold = int(length * 0.9)

    f_train = open(os.path.join(base_dir, 'training.txt'), 'w')
    f_valid = open(os.path.join(base_dir, 'validation.txt'), 'w')


    for i in range(length):
        the_name = image_filenames[i].split(os.sep)[-1]
        if i < threshold:
            f_train.write(the_name + '\n')
        else: 
            f_valid.write(the_name + '\n')
    f_train.close()
    f_valid.close()

def clear(base_dir, base_dir_geolayout):
    '''
    description: the main function of data clearing
    parameter: the base dir of pretrained data and geolayout
    return: empty
    '''
    clear_one(base_dir, base_dir_geolayout, 'training')
    clear_one(base_dir, base_dir_geolayout, 'validation')
    clear_one(base_dir, base_dir_geolayout, 'testing')
    clear_norm(base_dir)
    data_split(base_dir)


if __name__ == "__main__":
    clear(base_dir, base_dir_geolayout)