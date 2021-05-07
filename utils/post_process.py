''' 
The postprocess code. It includes mean-shift clustering and iterations to further improve the result
'''

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms
import pandas as pd
import PIL
from PIL import Image
from utils.utils import *
from utils.metrics import *


def mean_shift_clustering(the_parameter_image):
    ''' 
    description: mean shift clustering algorithm
    parameter: the parameters of the image 
    return: the labels of all the pixels
    '''
    bandwidth = estimate_bandwidth(the_parameter_image, quantile = 0.2, n_samples = 1000)
    #print(bandwidth)
    ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
    ms.fit(the_parameter_image)
    labels = ms.labels_
    return labels

def get_useful_labels(labels, threshold_ratio, total_size):
    ''' 
    description: get useful labels and clear the labels
    parameter: the labels of all items, the threshold ratio, the total size of the picture
    return: useful labels, cleared labels(set useless labels to 0)
    '''
    threshold = threshold_ratio * total_size 
    unique_labels = np.unique(labels)
    useful_labels = []
    for label in unique_labels:
        label_mask = (labels == label)
        label_num = (labels == label).sum()
        if label_num > threshold_ratio:
            useful_labels.append(label)
        else: 
            labels[label_mask] = 0
            
    useful_labels = np.array(useful_labels)
    return useful_labels, labels

def post_process(device, seg_result, plane_info_per_pixel, intrinsics, threshold_ratio):
    ''' 
    description: the main function of post processing
    parameter: device, segmentation result, plane info result, intrinsics, threshold ratio
    return: the segmentation and average plane info 
    '''
    N, _, H, W = seg_result.size()
    for i in range(N):
        the_seg = seg_result[i]
        the_plane_info = plane_info_per_pixel[i]
        the_intrinsic = intrinsics[i]

        the_index_list = torch.linspace(1, H * W, steps = H * W, dtype = np.int).view(1, H, W)
        if device:
            the_index_list = the_index_list.cuda()

        selected_plane_info_a = torch.masked_select(the_plane_info[0:1, :, :], the_seg).unsqueeze(0)
        selected_plane_info_b = torch.masked_select(the_plane_info[1:2, :, :], the_seg).unsqueeze(0)
        selected_plane_info_c = torch.masked_select(the_plane_info[2:3, :, :], the_seg).unsqueeze(0)
        selected_plane_info_d = torch.masked_select(the_plane_info[3:4, :, :], the_seg).unsqueeze(0)
        selected_plane_info = torch.cat((selected_plane_info_a, selected_plane_info_b, selected_plane_info_c, selected_plane_info_d), dim = 0)
        selected_index = torch.masked_select(the_index_list, the_seg)

        the_seg = the_seg.cpu().numpy() 
        the_plane_info = the_plane_info.cpu().numpy()
        the_intrinsic = the_intrinsic.cpu().numpy()
        selected_plane_info = selected_plane_info.cpu().numpy()
        selected_index = selected_index.cpu().numpy()

        plane_info_pandas = pd.DataFrame(selected_plane_info.T, columns = list('abcd'))
        selected_labels = mean_shift_clustering(plane_info_pandas) + 1
        useful_labels, cleared_labels = get_useful_labels(selected_labels, threshold_ratio, H * W)

        total_labels = np.zeros((H * W), dtype = int)
        total_labels[selected_index] = cleared_labels
        total_labels = total_labels.reshape((1, H, W))










def main():
    args = init_args()
    device, dataset_training, dataset_validation, model, optimizer, start_epoch = init_model(args)


