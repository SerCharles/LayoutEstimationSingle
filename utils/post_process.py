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

def get_selected_plane_info(labels, plane_info_per_pixel):
    ''' 
    description: get the average plane infos
    parameter: labels, plane info per pixel
    return: average plane infos, unique labels
    '''
    _, H, W = plane_info_per_pixel.shape 
    unique_label = np.unique(labels) 
    average_plane_infos = np.zeros((len(unique_label), 4))
    for i in range(len(unique_label)):
        label = unique_label[i]
        if label == 0: 
            a_avg = 0.0 
            b_avg = 0.0 
            c_avg = 0.0 
            d_avg = 0.0 
        else: 
            mask = (label == labels)
            mask_sum = np.float(mask.sum())
            a = plane_info_per_pixel[0:1, :, :]
            b = plane_info_per_pixel[1:2, :, :]
            c = plane_info_per_pixel[2:3, :, :]
            d = plane_info_per_pixel[3:4, :, :]
            a_avg = np.float(a * mask).sum() / mask_sum
            b_avg = np.float(b * mask).sum() / mask_sum
            c_avg = np.float(c * mask).sum() / mask_sum
            d_avg = np.float(d * mask).sum() / mask_sum
        average_plane_infos[i][0] = a_avg
        average_plane_infos[i][1] = b_avg
        average_plane_infos[i][2] = c_avg
        average_plane_infos[i][3] = d_avg
    return average_plane_infos, unique_label

def get_label_per_pixel(average_plane_infos, unique_label, intrinsics, H, W):
    ''' 
    description: get the label of each pixel of all the pictures 
    parameter: the average plane infos of all planes, the unique labels, the intrinsics of the picture, height and width of the picture
    return: the labels of all pixels
    '''
    result_labels = np.zeros((1, H, W))
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    x0 = intrinsic[2][0]
    y0 = intrinsic[2][1]
    xx, yy = np.meshgrid(np.array([ii for ii in range(H)]), np.array([ii for ii in range(W)]))
    x_z = ((xx - x0) / fx)
    y_z = ((yy - y0) / fy)
    z_results = []

    print(x_z.shape)
    for i in range(len(unique_label)):
        label = unique_label[i]
        if label != 0:
            a = average_plane_infos[i][0]
            b = average_plane_infos[i][1]
            c = average_plane_infos[i][2]
            d = average_plane_infos[i][3]
            depth = -d / (x_z * a + y_z * b + c)
        else: 
            depth = -np.ones((H, W))

        current_max = 14530529
        positive_mask = (depth >= 0)
        negative_mask = ~ positive_mask
        depth_positive = depth * positive_mask
        depth_negative = negative_mask * current_max
        depth = depth_positive + depth_negative  

        z_results.append(depth.reshape(1, H, W))
    z_results = np.concatenate(z_results, axis = 0)
    min_index = np.argmin(z_results, axis = 0)
    return min_index

def post_process(device, seg_result, plane_info_per_pixel, intrinsics, threshold_ratio):
    ''' 
    description: the main function of post processing
    parameter: device, segmentation result, plane info result, intrinsics, threshold ratio
    return: the segmentation labels 
    '''
    N, _, H, W = seg_result.size()
    labels = []
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

        average_plane_infos, unique_label = get_selected_plane_info(total_labels, the_plane_info)
        print(unique_label)
        label_per_pixel = get_label_per_pixel(average_plane_infos, unique_label, the_intrinsic, H, W)

        labels.append(label_per_pixel.reshape(1, 1, H, W))
    
    labels = np.concatenate(labels, axis = 0)
    return labels





