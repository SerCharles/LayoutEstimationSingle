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
    bandwidth = estimate_bandwidth(the_parameter_image, quantile = 0.1, n_samples = 1000)
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
        if label_num > threshold:
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
            
            a_avg = np.float((a * mask).sum()) / mask_sum
            b_avg = np.float((b * mask).sum()) / mask_sum
            c_avg = np.float((c * mask).sum()) / mask_sum
            d_avg = np.float((d * mask).sum()) / mask_sum

        average_plane_infos[i][0] = a_avg
        average_plane_infos[i][1] = b_avg
        average_plane_infos[i][2] = c_avg
        average_plane_infos[i][3] = d_avg
    return average_plane_infos, unique_label

def get_label_per_pixel(average_plane_infos, unique_label, intrinsics, H, W):
    ''' 
    description: get the label of each pixel of all the pictures 
    parameter: the average plane infos of all planes, the unique labels, the intrinsics of the picture, height and width of the picture
    return: the labels of all pixels, the depth of all pixels
    '''
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    x0 = intrinsics[2][0]
    y0 = intrinsics[2][1]
    xx, yy = np.meshgrid(np.array([ii for ii in range(W)]), np.array([ii for ii in range(H)]))

    yy = H - 1 - yy
    x_z = ((xx - x0) / fx)
    y_z = ((yy - y0) / fy)



    z_results = []
    for i in range(len(unique_label)):
        label = unique_label[i]
        if label != 0:
            a = average_plane_infos[i][0]
            b = average_plane_infos[i][1]
            c = average_plane_infos[i][2]
            d = average_plane_infos[i][3]
            
            divided = x_z * a + y_z * b + c
            depth_inverse = -divided / d
        else: 
            depth_inverse = -np.ones((H, W))
        z_results.append(depth_inverse.reshape(1, H, W))
    z_results = np.concatenate(z_results, axis = 0)
    min_index = np.argmax(z_results, axis = 0)
    min_depth = 1 / np.max(z_results, axis = 0)
    return min_index, min_depth

def post_process(device, seg_result, plane_info_per_pixel, intrinsics, threshold_ratio):
    ''' 
    description: the main function of post processing
    parameter: device, segmentation result, plane info result, intrinsics, threshold ratio
    return: the segmentation labels 
    '''
    N, _, H, W = seg_result.size()
    labels = []
    depths = []
    labels_raw = []
    for i in range(N):
        the_seg = seg_result[i]
        the_plane_info = plane_info_per_pixel[i]
        the_intrinsic = intrinsics[i]

        the_index_list = torch.linspace(0, H * W - 1, steps = H * W, dtype = np.int).view(1, H, W)
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
        labels_raw.append(total_labels.reshape(1, 1, H, W))

        average_plane_infos, unique_label = get_selected_plane_info(total_labels, the_plane_info)
        label_per_pixel, depth_per_pixel = get_label_per_pixel(average_plane_infos, unique_label, the_intrinsic, H, W)

        labels.append(label_per_pixel.reshape(1, 1, H, W))
        depths.append(depth_per_pixel.reshape(1, 1, H, W))
    
    labels_raw = np.concatenate(labels_raw, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    depths = np.concatenate(depths, axis = 0)
    return labels_raw, labels, depths


def get_colors(seg):
    ''' 
    description: get the pixel color of the seg result
    parameter: labels
    return: colors
    '''
    seg_r = seg // 9
    seg_g = (seg - seg_r * 9) // 3
    seg_b = seg % 3
    seg_colors_r = (seg_r * 127).reshape((seg.shape[1], seg.shape[2]))
    seg_colors_b = (seg_g * 127).reshape((seg.shape[1], seg.shape[2]))
    seg_colors_g = (seg_b * 127).reshape((seg.shape[1], seg.shape[2]))
    seg_colors = np.stack((seg_colors_r, seg_colors_g, seg_colors_b), axis = 2)
    return seg_colors

def save_results(save_base, base_names, init_labels, final_labels_raw, final_labels, layout_seg):
    ''' 
    description: save the plane results
    parameter: save_base, the file base names of the batch, the init labels, final segmentation and final plane infos
    return: empty
    '''
    save_dir = os.path.join(save_base, 'seg')
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)

    batch_size = len(base_names)
    for i in range(batch_size):
        base_name = base_names[i]
        init_raw = init_labels[i]
        seg_raw = final_labels_raw[i]
        seg = final_labels[i]
        seg_gt = layout_seg[i]

        init_colors_raw = get_colors(init_raw)
        seg_colors_raw = get_colors(seg_raw)
        seg_colors = get_colors(seg)
        seg_colors_gt = get_colors(seg_gt)

        init_image_raw = Image.fromarray(np.uint8(init_colors_raw)).convert('RGB')
        seg_image_raw = Image.fromarray(np.uint8(seg_colors_raw)).convert('RGB')
        seg_image = Image.fromarray(np.uint8(seg_colors)).convert('RGB')
        seg_image_gt = Image.fromarray(np.uint8(seg_colors_gt)).convert('RGB')

        init_name_raw = os.path.join(save_dir, base_name + '_seg_raw_gt.png')
        seg_name_raw = os.path.join(save_dir, base_name + '_seg_raw.png')
        seg_name = os.path.join(save_dir, base_name + '_seg.png')
        seg_name_gt = os.path.join(save_dir, base_name + '_seg_gt.png')

        init_image_raw.save(init_name_raw)
        seg_image_raw.save(seg_name_raw)
        seg_image.save(seg_name)
        seg_image_gt.save(seg_name_gt)
    