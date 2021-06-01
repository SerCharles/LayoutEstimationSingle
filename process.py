''' 
After training the network, use this to conduct the clustering, iterative improvement and visualization
'''


import numpy as np
from pandas.core import base
from sklearn.cluster import MeanShift, estimate_bandwidth
import torch
from math import *
from torchvision import transforms
import pandas as pd
import PIL
from PIL import Image
from utils.utils import *
from utils.metrics import *
from train_valid_utils import *
from utils.post_process import *
from data.dataset import *
from utils.average_meters import *

def main():

    args = init_args()
    device, dataset_validation, valid_loader, model = init_valid_model(args)
    model.eval()
    all_base_names = dataset_validation.get_valid_filenames()
    flag = 0
    average_meter = AverageMeterValid()
    for i, (image, depth, layout_depth, layout_seg, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(valid_loader):
        start = time.time()
        
        if device:
            image = image.cuda()
            depth = depth.cuda()
            layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        with torch.no_grad():
            N, C, H, W = image.size()
            batch_size = image.size(0)
            base_names = all_base_names[flag : flag + batch_size]
            flag += batch_size

            useful_mask = depth > 0
            init_label = init_label * useful_mask
            mask_gt = torch.ne(init_label, 0)

            
            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)  
            seg_result = output[:, 0:2, :, :]
            norm_result = output[:, 2:5, :, :]
            depth_result = output[:, 5:, :, :]

            my_seg = get_seg(seg_result)
            norm_result = normalize(norm_result, args.epsilon)
            my_depth = get_predicted_depth(depth_result, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            '''
            my_seg = mask_gt
            norm_result = normal 
            my_depth = layout_depth
            '''

            plane_info_per_pixel = get_plane_info_per_pixel(device, norm_result, my_depth, intrinsic)
            
            my_seg_raw, my_seg, my_depth = post_process(device, my_seg, plane_info_per_pixel, intrinsic, args.threshold)
            layout_seg = layout_seg.cpu().numpy()
            accuracy = seg_metrics(my_seg, layout_seg)
            my_depth = torch.from_numpy(my_depth)
            my_mask = torch.ones((N, 1, H, W))
            my_mask = torch.eq(my_mask, 1)
            if device:
                my_depth = my_depth.cuda()
                my_mask = my_mask.cuda()
            rms, rel, rlog10, delta_1, delta_2, delta_3 = depth_metrics(my_depth, layout_depth, my_mask)
            average_meter.add_batch(batch_size, accuracy, rms, rel, rlog10, delta_1, delta_2, delta_3)
            end = time.time()
            the_time = end - start
            result_string = get_result_string_valid(i + 1, len(valid_loader), the_time, accuracy, rms, rel, rlog10, delta_1, delta_2, delta_3)
            print(result_string)
            save_base = os.path.join(args.save_dir, args.cur_name)

            init_label_np = init_label.cpu().numpy()
            save_results(save_base, base_names, init_label_np, my_seg_raw, my_seg, layout_seg)
    
    avg_acc, avg_rms, avg_rel, avg_rlog10, avg_delta_1, avg_delta_2, avg_delta_3 = average_meter.get_average()
    result_string = get_result_string_valid_acc(avg_acc, avg_rms, avg_rel, avg_rlog10, avg_delta_1, avg_delta_2, avg_delta_3)
    print(result_string)


if __name__ == "__main__":
    main()
