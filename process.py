''' 
After training the network, use this to conduct the clustering, iterative improvement and visualization
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
from train_valid_utils import *
from utils.post_process import *
from data.dataset import *


def main():

    args = init_args()
    device, dataset_validation, model = init_valid_model(args)
    model.eval()

    for i, (image, layout_depth, layout_seg, init_label, normal, intrinsic, mesh_x, mesh_y) in enumerate(dataset_validation):
        if device:
            image = image.cuda()
            layout_depth = layout_depth.cuda()
            init_label = init_label.cuda()
            normal = normal.cuda()
            intrinsic = intrinsic.cuda()
            mesh_x = mesh_x.cuda() 
            mesh_y = mesh_y.cuda()
        with torch.no_grad():
            batch_size = image.size(0)
            mask_gt = torch.ne(init_label, 0)

            the_input = torch.cat((image, mesh_x, mesh_y), dim = 1)
            output = model(the_input)  
            seg_result = output[:, 0:2, :, :]
            norm_result = output[:, 2:5, :, :]
            depth_result = output[:, 5:, :, :]

            my_seg = get_seg(seg_result)
            norm_result = normalize(norm_result, args.epsilon)
            my_depth = get_predicted_depth(depth_result, args.ordinal_beta, args.ordinal_gamma, args.discretization)
            plane_info_per_pixel = get_plane_info_per_pixel(device, norm_result, my_depth, intrinsic)

            my_seg = post_process(device, my_seg, plane_info_per_pixel, intrinsic, args.threshold)
            layout_seg = layout_seg.cpu().numpy()
            accuracy = seg_metrics(my_seg, layout_seg)
            print(accuracy)



if __name__ == "__main__":
    main()