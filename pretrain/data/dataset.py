''' 
The dataset used for pretraining the dataset
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


class MatterPortDataSet(Dataset):
    def load_image(self, file_name):
        '''
        description: used in loading RGB image
        parameter: filename
        return: image info of PIL
        '''
        fp = open(file_name, 'rb')
        pic = Image.open(fp)
        pic_array = np.array(pic)
        fp.close()
        pic = Image.fromarray(pic_array)
        return pic
        
    def load_depth(self, file_name):
        '''
        description: used in loading depth/norm/segmentation images
        parameter: filename
        return: info of PIL
        '''
        fp = open(file_name, 'rb')
        pic = Image.open(fp)
        pic_array = np.array(pic)
        fp.close()
        pic = Image.fromarray(pic_array)
        pic = pic.convert("I")
        return pic

    def load_intrinsic(self, file_name):
        '''
        description: used in loading intrinsics
        parameter: filename
        return: intrinsic numpy array
        '''
        f = open(file_name, 'r')
        words = f.read().split()
        f.close()
        fx = float(words[0])
        fy = float(words[1])
        x0 = float(words[2])
        y0 = float(words[3])
        intrinsic = np.zeros((3, 3), dtype = float)
        intrinsic[0][0] = fx 
        intrinsic[1][1] = fy 
        intrinsic[2][0] = x0
        intrinsic[2][1] = y0 
        intrinsic[2][2] = 1.0
        return intrinsic

    def transform_intrinsics(self, new_size, old_size, intrinsic):
        '''
        description: transform the intrinsics
        parameter: the size of output picture, the size of input picture, original intrinsic
        return: new intrinsic
        '''
        new_H = new_size[0]
        new_W = new_size[1]
        old_H = old_size[1]
        old_W = old_size[0]
        intrinsic[0][0] = intrinsic[0][0] / old_W * new_W
        intrinsic[2][0] = intrinsic[2][0] / old_W * new_W
        intrinsic[1][1]= intrinsic[1][1] / old_H * new_H
        intrinsic[2][1] = intrinsic[2][1] / old_H * new_H
        return intrinsic

    def __init__(self, base_dir, the_type):
        '''
        description: init the dataset
        parameter: the base dir of the dataset, the type(training, validation, testing)
        return:empty
        '''
        self.size = [240, 320]
        self.setTransform(self.size)
        self.base_dir = base_dir 
        self.type = the_type

        self.image_filenames = []
        self.depth_filenames = []
        self.nx_filenames = []
        self.ny_filenames = []
        self.nz_filenames = []
        self.intrinsic_filenames = []
        self.seg_filenames = []
        
        f = open(os.path.join(self.base_dir, self.type + '.txt'), 'r') 
        filenames = f.read().split('\n')
        self.length = 0

        for filename in filenames:
            if len(filename) <= 0:
                continue
            self.length += 1
            base_name = filename[:-9]
            group_name = filename[-7]
            ins_name = filename[-5]
            image_name = base_name + '_i' + group_name + '_' + ins_name + '.jpg'
            depth_name = base_name + '_d' + group_name + '_' + ins_name + '.png'
            nx_name = base_name + '_d' + group_name + '_' + ins_name + '_nx.png'
            ny_name = base_name + '_d' + group_name + '_' + ins_name + '_ny.png'
            nz_name = base_name + '_d' + group_name + '_' + ins_name + '_nz.png'
            intrinsic_name = base_name + '_pose_' + group_name + '_' + ins_name + '.txt'
            seg_name = base_name + '_s' + group_name + '_' + ins_name + '.png'
            self.image_filenames.append(image_name)
            self.depth_filenames.append(depth_name)
            self.nx_filenames.append(nx_name)
            self.ny_filenames.append(ny_name)
            self.nz_filenames.append(nz_name)
            self.intrinsic_filenames.append(intrinsic_name)
            self.seg_filenames.append(seg_name)
            


    def setTransform(self, size):
        '''
        description: set the transformation of the input picture
        parameter: size
        return: empty
        '''
        self.transform_float = transforms.Compose([transforms.Resize(size, interpolation = Image.BILINEAR), transforms.ToTensor()])
        self.transform_int = transforms.Compose([transforms.Resize(size, interpolation = Image.NEAREST), transforms.ToTensor()])


    def __getitem__(self, i):
        '''
        description: get one part of the item
        parameter: the index 
        return: the data
        '''
        image_name = os.path.join(self.base_dir, 'image', self.image_filenames[i])
        depth_name = os.path.join(self.base_dir, 'depth', self.depth_filenames[i])
        nx_name = os.path.join(self.base_dir, 'norm', self.nx_filenames[i])
        ny_name = os.path.join(self.base_dir, 'norm', self.ny_filenames[i])
        nz_name = os.path.join(self.base_dir, 'norm', self.nz_filenames[i])
        intrinsic_name = os.path.join(self.base_dir, 'intrinsic', self.intrinsic_filenames[i])
        seg_name = os.path.join(self.base_dir, 'seg', self.seg_filenames[i])

        image = self.load_image(image_name)
        depth = self.load_depth(depth_name)
        nx = self.load_depth(nx_name)
        ny = self.load_depth(ny_name)
        nz = self.load_depth(nz_name)
        intrinsic = self.load_intrinsic(intrinsic_name)
        seg = self.load_depth(seg_name)

        x_size = image.size[1]
        y_size = image.size[0]
        xx, yy = np.meshgrid(np.array([ii for ii in range(x_size)]), np.array([ii for ii in range(y_size)]))
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        x0 = intrinsic[2][0]
        y0 = intrinsic[2][1]
        mesh_x = Image.fromarray((xx - x0) / fx)
        mesh_y = Image.fromarray((yy - y0) / fy)

        old_size = image.size
        intrinsic = self.transform_intrinsics(self.size, old_size, intrinsic)
        intrinsic = torch.tensor(intrinsic, dtype = torch.float)
        image = self.transform_float(image)
        seg = self.transform_int(seg)

        depth = self.transform_int(depth).float() / 4000.0

        nx = self.transform_int(nx).float()
        ny = self.transform_int(ny).float()
        nz = self.transform_int(nz).float()   
        nx = nx / 32768.0 - 1
        ny = ny / 32768.0 - 1
        nz = -(nz / 32768.0 - 1)
         
        mesh_x = self.transform_float(mesh_x)
        mesh_y = self.transform_float(mesh_y)

        normal = torch.cat((nx, ny, nz), dim = 0).float()
        normal_length = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        normal = normal / (normal_length + 1e-8)
            
        intrinsic = torch.tensor(intrinsic, dtype = torch.float)
        return image, seg, depth, normal, intrinsic, mesh_x, mesh_y


    def get_valid_filenames(self):
        '''
        description: get the file names of all valid data
        parameter: empty
        return: the filenames
        '''
        if not self.type == 'validation':
            return []
        result = []
        for i in range(self.length):
            base_name = self.depth_filenames[i][:-4]
            result.append(base_name)
        return result


    def __len__(self):
        '''
        description: get the length of the dataset
        parameter: empty
        return: the length
        '''
        return self.length


#unit test code
def data_test():
    a = MatterPortDataSet('E:\\dataset\\geolayout_pretrain', 'training')
    i = 0
    print('length:', a.__len__())
    image, seg, depth, normal, intrinsic, mesh_x, mesh_y = a.__getitem__(i)

    print('filename:', a.layout_depth_filenames[i])
    print('filename:', a.layout_depth_filenames[i + 1])
    print('image:', image, image.size())
    print('seg:', seg, seg.size())
    print('depth:', depth, depth.size())
    print('normal', normal, normal.size())
    print('intrinsic:', intrinsic, intrinsic.shape)
    print('mesh_x', mesh_x, mesh_x.size())
    print('mesh_y', mesh_y, mesh_y.size())
    print(torch.sum(torch.le(depth, 0)))
    
#data_test()