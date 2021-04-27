''' 
The dataset used for loading and preprocessing Matterport3D-Layout dataset
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



    def __init__(self, base_dir, the_type):
        '''
        description: init the dataset
        parameter: the base dir of the dataset, the type(training, validation, testing)
        return:empty
        '''
        self.setTransform()

        self.base_dir = base_dir 
        self.type = the_type
        self.mat_path = os.path.join(base_dir, the_type, the_type + '.mat')
        self.depth_filenames = []
        self.image_filenames = []
        self.init_label_filenames = []
        self.layout_depth_filenames = []
        self.layout_seg_filenames = []
        self.points = []
        self.intrinsics = []
        self.faces = []
        self.params = []
        self.norms_x = [] 
        self.norms_y = [] 
        self.norms_z = [] 

        self.boundarys = [] 
        self.radiuss = [] 
    

        with h5py.File(self.mat_path, 'r') as f:
            data = f['data']
            
            images = data['image'][:]
            intrinsics_matrixs = data['intrinsics_matrix'][:]
            self.length = len(images)

            if not the_type == 'testing':
                depths = data['depth'][:]
                images = data['image'][:]
                init_labels = data['init_label'][:]
                intrinsics_matrixs = data['intrinsics_matrix'][:]
                layout_depths = data['layout_depth'][:]
                layout_segs = data['layout_seg'][:]
                models = data['model'][:]
                points = data['point'][:]

            
            for i in range(self.length):
                image_id = f[images[i][0]]
                the_string = ''
                for item in image_id: 
                    the_string += chr(item[0])
                self.image_filenames.append(the_string)

                the_intrinsic = f[intrinsics_matrixs[i][0]]
                the_intrinsic = np.array(the_intrinsic)
                self.intrinsics.append(the_intrinsic)


                if not the_type == 'testing':
    
                    depth_id = f[depths[i][0]]
                    init_label_id = f[init_labels[i][0]]
                    layout_depth_id = f[layout_depths[i][0]]
                    layout_seg_id = f[layout_segs[i][0]]
                    the_model = f[models[i][0]]
                    the_point = f[points[i][0]]

                    the_point = np.array(the_point)
                    the_model_faces = the_model['face']
                    the_model_params = the_model['params']


                    faces = []
                    params = []
                    for j in range(len(the_model_faces)):
                        face = f[the_model_faces[j][0]][0][0]
                        param = f[the_model_params[j][0]][0][0]
                        faces.append(face)  
                        params.append(param)

                    self.faces.append(faces) 
                    self.params.append(params)

                
                    self.points.append(the_point)

                    the_string = ''
                    for item in depth_id: 
                        the_string += chr(item[0])
                    self.depth_filenames.append(the_string)


                    the_string = ''
                    for item in init_label_id: 
                        the_string += chr(item[0])
                    self.init_label_filenames.append(the_string)

                    the_string = ''
                    for item in layout_depth_id: 
                        the_string += chr(item[0])
                    self.layout_depth_filenames.append(the_string)

                    the_string = ''
                    for item in layout_seg_id: 
                        the_string += chr(item[0])
                    self.layout_seg_filenames.append(the_string)

    def setTransform(self):
        '''
        description: set the transformation of the input picture
        parameter: empty
        return: empty
        '''
        self.transform = transforms.Compose([transforms.Resize([240, 320]), transforms.ToTensor()])


    def __getitem__(self, i):
        '''
        description: get one part of the item
        parameter: the index 
        return: the data
        '''

        image_name = os.path.join(self.base_dir, self.type, 'image', self.image_filenames[i])
        image = self.load_image(image_name)
        x_size = image.size[1]
        y_size = image.size[0]
        xx, yy = np.meshgrid(np.array([ii for ii in range(x_size)]), np.array([ii for ii in range(y_size)]))
        intrinsic = self.intrinsics[i]
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        x0 = intrinsic[2][0]
        y0 = intrinsic[2][1]
        mesh_x = Image.fromarray((xx - x0) / fx)
        mesh_y = Image.fromarray((yy - y0) / fy)


        if not self.type == 'testing':
            base_name = self.depth_filenames[i][:-4]
            depth_name = os.path.join(self.base_dir, self.type, 'depth', self.depth_filenames[i])
            init_label_name = os.path.join(self.base_dir, self.type, 'init_label', self.init_label_filenames[i])
            layout_depth_name = os.path.join(self.base_dir, self.type, 'layout_depth', self.layout_depth_filenames[i])
            layout_seg_name = os.path.join(self.base_dir, self.type, 'layout_seg', self.layout_seg_filenames[i])
            nx_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_nx.png')
            ny_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_ny.png')
            nz_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_nz.png')
            boundary_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_boundary.png')
            radius_name = os.path.join(self.base_dir, self.type, 'normal', base_name + '_radius.png')

            #layout_depth = self.load_depth(layout_depth_name)
            #layout_seg = self.load_depth(layout_seg_name)
            nx = self.load_image(nx_name)
            ny = self.load_image(ny_name)
            nz = self.load_image(nz_name)


                
        if self.type == 'testing':
            image = self.transform(image)
            intrinsic = torch.tensor(self.intrinsics[i], dtype = torch.float)
            mesh_x = self.transform(mesh_x)
            mesh_y = self.transform(mesh_y)
            return image, intrinsic, mesh_x, mesh_y
        else:
            image = self.transform(image)
            #layout_depth = self.transform(layout_depth) / 4000.0
            #layout_seg = self.transform(layout_seg)
            mesh_x = self.transform(mesh_x)
            mesh_y = self.transform(mesh_y)

            nx = self.transform(nx).float()
            ny = self.transform(ny).float()
            nz = self.transform(nz).float()
            normal = torch.cat((nx, ny, nz), dim = 0).float()
            normal_length = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)

            normal = normal / (normal_length + 1e-8)
            intrinsic = torch.tensor(self.intrinsics[i], dtype = torch.float)
            return image,  normal, intrinsic, mesh_x, mesh_y
            #return image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y


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
    a = MatterPortDataSet('E:\\dataset\\geolayout', 'training')
    i = 0
    print('length:', a.__len__())
    #image, layout_depth, layout_seg, normal, intrinsic, mesh_x, mesh_y = a.__getitem__(i)
    image, normal, intrinsic, mesh_x, mesh_y = a.__getitem__(i)
    print('filename:', a.layout_depth_filenames[i])
    print('filename:', a.layout_depth_filenames[i + 1])
    print('image:', image, image.size())
    #print('layout_depth:', layout_depth, layout_depth.size())
    #print('layout_seg:', layout_seg, layout_seg.size())
    print('normal', normal, normal.size())
    print('intrinsic:', intrinsic, intrinsic.shape)
    print('mesh_x', mesh_x, mesh_x.size())
    print('mesh_y', mesh_y, mesh_y.size())
    print(normal.mean())
    
    b = MatterPortDataSet('E:\\dataset\\geolayout', 'testing')
    j = 10
    print('length:', b.__len__())
    image, intrinsic, mesh_x, mesh_y = b.__getitem__(j)
    print('image:', image, image.size())
    print('intrinsic:', intrinsic, intrinsic.shape)
    print('mesh_x', mesh_x, mesh_x.size())
    print('mesh_y', mesh_y, mesh_y.size())
#data_test()