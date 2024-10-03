
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split
import os
from torchvision import datasets, transforms
from einops import rearrange
from torchvision import transforms as T, utils
import sys
import numpy as np
import csv
from PIL import Image
import pandas
import scipy
import glob
import math
import psutil
from multiprocessing import Pool
from occluder.datasets.pointcloud import PointCloud



def normalize_point_cloud(point_cloud, 
                          min_values=torch.tensor([[0.553384335154827, 0.553384335154827, 0.08]]), 
                          max_values=torch.tensor([[1.0008524590163934, 0.7369326047358835, 0.386]])):
    return (point_cloud - min_values) / (max_values - min_values)

def denormalize_point_cloud(normalized_pc, 
                            min_values=torch.tensor([[0.553384335154827, 0.553384335154827, 0.08]]), 
                            max_values=torch.tensor([[1.0008524590163934, 0.7369326047358835, 0.386]])):
    return normalized_pc * (max_values - min_values) + min_values


def normalize(s):
    return s/s.max()


class CoreDataset(Dataset):

    def __init__(self, path_to_data, root_to_image=None, start_data=0, end_data= -1,
                image_size=64, seq_length=1024,
                return_measurements=True,
                return_normalized=True, return_images=True, augument=True,
                image_channel=3,):


        super(CoreDataset, self).__init__()

        self.path_to_data=path_to_data
        if isinstance(path_to_data, list):
            # print(True)
            self.path = []
            for p in path_to_data:
                self.path += glob.glob(f"{p}/**.mat")[start_data:end_data]
              
        else:
            self.path = glob.glob(f"{path_to_data}/**.mat")[start_data:end_data]
        self.start=start_data
        self.end = end_data
        self.image_size=image_size
        self.seq_length=seq_length
        np.random.seed(2023)
        np.random.shuffle(self.path)
        self.norm = return_normalized
        self.return_measurements = return_measurements
        self.return_images = return_images
        self.augument = augument
        self.image_channel=image_channel
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor()])
        self.root_to_image = root_to_image
        self.time = 0



    def __len__(self):
        return len(self.path)



    def __getitem__(self, idx):


        if idx == 0 and self.time == 100:
            self.path = glob.glob(f"{self.path_to_data}/**.mat")[self.start:self.end]
            self.time=0

        elif idx == 0:
            self.time+=1

        try:
        
            f=scipy.io.loadmat(self.path[idx])
        except:
            f=scipy.io.loadmat(self.path[idx-1])
      

        data = {}

        pc = torch.from_numpy(f["pointcloud"])
        
        
        if self.root_to_image is not None:
            # sc = f["scene"]
            image = Image.open(os.path.join(self.root_to_image, f["scene"][0]))
            image = image.convert("RGB")
            image = self.transform(image)
                    
            data["image"] = image

    
        if self.return_measurements:
            # meas = normalize(torch.load(self.measurements[id], map_location="cpu"))
            m = torch.from_numpy(normalize(f["measurement"]+1e-12))


            r = np.random.choice( [0, 1], p = [0.8, 0.2] )

            if r == 1 and self.augument:
                pc = rotate_pointcloud(pc, 180, 1)
                if self.root_to_image is not None:
                    image = torch.flip(image, (1,))
            else:
                m = torch.flip(m, (0,))
                

            r = np.random.choice( [0, 1], p = [0.9, 0.1] )
            if r==1:
                m = torch.flip(m, (-1,))
                if self.root_to_image is not None:
                    image = torch.flip(image, (0,))
                
            if self.image_channel==1:

                data = {"measurements": m.mean(-1, keepdim=True)}
                
            else:
                data = {"measurements": m}

        mini = pc.min()
        maxi = pc.max()
        scale = maxi - mini     

        pc = ((pc-mini)/scale)-0.5
      
        pc = PointCloud(coords=pc.numpy(), channels={})
        pc = pc.farthest_point_sample(self.seq_length)
        data['pointclouds'] = torch.from_numpy(pc.coords)
        return data



def rotate_pointcloud(pointcloud, degrees, axis):
    """
    Rotate a point cloud around a specific axis by a given number of degrees using PyTorch.
    
    Args:
        pointcloud (torch.Tensor): Input point cloud of shape (N, 3), where N is the number of points.
        degrees (float): Number of degrees to rotate the point cloud.
        axis (int): Axis to rotate around (0 for X-axis, 1 for Y-axis, 2 for Z-axis).
    
    Returns:
        torch.Tensor: Rotated point cloud.
    """
    radians = math.radians(degrees)
    rotation_matrix = torch.eye(3)
    
    cos_theta = torch.cos(torch.tensor(radians))
    sin_theta = torch.sin(torch.tensor(radians))
    
    if axis == 0:
        rotation_matrix[1, 1] = cos_theta
        rotation_matrix[1, 2] = -sin_theta
        rotation_matrix[2, 1] = sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 1:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 2] = sin_theta
        rotation_matrix[2, 0] = -sin_theta
        rotation_matrix[2, 2] = cos_theta
    elif axis == 2:
        rotation_matrix[0, 0] = cos_theta
        rotation_matrix[0, 1] = -sin_theta
        rotation_matrix[1, 0] = sin_theta
        rotation_matrix[1, 1] = cos_theta
    else:
        raise ValueError("Invalid axis. Must be 0, 1, or 2.")
    
    rotated_pointcloud = torch.matmul(pointcloud, rotation_matrix)
    return rotated_pointcloud


def get_data_iterator(iterable):
    """
    
    Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
        
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()



if __name__=="__main__":

    train_dset = CoreDataset(
            path_to_data="/home/predstan/research/SSD/Objects3")

    print(len(train_dset))
    val_loader = DataLoader(train_dset, batch_size=6, num_workers=4)
    i=0
    for b in val_loader:
        print(b["pointclouds"].max())
        print(i+1)
        # print(b["categories"])
        # print(b["measurements"].shape)
        i+=1
        print("\n")

    
    # # print(len(train_dset.measurements))
    # # print(len(train_dset.point_clouds))
    # # print(train_dset[10].keys())
    # print(train_dset[-1]["categories"])
    # print(train_dset[-1][self.point])
    # # print(len(train_dset.cate_indexes))
