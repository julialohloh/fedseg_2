"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
import h5py
import random
from random import randint
from random import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset




class RandomTransforms:
    def __init__(self, dims, prob1=1, prob2=0.5, border = [8,8,8,8]):
        # tblr
        self.height = dims['height']
        self.width = dims['width']
        self.layers = dims['layers']
        self.prob1 = prob1
        self.prob2 = prob2
        self.b = border

    def __call__(self, image, target, weight):

        if random.random()<self.prob1:
            pad_image = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3], 1))
            pad_target = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3], self.layers))
            pad_weight = np.zeros((self.height+self.b[0]+self.b[1], self.width+self.b[2]+self.b[3]))
            
            # mirror image
            pad_image[self.b[0]:-self.b[1],self.b[2]:-self.b[3],:] = image 
            pad_image[:self.b[0],self.b[2]:-self.b[3]] = image[self.b[0]-1:None:-1]# fill up top border
            pad_image[-self.b[1]:,self.b[2]:-self.b[3]] = image[:-self.b[1]-1:-1] # fill up bottom border
            pad_image[:,self.b[2]-1:None:-1] = pad_image[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_image[:,-self.b[3]:] = pad_image[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            # mirror target
            pad_target[self.b[0]:-self.b[1],self.b[2]:-self.b[3],:] = target 
            pad_target[:self.b[0],self.b[2]:-self.b[3]] = target[self.b[0]-1:None:-1]# fill up top border
            pad_target[-self.b[1]:,self.b[2]:-self.b[3]] = target[:-self.b[1]-1:-1] # fill up bottom border
            pad_target[:,self.b[2]-1:None:-1] = pad_target[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_target[:,-self.b[3]:] = pad_target[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            # mirror weight
            pad_weight[self.b[0]:-self.b[1],self.b[2]:-self.b[3]] = weight 
            pad_weight[:self.b[0],self.b[2]:-self.b[3]] = weight[self.b[0]-1:None:-1]# fill up top border
            pad_weight[-self.b[1]:,self.b[2]:-self.b[3]] = weight[:-self.b[1]-1:-1] # fill up bottom border
            pad_weight[:,self.b[2]-1:None:-1] = pad_weight[:,self.b[2]:2*self.b[2]] # fill up left border
            pad_weight[:,-self.b[3]:] = pad_weight[:,-self.b[3]-1:-2*self.b[3]-1:-1] # fill up right border

            loc = [randint(0,16-1), randint(0,16-1)]
            image = pad_image[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
            target = pad_target[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
            weight = pad_weight[loc[0]:loc[0]+self.height, loc[1]:loc[1]+self.width]
        
        if random.random() < self.prob2:
            '''
            flipping
            '''
            
            image = np.flip(image,1)
            target = np.flip(target,1)
            weight = np.flip(weight,1)
        if random.random() < self.prob2:
            '''
            flipping
            '''
            
            image = np.flip(image,1)
            target = np.flip(target,1)
            weight = np.flip(weight,1)
        return image, target, weight
    
transforms2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])


    
class ImdbData(Dataset):
    
    def __init__(self, config, X, y, W, dims, transform=None):
        self.X = X
        self.y = y
        self.w = W
        self.height = dims['height']
        self.width = dims['width']
        self.layers = dims['layers']
        self.transform = transform
        self.transform2 = transforms2

    def __getitem__(self, index):
        img = np.transpose(self.X[index], (1,2,0)) 
        label = np.transpose(self.y[index],(1,2,0))
        weight = self.w[index]
        if self.transform is not None:
            img, label, weight = self.transform(img, label, weight)

        img = torch.from_numpy(img.copy()).float().permute(2,0,1)
        label = torch.from_numpy(label.copy()).long().permute(2,0,1)
        weight = torch.from_numpy(weight.copy()).float()
    
        
        if self.transform2 is not None:
            img = self.transform2(img)
    
        return img, label, weight


    def __len__(self):
        return len(self.X)


def get_imdb_data(data_dir):

    # Load DATA
    
    with h5py.File(os.path.join(data_dir,'training_intermediate.hdf5'),'r') as hf: 
        train_images=hf['data'][()]
        train_labels=hf['lmap'][()]
        train_wmaps=hf['wmap'][()]

    with h5py.File(os.path.join(data_dir,'val_intermediate.hdf5'),'r') as hf: 
        val_images=hf['data'][()]
        val_labels=hf['lmap'][()]
        val_wmaps=hf['wmap'][()]
        

    return train_images, train_labels, train_wmaps, val_images, val_labels, val_wmaps
