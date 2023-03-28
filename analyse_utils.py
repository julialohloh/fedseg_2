#!/usr/bin/env python

####################
# Required Modules #
####################

#Generic Built in

import os
import shutil
from datetime import timedelta

#Libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
#Custom
# from early_stop import EarlyStopping
from losses import CombinedLoss
from models.relaynet.relay_net import ReLayNet

##################
# Configurations #
##################



#############
# Functions #
#############

def calc_dice(truth:torch.tensor,pred:torch.tensor)->float:
    """
    Calculates binary dice for each class and then returns the average dice across all classes

    Args:
        truth (torch.tensor): Ground truth
        pred (torch.tensor): Predictions from model

    Returns:
        dices: Dictionary containing the dice score of each class, and the average dice across all classes
    """
    pred_cpu = pred.cpu()
    true_cpu = truth.cpu()
    
    pred_unique = torch.unique(pred_cpu)
    true_unique = torch.unique(true_cpu)
    
    # dices = []
    dices = dict()
    pred_list = []
    true_list = []

    # convert the classes into 0,1,2,3,4,.....
    # E.g [banana,apple,lemon] -> [0,1,2]
    for i in range(len(pred_unique)):
        pred_cpu[pred_cpu==pred_unique[i]] = i
    
    for i in range(len(true_unique)):
        # print(cls)
        true_cpu[true_cpu==true_unique[i]] = i

    # Converts each class(0,1,2,3....) into boolean arrays and append them to a list
    # E.g
    # [0,0,0]    [1,1,1] [0,0,0] [0,0,0]
    # [1,1,1] => [0,0,0],[1,1,1],[0,0,0]
    # [2,2,2]    [0,0,0] [0,0,0] [1,1,1]
    # The resulting list will contain all the classes, with their index in the list corresponding to their class
    # E.g index[0] => class label 0
    for i in range(len(pred_unique)):
        pred_bool = torch.where(pred_cpu == i,True,False)
        true_bool = torch.where(true_cpu == i,True,False)
        pred_list.append(pred_bool)
        true_list.append(true_bool)

    for i in range(len(pred_unique)):
        intersection = torch.logical_and(pred_list[i],true_list[i])
        dice = 2 * intersection.sum()/(pred_list[i].sum() + true_list[i].sum())
        # dices.append(dice)
        dices[f"{i}"] = dice
    dices["avg"] = np.average(list(dices.values()))
    # return np.average(dices)
    return dices

def create_dataloader(dataset,batch_size:int,shuffle:bool):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

def test_model(model_params,state_dict_path,data_loader,output_path):
    model = ReLayNet(model_params)
    criterion = CombinedLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_list = []
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()
    count = 0
    running_test_loss = 0.0
    test_losses = []
    dirs_list = []

    # test_dices = []
    test_dices = {
        "0": [],
        "1": [],
        "2": [],
        "avg": []
    }

    if os.path.exists(f"{output_path}"):
        print("Output path already exists. Deleting the folder...")
        shutil.rmtree(f"{output_path}")
    print("Creating results directory and subdirectories...")
    os.mkdir(f'{output_path}')
    os.mkdir(f"{output_path}/truths/")
    os.mkdir(f"{output_path}/preds/")
    os.mkdir(f"{output_path}/display/")
    with torch.no_grad():
        for i,data in enumerate(tqdm(data_loader)):
            img,mask,weights,file_names = data
            img = img.to(device)
            mask = mask.to(device)
#             mask = torch.permute(mask,(0,3,1,2))
            weights = weights.to(device)
            pred = model(img)

            maxxed_pred = torch.argmax(pred,dim=1)
            loss = criterion(pred,mask,weights)
#             Calculate dice score in process
            test_dice = calc_dice(maxxed_pred,mask)
            # test_dices.append(test_dice)
            test_dices["0"].append(test_dice["0"])
            test_dices["1"].append(test_dice["1"])
            test_dices["2"].append(test_dice["2"])
            test_dices["avg"].append(test_dice["avg"])

            running_test_loss += loss.item()
            test_losses.append(loss.item())
            print(f"pred shape:{pred.shape}")
            print(f"maxxed pred shape:{maxxed_pred.shape}")
            print(f"test dice:{test_dice}")
            print(f"{file_names}")
            file_pred = zip(mask,maxxed_pred,file_names)
            for truth,pred,name in file_pred:

                # print(f"pred shape is :{pred.shape}")
                # print(f"pred type is :{pred.type()}")
                pil = transforms.ToPILImage()
                truth_pil = pil(truth)
                img_pil = pil(pred.type(torch.uint8))
                # print(img_pil.size)
                truth_pil.save(f"{output_path}/truths/{name}")
                img_pil.save(f"{output_path}/preds/{name}")
                count+=1
    for root,dirs,files in os.walk(f"{output_path}/preds/"):
        for f in files:
            pred = Image.open(os.path.join(root,f))
            pred_arr = np.array(pred)
            pred_save = plt.imsave(f"{output_path}/display/" + f"{f}",pred_arr)

    avg_test_dice_0 = np.average(test_dices["0"])
    avg_test_dice_1 = np.average(test_dices["1"])
    avg_test_dice_2 = np.average(test_dices["2"])
    avg_test_dice = np.average(test_dices["avg"])
    avg_test_loss = np.average(test_losses)

    return avg_test_loss,avg_test_dice_0,avg_test_dice_1,avg_test_dice_2,avg_test_dice

def calc_individual_dice(root_dir:str):
    pred_path = os.path.join(root_dir,"preds")
    truth_path = os.path.join(root_dir,"truths")

    preds = sorted(os.listdir(pred_path))
    truths = sorted(os.listdir(truth_path))

    df = {
        "name":[],
        "avg":[],
        "0":[],
        "1":[],
        "2":[]
    }
    pair = zip(preds,truths)
    for p,t in pair:
        pred_img_path = os.path.join(pred_path,p)
        truth_img_path = os.path.join(truth_path,t)
        pred_mask = torch.from_numpy(np.array(Image.open(pred_img_path)))
        truth_mask = torch.from_numpy(np.array(Image.open(truth_img_path)))
        dice = calc_dice(pred_mask,truth_mask)

        df["name"].append(p)
        df["avg"].append(dice["avg"])
        df["0"].append(float(dice["0"]))
        df["1"].append(float(dice["1"]))
        df["2"].append(float(dice["2"]))

    return pd.DataFrame(df)


# preds = os.listdir("results_segweights_run_1_half_y2_conv/preds/")
# print(len(preds))
df = calc_individual_dice(root_dir="results_segweights_run_1_half_y2_conv")

###########
# Classes #
###########
class MsdTestDataset(Dataset):
    def __init__(self,root_dir,mode,w1,w2,transform=None) -> None:
        """

        Args:
            root_dir (str): Base directory
            mode (str): Either 'train' or 'val'. Denotes the type of data(E.g train or val)
            w1 (int): Pixel weighing factor for pixels near boundaries
            w2 (int): Pixel weighing factor for pixels belonging to non-dominant classes
            transform (_type_, optional): Pytorch transforms created via pytorch's transform.Compose. Defaults to None.
        """
        super().__init__()

        ###########################
        # Implementation footnote #
        ###########################
        # We use sorted(os.listdir) because the same file(image) will appear in all the different folders with the same name
        # E.g test_img.png will exist in the following directories in the following format
        # data_dir: test_img.tif
        # display_dir: test_img.png
        # seg_dir: test_img.png
        # weight_dir: test_img.npy
        # Therefore, when we use sorted(which is a stable sort), they will all be in the same order

        # data
        self.data_dir = root_dir + f"/data/{mode}/"
        self.data_files = sorted(os.listdir(self.data_dir))
        # labels
        self.display_dir = root_dir + f"/display/{mode}/"
        self.display_files = sorted(os.listdir(self.display_dir))

        self.seg_dir = root_dir + f"/segmentations/{mode}/"
        self.seg_files = sorted(os.listdir(self.seg_dir))
        
        self.weight_dir = root_dir + f"/weights/{mode}/"
        self.weight_files = sorted(os.listdir(self.weight_dir))
        
        self.transforms = {
            "train": transforms.Compose([transforms.ToTensor()]),
            "val": transforms.Compose([transforms.ToTensor()]),
            "test": transforms.Compose([transforms.ToTensor()])
        }
        self.transform = self.transforms[mode]
        self.w1 = w1
        self.w2 = w2

    def __len__(self):
        return len(os.listdir(self.data_dir))        

    def __getitem__(self, index):
        # return super().__getitem__(index)
        # load images
        file_path = os.path.join(self.data_dir,self.data_files[index])
        display_path = os.path.join(self.display_dir,self.display_files[index])
        seg_path = os.path.join(self.seg_dir,self.seg_files[index])
        weight_path = os.path.join(self.weight_dir,self.weight_files[index])
        pred_name = self.seg_files[index]
        file = Image.open(file_path)
        display = Image.open(display_path)
        seg = Image.open(seg_path)
        ###########################
        # Implementation Footnote #
        ###########################
        # uncomment the following 3 lines and comment out seg=torch.from_numpy(np.array(seg)) to use 3d mask

#         test_mask = seg.convert("RGB")
        # test_mask_arr = np.array(test_mask)
#         seg = torch.from_numpy(np.array(test_mask_arr))
        if self.transform is not None:
            file = self.transform(file)
            display = self.transform(display)

# Comment out this line and uncomment the 3 lines above to use 3d mask     
        seg = torch.from_numpy(np.array(seg))
        
        weights = np.load(weight_path)

        return file,seg,weights,pred_name

