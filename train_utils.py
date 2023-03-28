#!/usr/bin/env python

####################
# Required Modules #
####################

#Generic Built in

import os
import time
from datetime import timedelta

#Libs
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

#Custom
from early_stop import EarlyStopping
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
        float: Average dice score across all classes
    """
    pred_cpu = pred.cpu()
    true_cpu = truth.cpu()
    
    pred_unique = torch.unique(pred_cpu)
    true_unique = torch.unique(true_cpu)
    
    dices = []
    pred_list = []
    true_list = []

    # convert the classes into 0,1,2,3,4,.....
    for i in range(len(pred_unique)):
        # print(cls)
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
        # print(dice)
        dices.append(dice)
    return np.average(dices)

def train_model(epochs:int,model,train_loader,val_loader,criterion:CombinedLoss,scheduler,optimizer:torch.optim,patience:int,checkpoint_path:str):
    """
    Trains the model

    Args:
        epochs (int): num of epochs to run training for
        model (_type_): Instance of model to train
        train_loader (torch.utils.data.Dataloader): Pytorch Dataloader for training data
        val_loader (torch.utils.data.Dataloader): Pytorch Dataloader for validation data
        criterion (CombinedLoss): Loss function
        optimizer (torch.optim): Optimizer
        patience (int): num of epochs for early stopping. Early stopping will trigger if there are #patience of epochs w/o improvement
        checkpoint_path (str): Filepath to save model weights
    """
    print("Training started!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # start_time = time.time()
    train_losses = []
    train_dices = []
    valid_losses = []
    val_dices = []
    avg_train_losses = []
    avg_train_dice = []
    avg_valid_losses = []
    avg_val_dice = []
    running_train_loss = 0.0
    running_valid_loss = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True,path=checkpoint_path)
    for e in range(1,epochs+1):
        print(f"Epoch {e}/{epochs}")
        ###################
        # train the model #                    
        ###################
        model.train()
        for i,data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            img,mask,weights = data
#             mask = torch.permute(mask,(0,3,1,2))

            img = img.to(device)
            mask = mask.to(device)
            weights = weights.to(device)

            outputs = model(img)

            # calc train dice
            maxxed = torch.argmax(outputs,dim=1)
#             train_dice = calc_dice(maxxed,mask)
            train_dice = calc_dice(mask,maxxed)

            loss = criterion(outputs,mask,weights)
            loss.backward()
            optimizer.step()
#             scheduler.step()

            # running_stats
            running_train_loss += loss.item()
            train_losses.append(loss.item())
            train_dices.append(train_dice)
            if i%100 == 0:
                print(f'Training Epoch {e}, Iter {i + 1:5d} Running loss: {running_train_loss :.3f}, Dice: {train_dice}')
        ######################
        # validate the model #                    
        ######################
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(tqdm(val_loader)):
                img,mask,weights = data
                img = img.to(device)
                mask = mask.to(device)
    #             mask = torch.permute(mask,(0,3,1,2))
                weights = weights.to(device)
                outputs = model(img)
                loss = criterion(outputs,mask,weights)
                # calc dice loss
                maxxed = torch.argmax(outputs,dim=1)
                val_dice = calc_dice(mask,maxxed)
#                 val_dice = calc_dice(mask,outputs)
                # running_stats
                running_valid_loss += loss.item()
                valid_losses.append(loss.item())
                val_dices.append(val_dice)
                if j%100 == 0:
                    print(f'Validation Epoch {e}, Iter {j + 1:5d} Running loss: {running_valid_loss :.3f}, Dice: {val_dice}')
        
        epoch_train_loss = np.average(train_losses)
        epoch_train_dice = np.average(train_dices)
        epoch_valid_loss = np.average(valid_losses)
        epoch_val_dice = np.average(val_dices)
        avg_train_losses.append(epoch_train_loss)
        avg_train_dice.append(epoch_train_dice)
        avg_valid_losses.append(epoch_valid_loss)
        avg_val_dice.append(epoch_val_dice)

        epoch_len = len(str(epochs+1))
        
        print_msg = (f'[{e:>{epoch_len}}/{epochs+1:>{epoch_len}}] ' +
                     f'avg_train_loss: {epoch_train_loss:.5f} ' +
                     f'avg_train_dice: {epoch_train_dice:.5f} ' +
                     f'avg_valid_loss: {epoch_valid_loss:.5f} ' +
                     f'avg_valid_dice: {epoch_val_dice:.5f}')
        
        print(print_msg)

        running_train_loss = 0.0
        running_valid_loss = 0.0
        train_losses = []
        valid_losses = []
        train_dices = []
        val_dices = []
        scheduler.step()

        early_stopping((epoch_val_dice*-1),model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    model.load_state_dict(torch.load(f'{checkpoint_path}'))

    return  model, avg_train_losses, avg_valid_losses

def run_training(root_dir:str,num_channels:int,num_filters:int,kernel_h:int,kernel_w:int,kernel_c:int,stride_conv:int,\
    pool:int,stride_pool:int,num_class:int,epochs:int,lr:float,momentum:float,weight_decay:float,step_size:int,gamma:float,w1:int,w2:int,batch_size:int,patience:int,\
        checkpoint_path:str):
    """
    Instantiates the model,loss function,optimiser and runs the model training loop

    Args:
        num_channels (int): input channels
        num_filters (int): num filters
        kernel_h (int): kernel height
        kernel_w (int): kernel width
        kernel_c (int): kernel channels
        stride_conv (int): 
        stride_pool (int): 
        num_class (int): 
        epochs (int): 
        lr (float): 
        momentum (float): 
        weight_decay (float): 
        step_size (int): 
        gamma (float): 
        w1 (int): _description_
        w2 (int): _description_
        batch_size (int): 
        patience (int): num of epochs for early stopping. Early stopping will trigger if there are #patience of epochs w/o improvement
    """
    print("Checking device...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    print("Instantiating model...")
    model_params = {
        'num_channels': num_channels,
        'num_filters': num_filters,
        'kernel_h': kernel_h,
        'kernel_w': kernel_w,
        'kernel_c': kernel_c,
        'stride_conv': stride_conv,
        'pool': pool,
        'stride_pool': stride_pool,
        'num_class': num_class,
        'epochs': epochs
    }
    model = ReLayNet(model_params)
    model = model.to(device)
    print("Relaynet instantiated!")

    print("Setting up criterion")
    criterion = CombinedLoss()
    print("Setting up criterion")

    print("Initializing criterion...")
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    print("Criterion initialized!")

    print("Initializing scheduler...")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
    print("Scheduler initialized!")

    print("Initializing train loader...")
    transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()]),
        "test": transforms.Compose([transforms.ToTensor()])

    }
    train_dataset = MsdDataset(root_dir=root_dir,mode="train",transform=transform["train"],w1=w1,w2=w2)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    print("Train loader initialized!")

    print("Initializing val loader...")
    val_dataset = MsdDataset(root_dir=root_dir,mode="val",transform=transform["val"],w1=w1,w2=w2)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    print("Val loader initialized!")

    print("Starting training...")
    trained_model = train_model(epochs=epochs, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,scheduler=scheduler,optimizer=optimizer, \
        patience=patience,checkpoint_path=checkpoint_path)
    print("Finished training!")

###########
# Classes #
###########

class MsdDataset(Dataset):
    """
    Custom Dataset for MSD data

    Args:
        Dataset (_type_): The pytorch class that we are inheriting from
    """
    def __init__(self,root_dir:str,mode:str,w1:int,w2:int,transform=None) -> None:
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


        # Input image
        self.data_dir = root_dir + f"/data/{mode}/"
        self.data_files = sorted(os.listdir(self.data_dir))
        # MSD colorised segmentation mask for visual purpose
        self.display_dir = root_dir + f"/display/{mode}/"
        self.display_files = sorted(os.listdir(self.display_dir))
        # Groudn truths
        self.seg_dir = root_dir + f"/segmentations/{mode}/"
        self.seg_files = sorted(os.listdir(self.seg_dir))
        
        self.weight_dir = root_dir + f"/weights/{mode}/"
        self.weight_files = sorted(os.listdir(self.weight_dir))
        
        self.transform = transform
        self.w1 = w1
        self.w2 = w2

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def calc_weights(self,w1,w2,label):
        """
        W1: Weighting for pixels that are proximal to tissue-transition regions. E.g pixels near the boundary/edges between to different segments
            Equates one if gradient between 2 pixels is more than 1 -> If the pixel (x) is besides
        W2: Equals one if the class labFw2el belongs to an under-represented class
        
        """
        # raw_tensor = torch.from_numpy(raw) 
        label_tensor = torch.from_numpy(label)
        
        # shape is (H,W)
        # print(label_tensor.shape)
        # Calculating the weights for W1

        # Initialise w1 weight map with all zeroes first
        w1_map = torch.zeros(label_tensor.shape)
        # print(f"Initialised w1_map \n {w1_map.shape}")
        # print("Calculating w1 map...")

        num_rows = label_tensor.shape[0]
        for row in range(1,num_rows):
            # We use row and row-1 so that we won't get an index out of bounds error while
            # iterating
            first_row = label_tensor[row-1,:]
            second_row = label_tensor[row,:]
            prev_a = None
            prev_b = None

            # iterate through each column in each rows
            # print(len(first_row))
            for col in range(len(first_row)):
                a = first_row[col]
                b = second_row[col]
                if a != b:
                    # There exists a boundary between a and b, so we should weigh these pixels
                    w1_map[row-1,col] = 1
                    w1_map[row,col] = 1
                else:
                    # if we are not at the first(leftmost) col, we check if pixels side by side are the same
                    # If they are not the same, there exists a boundary between a/b and prev_a/b so we should weigh
                    # these pixels
                    if (col != 0) and (prev_a is not None) and (prev_b is not None):
                        if a != prev_a:
                            w1_map[row-1,col] = 1
                            w1_map[row-1,col-1] = 1
                        if b != prev_b:
                            w1_map[row,col] = 1
                            w1_map[row,col-1] = 1
                    elif (col != 0 ) and (prev_a is None) and (prev_b is None):
                        raise Exception(f"Something went wrong, we were at row {row} col {col} and prev_a or prev_b is NOT none. prev_a:{prev_a}, prev_b:{prev_b}")
                    else:
                        # We are at the first(leftmost) col, and prev_a and prev_b is None so there is nothing to compare to
                        pass
                prev_a = a
                prev_b = b
        # End
        # print(f"Finished calculating W1 map")
        # print(w1_map)
        w1_map.float()
        # Initialise w2 weight map with all zeroes first
        w2_map = torch.zeros(label_tensor.shape)
        # class label/idx 2 is the "dominant" class so we will weigh the pixels with a class label that is != 2
        # w2_map = torch.eq(label_tensor,2).long()
        w2_map = torch.eq(label_tensor,2)
        # return 1 if value of w2_map = False, else return 0
        w2_map = torch.where(w2_map == False,1,0).float()
        # weighted_map = 1 + (w1*w1_map) + (w2*w2_map)
        # print(f"W1 : {w1}")
        # print(f"W1_map is \n {w1_map}\n")
        # print(f"W2 : {w2}")
        # print(f"W2_map is \n {w2_map}\n")
        # print(f"W1_weighted is : {(w1*w1_map)}\n")
        # print(f"W2_weighted is : {(w2*w2_map)}\n")
        # print(f"W1_weighted + W2_weighted is {np.add((w1*w1_map),(w2*w2_map))}")
        # print(f"w1 type is : {w1_map.type()}")
        # print(f"w2 type is : {w2_map.type()}")
        # print(f"one_map shape is {one_map.shape}")
        w1_weighted_map = w1 * w1_map
        w2_weighted_map = w2 * w2_map
        one_map = torch.ones(w1_map.shape)
        w1w2_map = torch.add(w1_weighted_map,w2_weighted_map)
        weighted_map = torch.add(one_map,w1w2_map)
        return weighted_map
        

    def __getitem__(self, index):
        # return super().__getitem__(index)

        # load images
        file_path = os.path.join(self.data_dir,self.data_files[index])
        display_path = os.path.join(self.display_dir,self.display_files[index])
        seg_path = os.path.join(self.seg_dir,self.seg_files[index])
        weight_path = os.path.join(self.weight_dir,self.weight_files[index])

        file = Image.open(file_path)
        display = Image.open(display_path)
        seg = Image.open(seg_path)
        ###########################
        # Implementation Footnote #
        ###########################
        # uncomment the following 3 lines and comment out seg=torch.from_numpy(np.array(seg)) to use 3d mask

#         test_mask = seg.convert("RGB")
#         test_mask_arr = np.array(test_mask)
#         seg = torch.from_numpy(np.array(test_mask_arr))
        if self.transform is not None:
            file = self.transform(file)
            display = self.transform(display)
            # seg = self.transform(seg)

        # Comment out this line and uncomment the 3 lines above to use 3d mask
        seg = torch.from_numpy(np.array(seg))
        weights = np.load(weight_path)

        return file,seg,weights

##########
# Script #
##########