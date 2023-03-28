
import argparse
import ast
import json
import numpy as np
import time
from json import encoder
from autoencoder.cnnautoencoder_asym import AE
import importlib
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from preprocessing import LazyDataset,PatchedImage,PatchCoord
import imseg_wrapper
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from PIL import Image
import dask
import dask.array as da



def fit(epochs,model,train_loader,val_loader,criterion,optimizer,scheduler):
    print("START!")
    start_time = time.time()
    for e in range(epochs):
        print(f"Epoch {e}/{epochs}")
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            print(f"image {i}")
        #     print(next(iter(train_loader)))
            img,mask = data
            print(img)
            print(mask)
            # fwd & backward
            print("Inputting to model")
            outputs = model(img)
            print("Fwd pass done")
            loss = criterion(outputs,mask)
            loss.backward()
            optimizer.step()

            # running_stats
            running_loss += loss.item()
            print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss :.3f}')
            running_loss = 0.0





"Accepts model architecture and path to dataset"
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test Argument Parser for Snakemake")
    
    parser.add_argument("-train",dest="train_dir",type=str,default="data/train",help="Directory containing the image datasets")
    parser.add_argument("-test",dest="test_dir",type=str,default="data/test",help="Directory containing the image datasets")
    parser.add_argument("-output",dest="output_dir",type=str,default="preds",help="Directory containing the predicted images")

    parser.add_argument("-model",dest="model_import",type=str,help="module import for user-defined model.")
    parser.add_argument("-model_num_channels",dest="model_num_channels",type=int,default=1)
    parser.add_argument("-model_num_filters",dest="model_num_filters",type=int,default=64)
    parser.add_argument("-model_kernel_h",dest="model_kernel_h",type=int,default=7)
    parser.add_argument("-model_kernel_w",dest="model_kernel_w",type=int,default=3)
    parser.add_argument("-model_kernel_c",dest="model_kernel_c",type=int,default=1)
    parser.add_argument("-model_stride_conv",dest="model_stride_conv",type=int,default=1)
    parser.add_argument("-model_pool",dest="model_pool",type=int,default=2)
    parser.add_argument("-model_stride_pool",dest="model_stride_pool",type=int,default=2)
    parser.add_argument("-model_num_class",dest="model_num_classes",type=int,default=3)
    parser.add_argument("-model_epochs",dest="model_epochs",type=int,default=6)

    parser.add_argument("-optim_type",dest="optim_type",type=str,default="sgd")
    parser.add_argument("-optim_lr",dest="optim_lr",type=float,default=0.1)
    parser.add_argument("-optim_momentum",dest="optim_momentum",type=float,default=0.9)
    parser.add_argument("-optim_weight_decay",dest="optim_weight_decay",type=float,default=0.0001)

    parser.add_argument("-sched_type",dest="sched_type")
    parser.add_argument("-sched_step_size",dest="sched_step_size",type=int,default=30)
    parser.add_argument("-sched_gamma",dest="sched_gamma",type=float,default=0.1)

    parser.add_argument("-hyper_epochs",dest="hyper_epochs",type=int,default=6)
    parser.add_argument("-hyper_lr",dest="hyper_lr",type=float,default=0.1)

    # parser.add_argument("-model_args",help="arguments to instantiate user-defined model",nargs="+")
    
    args = parser.parse_args()
    
    # print(args.)
    train_dir = args.train_dir
    test_dir = args.test_dir
    output_dir = args.output_dir

    model_import = args.model_import
    user_model = importlib.import_module(model_import)
    model_num_channels = args.model_num_channels
    model_num_filters = args.model_num_filters
    model_kernel_h = args.model_kernel_h
    model_kernel_w = args.model_kernel_w
    model_kernel_c = args.model_kernel_c
    model_stride_conv = args.model_stride_conv
    model_pool = args.model_pool
    model_stride_pool = args.model_stride_pool
    model_num_classes = args.model_num_classes
    model_epochs = args.model_epochs

    optim_type = args.optim_type
    optim_lr = args.optim_lr
    optim_momentum = args.optim_momentum
    optim_weight_decay = args.optim_weight_decay

    sched_type = args.sched_type
    sched_step_size = args.sched_step_size
    sched_gamma = args.sched_gamma

    hyper_epochs = args.hyper_epochs
    hyper_lr = args.hyper_lr

    
    # instantiate user model
    model_params = {
        'num_channels': model_num_channels,
        'num_filters': model_num_filters,
        'kernel_h': model_kernel_h,
        'kernel_w': model_kernel_w,
        'kernel_c': model_kernel_c,
        'stride_conv': model_stride_conv,
        'pool': model_pool,
        'stride_pool': model_stride_pool,
        'num_class': model_num_classes,
        "epochs": model_epochs
    }
    print("Instantiating model...")
    model = user_model.ReLayNet(model_params)
    print("Relaynet instantiated!")
    print(model)

    print("Setting up criterion")
    criterion = nn.CrossEntropyLoss()
    print("Setting up criterion")

    print("Initializing criterion...")
    optimizer = torch.optim.SGD(model.parameters(),lr=optim_lr,momentum=optim_momentum,weight_decay=optim_weight_decay)
    print("Criterion initialized!")

    print("Initializing scheduler...")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=sched_step_size,gamma=sched_gamma)
    print("Scheduler initialized!")

    print("Initializing train loader...")
    transform = {
        "train": transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])

    }
    train_dataset = ImageFolder(root=train_dir,transform=transform["train"])
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=True)
    print("Train loader initialized!")

    print("Initializing test loader...")
    test_dataset = ImageFolder(root=test_dir,transform=transform["test"])
    test_loader = DataLoader(test_dataset,batch_size=4,shuffle=True)
    print("Test loader initialized!")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Starting training...")
    history = fit(epochs=model_epochs, model=model, train_loader=train_loader, val_loader=test_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
