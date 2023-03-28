#!/usr/bin/env python

####################
# Required Modules #
####################

#Generic Built in
import asyncio
import os
import pathlib
import shutil
from datetime import timedelta
import typing
#Libs
import numpy as np
import pandas as pd
import ray
import torch
from PIL import Image

##################
# Configurations #
##################



#############
# Functions #
#############


def get_files(train_path:str,val_path:str,test_path:str,source_folder:str)->pd.DataFrame:
    """
    Reads the CSV files of the train,val and test files and returns a Pandas dataframe that is the concatenation
    of all 3 CSV files.

    Input:
        train_files(str): path to CSV file containing files to be used in training
        val_files(str): path to CSV file containing files to be used in validation
        test_files(str): path to CSV file containing files to be used in testing
        source_folder(str): String containing the root folder of the source data E.g if the input folder is 
                            './msd' then the source folder will be msd
    Output:
        combined_files: Pandas dataframe that is the concatenation of all 3 files(train,val,test)
    """
    train_files = pd.read_csv(train_path,index_col=None)
    val_files = pd.read_csv(val_path,index_col=None)
    test_files = pd.read_csv(test_path,index_col=None)

    train_files["folders"] = train_files["folders"].replace("^./*",f"{source_folder}/",regex=True)
    val_files["folders"] = val_files["folders"].replace("^./*",f"{source_folder}/",regex=True)
    test_files["folders"] = test_files["folders"].replace("^./*",f"{source_folder}/",regex=True)
    combined_files = pd.concat([train_files,val_files,test_files])
    return combined_files

@ray.remote
def calc_weights_ray(w1:int,w2:int,queue)->dict:
    """
    W1: Weighting for pixels that are proximal to tissue-transition regions. E.g pixels near the boundary/edges between to different segments
        Equates one if gradient between 2 pixels is more than 1 -> If the pixel (x) is besides
    W2: Equals one if the class label belongs to an under-represented class
    
    """
    details = queue
    label=details["source"]
    dest = details["dest"]
    label_img = Image.open(label)
    label_arr = np.array(label_img)
    # raw_tensor = torch.from_numpy(raw) 
    label_tensor = torch.from_numpy(label_arr)
    
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

    w1_weighted_map = w1 * w1_map
    w2_weighted_map = w2 * w2_map
    one_map = torch.ones(w1_map.shape)
    w1w2_map = torch.add(w1_weighted_map,w2_weighted_map)
    weighted_map = torch.add(one_map,w1w2_map)
    output_details={
        "dest":dest,
        "weights":weighted_map
    }
    # output_queue.append(output_details)
    return output_details
    

@ray.remote(num_returns=2)
def create_dest_ray(train_val_test_paths:dict[str,str],path:str,root:str,set_files:dict,f):
    """
    Does the following:
    1. Creates a dictionary of the filepaths to take the files from, and save the files to 
    2. Calls the remote ray function copy_files_ray() with the dict of file paths which returns an obj reference
    3. Returns the obj reference returned by the remote ray func copy_files_ray() and a dict that contains the filepaths to
    load the image used to calc the weights, and the filepath to save the calculated weights to

    Args:
        train_val_test_paths (dict[str,str]): Dictionary of file paths
        path (str): cur working dir
        root (str): filepath
        set_files (dict): _description_
        f (_type_): the file to process

    Returns:
       files_copied: Ray obj reference
       weights_details: dict of file paths
    """
    data_train_path = train_val_test_paths["data_train_path"]
    data_val_path = train_val_test_paths["data_val_path"]
    data_test_path = train_val_test_paths["data_test_path"]
    
    display_train_path = train_val_test_paths["display_train_path"]
    display_val_path = train_val_test_paths["display_val_path"]
    display_test_path = train_val_test_paths["display_test_path"]

    seg_train_path = train_val_test_paths["seg_train_path"]
    seg_val_path = train_val_test_paths["seg_val_path"]
    seg_test_path = train_val_test_paths["seg_test_path"]

    weight_train_path = train_val_test_paths["weight_train_path"]
    weight_val_path = train_val_test_paths["weight_val_path"]
    weight_test_path = train_val_test_paths["weight_test_path"]
    
    root_split = root.split("/")
    prefix = root_split[-1]
    display_root = root.replace("/Data/","/Display/")
    seg_root = root.replace("/Data/","/Segmentations/")

    # weight_root = root.replace("/Data/","weights")

    f_new = f.replace(".tif","_display.png")
    # Removes the _display suffix to bring the filename inline with the ones in the Data folder
    f_std = f_new.replace("_display.png",".png")
    f_seg = f.replace(".tif",".png")
    file_new_name = prefix + f
    file_new_name2 = prefix + f_std
    file_new_name_seg = prefix + f_seg
    weight_file_name = file_new_name_seg.split(".")[0]

    # source
    source_cur_path = os.path.join(path,root,f)
    # visualised ground truth mask
    display_cur_path = os.path.join(path,display_root,f_new)
    # ground truth mask
    seg_cur_path = os.path.join(path,seg_root,f_seg)

    if (set_files[root]["type"] == "train"):
        source_dest_path = os.path.join(data_train_path,file_new_name)
        display_dest_path = os.path.join(display_train_path,file_new_name2)
        seg_dest_path = os.path.join(seg_train_path,file_new_name_seg)
        weight_dest_path = os.path.join(weight_train_path,weight_file_name)
    elif (set_files[root]["type"] == "val"):
        source_dest_path = os.path.join(data_val_path,file_new_name)
        display_dest_path = os.path.join(display_val_path,file_new_name2)
        seg_dest_path = os.path.join(seg_val_path,file_new_name_seg)
        weight_dest_path = os.path.join(weight_val_path,weight_file_name)
    elif (set_files[root]["type"] == "test"):
        source_dest_path = os.path.join(data_test_path,file_new_name)
        display_dest_path = os.path.join(display_test_path,file_new_name2)
        seg_dest_path = os.path.join(seg_test_path,file_new_name_seg)
        weight_dest_path = os.path.join(weight_test_path,weight_file_name)
    # else:
    #     raise Exception("Error, unknown type")
    dest_paths = {
        "source_cur_path":source_cur_path,
        "source_dest_path":source_dest_path,
        "display_cur_path":display_cur_path,
        "display_dest_path":display_dest_path,
        "seg_cur_path":seg_cur_path,
        "seg_dest_path":seg_dest_path,
        "weight_dest_path":weight_dest_path
    }
    weights_details={
        "source":seg_cur_path,
        "dest":weight_dest_path
    }
    files_copied = copy_files_ray.remote(dest_paths)
    return files_copied,weights_details

@ray.remote
def copy_files_ray(queue):
    paths = queue
    if len(paths) > 0:
        try:
            shutil.copy(paths["source_cur_path"],paths["source_dest_path"])
            shutil.copy(paths["display_cur_path"],paths["display_dest_path"])
            shutil.copy(paths["seg_cur_path"],paths["seg_dest_path"])
        except Exception as e:
            print(f"Error: {e} occured while copying files")
    # queue.task_done()

@ray.remote
def save_weights_ray(queue):
    """
    queue: Contains a dictionary containing the following 
        dest => Destination file path to save the weights to
        weights => The calculated weights    
    """
    details=queue
    file_path=details["dest"]
    weights=details["weights"]
    np.save(file_path,weights)

def prep_msd_data_ray(combined_files:pd.DataFrame,dest:str,w1:int,w2:int)->None:

    """
    Does the following
    1. Check if destination directories exists
    1a.Create directories if not exists
    2. Get file types(E.g train/test/val),num_files and partitions as an iterable
    3. Iterate through the files in the MSD raw dataset and do the following operations:-
    3a.Calculate weights asynchronously and add the calculated weights to a queue
    3b.Rename the files in the Data,Dispaly,Segmentations folders in the raw data and add the destination filepaths to a queue
    4. Async writers/processes will copy/write everything in the queue to their destinations

    There are 3 queues:
    1. paths_queue -> contains dictionary containing the source and destination paths of MSD images to copy from/to
    2. weights_calc_queue -> contains source path of the image that will be used to calculate the weights
    3. weights_queue -> contains the 
    Create_paths creates a dictionary of the paths and adds it to a queue

    """
    path = os.getcwd()
    ###########################
    # Implementation Footnote #
    ###########################
    # We will delete any existing directory that is specified as the destination directory
    if os.path.exists(dest):
        shutil.rmtree(f"{dest}")

    data_dest_root = os.path.join(path,dest,"data")
    display_dest_root = os.path.join(path,dest,"display")
    seg_dest_root = os.path.join(path,dest,"segmentations")
    weight_dest_root = os.path.join(path,dest,"weights")

    data_train_path = os.path.join(data_dest_root,"train")
    data_val_path = os.path.join(data_dest_root,"val")
    data_test_path = os.path.join(data_dest_root,"test")

    display_train_path = os.path.join(display_dest_root,"train")
    display_val_path = os.path.join(display_dest_root,"val")
    display_test_path = os.path.join(display_dest_root,"test")

    seg_train_path = os.path.join(seg_dest_root,"train")
    seg_val_path = os.path.join(seg_dest_root,"val")
    seg_test_path = os.path.join(seg_dest_root,"test")

    weight_train_path = os.path.join(weight_dest_root,"train")
    weight_val_path = os.path.join(weight_dest_root,"val")
    weight_test_path = os.path.join(weight_dest_root,"test")
    ###########################
    # Implementation Footnote #
    ###########################
    # Create the fodler in the specified file path. If parents folders in the filepath do not exist, 
    # we will create the parents as well. the parameter exist_ok=True is a flag so that we will not
    # raise a FileExistError if it already exists
    pathlib.Path(data_train_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(data_val_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(data_test_path).mkdir(parents=True,exist_ok=True)

    pathlib.Path(display_train_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(display_val_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(display_test_path).mkdir(parents=True,exist_ok=True)

    pathlib.Path(seg_train_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(seg_val_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(seg_test_path).mkdir(parents=True,exist_ok=True)

    pathlib.Path(weight_train_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(weight_val_path).mkdir(parents=True,exist_ok=True)
    pathlib.Path(weight_test_path).mkdir(parents=True,exist_ok=True)

    # iterate through train,val and test
    set_files = {}
    train_val_test_paths = {
        "data_train_path" : data_train_path,
        "data_val_path" : data_val_path,
        "data_test_path" : data_test_path,
        "display_train_path" : display_train_path,
        "display_val_path" : display_val_path,
        "display_test_path" : display_test_path,
        "seg_train_path" : seg_train_path,
        "seg_val_path" : seg_val_path,
        "seg_test_path" : seg_test_path,
        "weight_train_path" : weight_train_path,
        "weight_val_path" : weight_val_path,
        "weight_test_path" : weight_test_path
    }
    for each in combined_files.itertuples():
        # Indices are (index,folderpath,mode,number,partition#)
        file_path = each[1]
        # file_path = file_path.replace("\\","/")
        # print(file_path)
        file_type = each[2]
        file_num = each[3]
        file_partition = each[4]
        set_files[file_path] = {"type":file_type,"num":file_num,"identifier":file_partition}
      
    count = 1
    path = os.getcwd()
    # total_paths_queue = []

    # total_calc_weights_queue = []
    # weights_paths_queue = []
    
    ###########################
    # Implementation Footnote #
    ###########################
    # MAX_TASK is used to limit the number of remote tasks so that we don't run out of memory

    weighted_map_queue = []
    saved_weights_queue = []
    files_copied_queue = []
    MAX_TASKS = 100
    count = 0
    for root,dirs,files in os.walk("msd_data/"):
        root = root.replace("\\","/")
        if (root in set_files):
            for f in files:
                if count < MAX_TASKS:
                    files_copied,weights_details = create_dest_ray.remote(train_val_test_paths=train_val_test_paths,path=path,root=root,set_files=set_files,f=f)
                    files_copied_queue.append(files_copied)
                    weighted_map = calc_weights_ray.remote(w1=w1,w2=w2,queue=weights_details)
                    weighted_map_queue.append(weighted_map)
                    saved_weights = save_weights_ray.remote(weighted_map)
                    saved_weights_queue.append(saved_weights)
                    count += 1
                else:
                    ###########################
                    # Implementation Footnote #
                    ###########################
                    # MAX_TASK is used to limit the number of remote tasks so that we don't run out of memory
                    files_copied,weights_details = create_dest_ray.remote(train_val_test_paths=train_val_test_paths,path=path,root=root,set_files=set_files,f=f)
                    # paths_queue.append(dest_paths)
                    files_copied_queue.append(files_copied)
                    # calc_weights_queue.append(weights_details)
                    # files_copied = copy_files_ray.remote(dest_paths)
                    weighted_map = calc_weights_ray.remote(w1=w1,w2=w2,queue=weights_details)
                    weighted_map_queue.append(weighted_map)
                    saved_weights = save_weights_ray.remote(weighted_map)
                    saved_weights_queue.append(saved_weights)

                    copied_files = ray.get(files_copied_queue)
                    # print(f"Len:{len(copied_files)}")
                    weighted_maps = ray.get(weighted_map_queue)
                    saved_weights = ray.get(saved_weights_queue)
                    weighted_map_queue = []
                    saved_weights_queue = []
                    files_copied_queue = []
                    count = 0
    weighted_maps = ray.get(weighted_map_queue)
    saved_weights = ray.get(saved_weights_queue)
    copied_files = ray.get(files_copied_queue)
