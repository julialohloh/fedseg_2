# Dependencies
import os
import pandas as pd
import pathlib
import shutil
from PIL import Image
import numpy as np
import torch
import dask
import time
from datetime import timedelta
import asyncio



def get_files(train_path:str,val_path:str,test_path:str,source_folder:str):
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

def calc_weights(w1,w2,label):
    """
    W1: Weighting for pixels that are proximal to tissue-transition regions. E.g pixels near the boundary/edges between to different segments
        Equates one if gradient between 2 pixels is more than 1 -> If the pixel (x) is besides
    W2: Equals one if the class label belongs to an under-represented class
    
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
    
def prep_msd_data(combined_files,dest,w1,w2):
    start_time = time.time()
    path = os.getcwd()
    # make dirs if not exists
#     print(f"DEST IS :{dest}")
    if os.path.exists(dest):
        print("Output path already exists. Deleting the folder...")
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
#     else:
#         print("ALRDY EXISTS")

    # iterate through train,val and test
    set_files = {}
    
    for each in combined_files.itertuples():
        # Indices are (index,folder,mode,number,partition#)
        file_path = each[1]
        # file_path = file_path.replace("\\","/")
        # print(file_path)
        file_type = each[2]
        file_num = each[3]
        file_partition = each[4]
        set_files[file_path] = {"type":file_type,"num":file_num,"identifier":file_partition}
      
        # print(f"File path is {file_path}, file type is {file_type}, identifier num is {file_identifier}")
    # return set_files
    count = 1
    path = os.getcwd()
    print("walking through msd_data...")
    for root,dirs,files in os.walk("msd_data/"):
        # if count %100==0:
        #     print(count)    
        root = root.replace("\\","/")
        if (root in set_files) and (set_files[root]["type"] == "train"):
            # print(root)
            # print(files)
            for f in files:
                root_split = root.split("/")
                prefix = root_split[-1]
                display_root = root.replace("/Data/","/Display/")
                seg_root = root.replace("/Data/","/Segmentations/")
                # no need for weight_root?
                weight_root = root.replace("/Data/","weights")

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
                source_dest_path = os.path.join(data_train_path,file_new_name)
                # ground truth segmentation mask
                display_cur_path = os.path.join(path,display_root,f_new)
                display_dest_path = os.path.join(display_train_path,file_new_name2)
                # black background
                seg_cur_path = os.path.join(path,seg_root,f_seg)
                seg_dest_path = os.path.join(seg_train_path,file_new_name_seg)
                # calculate weight of the image
                label_img = Image.open(seg_cur_path)
                label_arr = np.array(label_img)
                weight_map = calc_weights(w1,w2,label_arr)
                # save weightmap as image
                weight_dest_path = os.path.join(weight_train_path,weight_file_name)
                weight_map_arr = np.array(weight_map)
                

                shutil.copy(source_cur_path,source_dest_path)
                shutil.copy(display_cur_path,display_dest_path)
                shutil.copy(seg_cur_path,seg_dest_path)
                np.save(weight_dest_path,weight_map_arr)
        elif (root in set_files) and (set_files[root]["type"] == "val"):
        #     print(root)
        #     print(files)
            for f in files:
                root_split = root.split("/")
                prefix = root_split[-1]
                # print(root)
                display_root = root.replace("/Data/","/Display/")
                seg_root = root.replace("/Data/","/Segmentations/")

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
                source_dest_path = os.path.join(data_val_path,file_new_name)
                # ground truth segmentation mask
                display_cur_path = os.path.join(path,display_root,f_new)
                display_dest_path = os.path.join(display_val_path,file_new_name2)
                # black background
                seg_cur_path = os.path.join(path,seg_root,f_seg)
                seg_dest_path = os.path.join(seg_val_path,file_new_name_seg)
                # calculate weight of the image
                label_img = Image.open(source_cur_path)
                label_arr = np.array(label_img)
                weight_map = calc_weights(w1,w2,label_arr)
                # save weightmap as image
                weight_dest_path = os.path.join(weight_val_path,weight_file_name)
                weight_map_arr = np.array(weight_map)


                shutil.copy(source_cur_path,source_dest_path)
                shutil.copy(display_cur_path,display_dest_path)
                shutil.copy(seg_cur_path,seg_dest_path)
                np.save(weight_dest_path,weight_map_arr)
        elif (root in set_files) and (set_files[root]["type"] == "test"):
            for f in files:
                root_split = root.split("/")
                prefix = root_split[-1]

                display_root = root.replace("/Data/","/Display/")

                seg_root = root.replace("/Data/","/Segmentations/")

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
                source_dest_path = os.path.join(data_test_path,file_new_name)
                # ground truth segmentation mask
                display_cur_path = os.path.join(path,display_root,f_new)
                display_dest_path = os.path.join(display_test_path,file_new_name2)
                # black background
                seg_cur_path = os.path.join(path,seg_root,f_seg)
                seg_dest_path = os.path.join(seg_test_path,file_new_name_seg)
                # calculate weight of the image
                label_img = Image.open(source_cur_path)
                label_arr = np.array(label_img)
                weight_map = calc_weights(w1,w2,label_arr)
                # save weightmap as image
                weight_dest_path = os.path.join(weight_test_path,weight_file_name)
                weight_map_arr = np.array(weight_map)
                
 
                shutil.copy(source_cur_path,source_dest_path)
                shutil.copy(display_cur_path,display_dest_path)
                shutil.copy(seg_cur_path,seg_dest_path)
                np.save(weight_dest_path,weight_map_arr)
    elapsed_time = time.time() - start_time
    time_msg = f"Data prep took {timedelta(seconds=elapsed_time)} seconds"
    print(time_msg)
                
# if __name__ == "__main__"  :

#     destination = "data_raw_img"
#     prep_msd_data(combined_files,destination,w1=10,w2=5)
