import os
import pandas as pd
import pathlib
import shutil
from PIL import Image
import numpy as np
import torch
import dask
import asyncio

# async def produce(queue:asyncio.Queue):
#     for i in range(1,101):
#         print(f"producing {i}/100...")
#         num = np.random.randint(255)
#         print(f"Putting {num} into queue...")
#         await queue.put(num)
#         print(f"{num} entered into queue!")
#         # queue.put(num)

# async def consume(queue:asyncio.Queue):
#     while True:
#         num_to_write = await queue.get()
#         print(f"Writing {num_to_write} to disk...")
#         print(f"{num_to_write} written to disk!")
#         queue.task_done()

# async def test_async():
#     # mkdir
#     print(f"making dir....")
#     print(f"Directories created!")

#     # create queue
#     write_queue = asyncio.Queue()
#     # iterate through dir
#     await produce(write_queue)
#     consumers = [asyncio.create_task(consume(write_queue)) for n in range(3)]
#     await write_queue.join()
#     for c in consumers:
#         c.cancel()


# asyncio.run(test_async())

async def calc_weights_async(w1:int,w2:int,queue:asyncio.Queue,output_queue:asyncio.Queue):
    """
    W1: Weighting for pixels that are proximal to tissue-transition regions. E.g pixels near the boundary/edges between to different segments
        Equates one if gradient between 2 pixels is more than 1 -> If the pixel (x) is besides
    W2: Equals one if the class label belongs to an under-represented class
    
    """
    while True:
        details = await queue.get()
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
        output_details={
            "dest":dest,
            "weights":weighted_map
        }
        print("Putting in output_queue...")
        await output_queue.put(output_details)
        print("Entered in output_queue!")
        # return weighted_map

async def create_dest_async(train_val_test_paths,path,root,set_files,f,paths_queue,calc_weights_queue):
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
    # ground truth segmentation mask
    display_cur_path = os.path.join(path,display_root,f_new)
    # black background
    seg_cur_path = os.path.join(path,seg_root,f_seg)
    # creation of dest paths are sequential 
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
    else:
        raise Exception("Error, unknown type")
    dest_paths = {
        "source_cur_path":source_cur_path,
        "source_dest_path":source_dest_path,
        "display_cur_path": display_cur_path,
        "display_dest_path":display_dest_path,
        "seg_cur_path": seg_cur_path,
        "seg_dest_path":seg_dest_path,
        "weight_dest_path":weight_dest_path
    }

    weights_details={
        "source":source_cur_path,
        "dest":weight_dest_path
    }
    print("Putting into paths_queue ...")
    await paths_queue.put(dest_paths)
    print("Entered item into paths_queue")
    print("Putting into calc_weights_queue...")
    await calc_weights_queue.put(weights_details)
    print("Entered into calc_weights_queue")


async def copy_files_async(queue=asyncio.Queue):
    while True:
        paths = await queue.get()
        shutil.copy(paths["source_cur_path"],paths["source_dest_path"])
        shutil.copy(paths["display_cur_path"],paths["display_dest_path"])
        shutil.copy(paths["seg_cur_path"],paths["seg_dest_path"])
        queue.task_done()
        print("files copied!")


async def save_weights_async(queue=asyncio.Queue):
    """
    queue: Contains a dictionary containing the following 
        dest => Destination file path to save the weights to
        weights => The calculated weights    
    """
    while True:
        details= await queue.get()
        file_path=details["dest"]
        weights=details["weights"]
        np.save(file_path,weights)
        queue.task_done()
        print("weights saved!")

def check_dirs(dest:str)->None:
    # Checking of dirs should not be async
    path = os.getcwd()
    if not os.path.exists(dest):
        # make dir
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
    else:
        print("ALRDY EXISTS")

async def prep_msd_data_async(combined_files,dest,w1,w2):
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
    # check_dirs(dest=dest)
    path = os.getcwd()
    # make dirs if not exists
    if not os.path.exists(dest):
        # make dir
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
    else:
        print("ALRDY EXISTS")

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
    paths_queue = asyncio.Queue()
    calc_weights_queue = asyncio.Queue()
    weights_paths_queue = asyncio.Queue()
    for root,dirs,files in os.walk("msd_data/"):
        # if count %100==0:
        #     print(count)    
        root = root.replace("\\","/")
        if (root in set_files):
            for f in files:
                # produce and add to the queues
                print("Waiting for craete_dest_async...")
                await create_dest_async(train_val_test_paths=train_val_test_paths,path=path,root=root,set_files=set_files,f=f,paths_queue=paths_queue,calc_weights_queue=calc_weights_queue)
                print("Create_dest_async created!")
    print("Waiting for calc_weights_async")
    # await calc_weights_async(w1=w1,w2=w2,queue=calc_weights_queue,output_queue=weights_paths_queue)
    asyncio.create_task(calc_weights_async(w1=w1,w2=w2,queue=calc_weights_queue,output_queue=weights_paths_queue))
    # calculators = [asyncio.create_task(calc_weights_async(w1=w1,w2=w2,queue=calc_weights_queue,output_queue=weights_paths_queue)) for n in range(4)]
    print("Waited for calc_weights_async successfully!")
    # consume the queues
    print("Waiting for copy_files_async...")
    await copy_files_async(queue=paths_queue)
    print("Async_files_copy executed successfully!")
    print("Waiting for save_weights_async...")
    await save_weights_async(queue=weights_paths_queue)
    print("Save_weighs_async executed successfully!")
    # await asyncio.gather(calc_weights_async(w1=w1,w2=w2,queue=calc_weights_queue,output_queue=weights_paths_queue),copy_files_async(queue=paths_queue),save_weights_async(queue=weights_paths_queue))
                # finish queues
    print("Awaiting path_queue joins...")
    await paths_queue.join()
    print("paths_queue join awaited!")
    await calc_weights_queue.join()
    await weights_paths_queue.join()
    for c in calculators:
        c.cancel()


train_files = pd.read_csv("msd_data/splitting_edited_train_edited.csv",index_col=None)
val_files = pd.read_csv("msd_data/splitting_edited_val_edited.csv",index_col=None)
test_files = pd.read_csv("msd_data/splitting_edited_test_edited.csv",index_col=None)
train_files["folders"] = train_files["folders"].replace("^./*","msd_data/",regex=True)
val_files["folders"] = val_files["folders"].replace("^./*","msd_data/",regex=True)
test_files["folders"] = test_files["folders"].replace("^./*","msd_data/",regex=True)
combined_files = pd.concat([train_files,val_files,test_files])
destination = "./data_async"
# prep_msd_data(combined_files,destination,w1=10,w2=5)
asyncio.run(prep_msd_data_async(combined_files,destination,w1=10,w2=5))

