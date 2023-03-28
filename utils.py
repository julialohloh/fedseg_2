import os
import numpy as np
import shutil

# root_path = os.path.join(os.getcwd(), 'data/CHASEDB1')
# folders = ['left','right','L_1stHO','R_1stHO','L_2ndHO','R_2ndHO']

def set_seed(seed: int):
    """This function sets the seed for os, random and numpy

    Args:
        seed (int): The integer to set the seed to.
        
    Returns:
        does not return anything 
    """
    
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)


def stratify_folders(data_path: str, stratify_path: str, folders: list):
    """This function stratifies the chase dataset into folders accordingly

    Args:
        data_path (path): This is the data folder source path
        stratify_path (path): This is the folder to copy the data to in it's
            stratified form.
        folders (list): This is a list of str of folder names for each stratified
            categories.

    Returns:
        does not return anything  
    """

    for folder in folders:
        if not os.path.exists(os.path.join(stratify_path, folder)):
            os.makedirs(os.path.join(stratify_path, folder))

    src = os.path.join(data_path)

    for dirs, subdirs, files in os.walk(src):
        for file in files:
            filename = os.path.join(src, file)
            if os.path.exists(filename):
                if file.endswith(".jpg"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[0]))
                if file.endswith("L.jpg"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[1]))
                elif file.endswith("R.jpg"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[2]))
                elif file.endswith("L_1stHO.png"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[3]))
                elif file.endswith("R_1stHO.png"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[4]))
                elif file.endswith("L_2ndHO.png"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[5]))
                elif file.endswith("R_2ndHO.png"):
                    shutil.copy(filename,os.path.join(stratify_path, folders[6]))


def get_folders_path(parent_path: str)-> list:
    """This function does an os.listdir to find all the folders name and returns its path.

    Args:
        parent_path (path): This is the parent folder path to look through.
    
    Returns:
        folders_path (list): A list of folders path.

    """

    folders = sorted(os.listdir(parent_path))

    folders_path = []

    for folder in folders:
        folder_path = os.path.join(parent_path, folder)
        folders_path.append(folder_path)

    return folders_path


def get_files_path(folder_path: str)-> list:
    """This function does an os.listdir to find all the files name and returns its path.

    Args:
        folder_path: This is the folder path contain the files to look through.

    Returns:
        file_paths: A list of file paths
    """

    filenames = sorted(os.listdir(folder_path))

    file_paths = []

    for file in filenames:
        file_path = os.path.join(folder_path, file)
        file_paths.append(file_path)

    return file_paths


def create_image_mask_dict(image_paths: list, mask_paths: list)-> dict:
    """This takes in a list of images path and masks path and creates a dictionary from it.

    Args:
        image_paths: This is a list of image paths
        mask_paths: This is a list of corresponding mask paths.
    
    Returns:
        image_mask_dict: This is a dictionary with image path as key and mask path
            as value.
    """

    image_mask_dict = {}

    for i in range(len(image_paths)):
        image_mask_dict.update({image_paths[i]:mask_paths[i]})

    return image_mask_dict


def random_idx_split(index_length: int, number_of_splits: int)-> list:
    """This function split a np.arange arr into sub-arrays with random shuffling

    Args:
        index_length: This is the value to create the larger np.arange array
        number_of_splits: This is the number of sub-arrays to create
    Returns:
        a: This is a list of the sub-arrays
    """
    
    a = np.arange(index_length)
    np.random.shuffle(a)
    a = np.array_split(a, number_of_splits)
    
    return a


def create_partitions(image_mask_dict: dict, 
                        partition_path: str, 
                        number_of_partitions: int):
    """This functions create partitions for the chase dataset

    Args:
        image_mask_dict: This is a dictionary of the image path(key) 
            and mask path(value)
        partition_path: This is a string of the partition folder path.
        number_of_partitions: This is the number of partition to create

    Returns:
        prints completed when done. 
    """
    
    if not os.path.exists(os.path.join(partition_path)):
        for i in range(number_of_partitions):
            os.makedirs(os.path.join(partition_path, 'images', str(i+1)))
            os.makedirs(os.path.join(partition_path, 'mask', str(i+1)))
            
    # create a list of partitions path
    image_partitions = []
    mask_partitions = []
    
    for i in range(number_of_partitions):
        path_1 = os.path.join(partition_path, 'images', str(i+1))
        path_2 = os.path.join(partition_path, 'mask', str(i+1))
        image_partitions.append(path_1)
        mask_partitions.append(path_2)
        
    # use internal function to create random list index
    idx = random_idx_split(len(image_mask_dict), number_of_partitions)
    
    key_list = list(image_mask_dict.keys())
        
    # # partition_paths[0], idx[0]
    for i in range(len(idx)):
        for indice in idx[i]:
            
            image_path = key_list[indice]
            mask_path = image_mask_dict[image_path]
            
            image_name = image_path.split('/')
            mask_name = mask_path.split('/')
            
            shutil.copy(os.path.join(os.getcwd(),image_path), 
                        os.path.join(image_partitions[i], image_name[-1]))

            shutil.copy(os.path.join(os.getcwd(),mask_path), 
                        os.path.join(mask_partitions[i], mask_name[-1]))
            
    return print('Completed')    
