####################
# REQURIED MODULES #
####################

# Libs
from distutils.log import error
import matplotlib.pyplot as plt
import PIL.Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms


###################
# Tensor  dataset #
###################



# # #############################################
# # # StackedDataset - for 1 channel simulation #
# # #############################################

# class StackedDataset(Dataset):
#     """
#         Stacked dataset functions to return size of dataset and reads image from filenames in filelist as torch dataset object

#     Attributes:
#         root_dir (str): Directory with all the images
#         fileList (list): list containing filenames of images
#         cfg (dict): Dictionary of configs
#         transformer (function): (1) converts to Grayscale and tensor
#                                 (2) converts to tensor 

#     """
#     def __init__(self, cfg: dict, fileList: list,root_dir: str,transform =None):
#         #load dataset from root dir
#         self.root_dir = root_dir
#         self.fileList = fileList
#         self.cfg = cfg  
#         if transform is None:
#             self.transform = transforms.Compose([ transforms.Grayscale(), 
#                                                   transforms.ToTensor(),])
#             self.transform2 = transforms.Compose([transforms.ToTensor(),])
    
#     ###################
#     #  CORE FUNCTIONS #
#     ###################

#     def __len__(self) -> str:
#         """Returns size of dataset

#         Returns:
#             length of fileList: str
#         """
#         return len(self.fileList)

#     def __getitem__(self,index: str) -> object:
#         """Memory efficient function for reading filenames from filelist (image, mask, groundtruth) and converts to image and subsequently to tensors. Mask and image tensors and stacked to become 4 channel image
        
#         Args:
#             index (str): for reading filename in order of index

#         Returns:
#            stack, gt (object): stack - 4 channel (image and mask)
#                              : gt - 1 channel (groundtruth)
#         """
#         filename = self.fileList[index]
#         image = PIL.Image.open(self.root_dir+self.cfg['IMAGE_PATH']+filename).convert('RGB')
#         mask = PIL.Image.open(self.root_dir+self.cfg['MASK_PATH']+filename)
#         gt = PIL.Image.open(self.root_dir+ self.cfg['GT_PATH']+filename)
#         image = self.transform2(image)
#         mask  = self.transform(mask)
#         gt  = self.transform(gt)
#         stack = torch.vstack((image, mask))
#         return stack,gt


# ############################################
# # TrainDataset - for 1 channel simulation #
# ############################################

# class TrainDataset(Dataset):
#     """
#         Function to obtain train dataset for 1 channel processing. Return size of dataset and reads image from filenames in filelist as torch dataset object

#     Attributes:
#         root_dir (str): Directory with all the images
#         fileList (list): list containing filenames of images
#         cfg (dict): Dictionary of configs
#         transformer (function): (1) converts to Grayscale and tensor
#                                 (2) converts to tensor 

#     """
#     def __init__(self, cfg: dict, fileList: list,root_dir: str,transform=None):
#         #load dataset from root dir
#         self.root_dir = root_dir
#         self.fileList = fileList
#         self.cfg = cfg      
#         if transform is None:
#             self.transfrom = transforms.Compose([transforms.Grayscale(),
#                                                   transforms.ToTensor(),])
    
#     ###################
#     #  CORE FUNCTIONS #
#     ###################

#     def __len__(self) -> str:
#         """Returns size of dataset

#         Returns:
#             length of fileList (str)
#         """
#         return len(self.fileList)

#     def __getitem__(self,index: str) -> object:
#         """Memory efficient function for reading filenames from filelist (image, mask) and converts to image and subsequently to tensors.
        
#         Args:
#             index (str): for reading filename in order of index

#         Returns:
#            image, mask (object): image - 1 channel (predicted mask)
#                                : mask - 1 channel (groundtruth)
#         """
#         filename = self.fileList[index]
#         image = PIL.Image.open(self.root_dir+self.cfg['MASK_PATH']+filename)
#         mask = PIL.Image.open(self.root_dir+self.cfg['GT_PATH']+filename)
        
#         image = self.transfrom(image)
#         mask  = self.transfrom(mask)
#         return image,mask

        
# ###########################################
# # TestDataset - for 1 channel simulation #
# ###########################################

# class TestDataset(Dataset):
#     """
#         Function to obtain test dataset for 1 channel processing. Return size of dataset and reads image from filenames in filelist as torch dataset object

#     Attributes:
#         root_dir (str): Directory with all the images
#         imageList (list): list containing filenames of images
#         maskList (list): list containing filenames of mask
#         cfg (dict): Dictionary of configs
#         transformer (function): (1) converts to Grayscale and tensor
#                                 (2) converts to tensor 
#     """
#     def __init__(self,cfg: dict, imageList: list, maskList: list, root_dir: str, transform=None):
#         #load dataset from root dir
#         self.root_dir = root_dir
#         self.imageList = imageList
#         self.maskList = maskList
#         self.cfg = cfg  
#         # transforms.Grayscale(),
#         if transform is None:
#             self.transfrom = transforms.Compose([ transforms.Grayscale(),
#                                                   transforms.ToTensor(),])

#     ###################
#     #  CORE FUNCTIONS #
#     ###################
#     def __len__(self) -> str:
#         """Returns size of dataset

#         Returns:
#             length of fileList: str
#         """
#         return len(self.imageList) + len(self.maskList)

#     def __getitem__(self,index:str) -> object:
#         """Memory efficient function for reading filenames from imagelist and masklist (image, mask) and converts to image and subsequently to tensors.
        
#         Args:
#             index (str): for reading filename in order of index

#         Returns:
#            image, mask (object): image - 1 channel (predicted mask)
#                                : mask - 1 channel (groundtruth)
#         """
#         imagename = self.imageList[index]
#         maskname = self.maskList[index]
#         image = PIL.Image.open(self.root_dir+ self.cfg['MASK_PATH']+imagename)
#         mask = PIL.Image.open(self.root_dir+self.cfg['GT_PATH']+maskname)
#         image = self.transfrom(image)
#         mask  = self.transfrom(mask)
#         return image, mask

# ##############
# # Dataloading #
# ##############

# class Dataloading():
#     """
#         Contains dataloading functionality- gets train and validation         dataset from combined dataset and plots out samples from datasets.

#     Attributes:
#         dataset (object): torch object obtained from torch dataset class function
        
#     """
#     def __init__(self, dataset: object) -> torch.utils.data.dataset:
#         # General attributes
#         self.dataset = dataset
    
#     ###########
#     # HELPERS #
#     ###########
    
#     def show_dataset(dataset: object,n_sample: int = 4):
#         """Function to show sample images from dataset

#         Args:
#             n_sample (int, optional): no. of images to return. Defaults to 4.
#         """
#         # show image
#         for i in range(n_sample):
#             image, mask = dataset[i]
#             image = transforms.ToPILImage()(image)
#             mask = transforms.ToPILImage()(mask)
#             print(i, image.size, mask.size)


#             plt.tight_layout()
#             ax = plt.subplot(1, n_sample, i + 1)
#             ax.set_title('Sample #{}'.format(i))
#             ax.axis('off')

#             # plt.imshow(image, cmap="Greys")
#             plt.imshow(mask,alpha = 0.3, cmap="Greys")

#             if i == n_sample-1:
#                 plt.show()
#                 break

    ###################
    # CORE FUNCTIONS #
    ###################
    
def get_dataset(dataset: torch.utils.data.dataset, random_seed: int, valid_ratio: int, train_ratio: int) -> tuple[torch.utils.data.dataset,torch.utils.data.dataset, torch.utils.data.dataset]:
    """Splits torch dataset into train and validation dataset

    Args:
        random_seed: fix seed for reproducibility
        valid_ratio: percentage split of the training set used for
        the validation set. Should be a float in the range [0, 1].
        train_ratio: percentage split of the training set used for
        the training set. Should be a float in the range [0, 1].

    Returns:
        train_dataset (torch.utils.data.dataset)
        valid_dataset (torch.utils.data.dataset)
        test_dataset (torch.utils.data.dataset)
    """
    error_msg = "[!] valid_ratio or train_ratio should be in the range [0, 1]."
    if valid_ratio<=0 or valid_ratio>=1 or train_ratio<=0 or train_ratio>=1 :
        raise ValueError(error_msg)
    else:
        pass
    if (valid_ratio + train_ratio)>1:
        raise ValueError("sum of valid ratio and train ratio must be less than 1")
    else:
        pass
    n = len(dataset)
    n_valid = int(valid_ratio*n)
    n_train = int(train_ratio*n)
    n_test = n-n_valid-n_train
    seed = torch.Generator().manual_seed(random_seed)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (n_train, n_valid, n_test), generator = seed)

    # train_dataset,valid_dataset = random_split(dataset,(n_train,n_valid), generator=seed)
    # train_dataset,test_dataset = random_split(dataset,(n_train,n_valid), generator=seed)
    return train_dataset,valid_dataset, test_dataset
