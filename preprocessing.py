#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import os
from pathlib import Path
from typing import Type, Union, Tuple, List

# Libs
import dask
import dask.array as da
import imagesize
from imageio.v3 import improps
import numpy as np
from osgeo import gdal
import pandas as pd
import torch
from torch.utils.data import Dataset

# Custom


##################
# Configurations #
##################


#######################################################################
# Helper Class - CustomLazyArray (class to retrieve image attributes) #
#######################################################################


class CustomLazyArray:
    """
    class containing operations to retrieve image attributes like
    height, width, channels for use later when calculating optimal chunksize

    Attributes:
        shape (Tuple[int]): Shape of ND structure
        ndim (int): No. of dimensions of ND structure
        dtype (Type): Datatypes of values within the ND structure
    """

    def __init__(self, filename: Union[Path, str], dtype: str = "uint8"):
        """constructs all neccessary attributes for CustomChunk

        Args:
            filename (Union[Path, str]): directory path to file
            dtype (str, optional): data type. Defaults to "uint8".
        """

        width, height = imagesize.get(filename)

        # from first pixel get channel info
        # need accurate channel info
        img_data = gdal.Open(filename, gdal.GA_ReadOnly)
        sample_img = img_data.ReadAsArray(float(0), float(0), 1, 1)
        if len(sample_img.shape) == 2:
            channel = 1
        else:
            channel = sample_img.shape[0]
        img_data = None

        self.shape = (height, width, channel)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)

    def __getitem__(self, val: slice):
        """Placeholder function to give custom array slicing capabilities"""
        pass


########################################################################
# Helper Class - PatchCoord (class to calculate all patch coordinates) #
########################################################################


class PatchCoord:
    """
    class containing operations to calculate all patch coordinates based on
        user-input coordinates

    Attributes:
        coord_x (int): patch top_left x-axis coordinate
        coord_y (int): patch top_left y-axis coordinate
    """

    def __init__(self, coord_x: int, coord_y: int):
        """constructs all neccessary attributes for CustomChunk

        Args:
                coord_x (int): patch top_left x-axis coordinate
                coord_y (int): patch top_left y-axis coordinate
        """
        self.coord_x = coord_x
        self.coord_y = coord_y

    def calculate_bottom_left_coord(self, end_coord: Tuple[int]) -> Tuple[int]:
        """derives bottom left coordinates based on top left and bottom right
        coordinates

        Args:
                end_coord (Tuple[int]): patch bottom_right coordinates

        Returns:
                Tuple[int]: patch bottom left coordinates
        """
        bottom_left_coord = PatchCoord(self.coord_x, end_coord.coord_y)
        return bottom_left_coord

    def calculate_top_right_coord(self, end_coord: Tuple[int]) -> Tuple[int]:
        """derives top right coordinates based on top left and bottom right
        coordinates

        Args:
                end_coord (Tuple[int]): patch top_right coordinates

        Returns:
                Tuple[int]: patch top right coordinates
        """
        top_right_coord = PatchCoord(end_coord.coord_x, self.coord_y)
        return top_right_coord


#################################################################################
# Lazy Loading Class - PatchedImage (class to lazyload and chunk up image data) #
#################################################################################


class PatchedImage:
    """
    Class containing operations to lazyload and chunk up image data

    Attributes:
        img_path (str): image file path
        image_height (int): Detected height of lazy loaded image (Default: None)
        image_width (int): Detected width of lazy loaded image (Default: None)
        chunk_size (Tuple[Tuple[int]]): Detected chunk size facilitated by Dask
            (Default: None)
        lazy_loaded_img_arr (da.Block): Lazy-loaded image chunks (Default: None)
    """

    def __init__(self, img_path: str = "empty"):
        """constructs all neccessary attributes for PatchedImage

        Args:
            img_path (str): image file path. Defaults to "empty", where
                initialized image has no information source. (Default: "empty")
        """
        self.img_path = img_path
        self.image_height = None
        self.image_width = None

        # chunk_size structure: ((3500,3500,3500,3500), (5000,5000,5000), (3,))
        self.chunk_size = None
        self.lazy_loaded_img_arr = None

    def set_chunk_size(self):
        """to get optimal chunk size using DASK"""
        chunk = CustomLazyArray(self.img_path)
        data_dask = da.from_array(chunk, chunks="auto")
        self.chunk_size = data_dask.chunks

        self.image_height = sum(self.chunk_size[0])
        self.image_width = sum(self.chunk_size[1])

    def _calculate_start_pt_chunk(self, step: Tuple[int]) -> List[int]:
        """to calculate starting point of chunk

        Args:
            step (Tuple[int]): starting point of steps

        Returns:
            Tuple[int]: starting point's pixel number
        """
        start = [0]
        start.extend(np.cumsum(step))
        start = start[:-1]
        return start

    def _calculate_chunk_count(self) -> Tuple[int, int]:
        """calculates total chunk count and chunk count for x-axis

        Returns:
            Tuple[int, int] : total chunk count, chunk count for x-axis
        """
        chunk_count_x = len(self.chunk_size[1])
        chunk_count_y = len(self.chunk_size[0])
        chunk_count = chunk_count_x * chunk_count_y
        return chunk_count, chunk_count_x

    def load(self) -> np.ndarray:
        """loads in image data using GDAL, gets optimal chunk size and
        image pixel data

        Returns:
            np.ndarray: image pixel data
        """
        img_data = gdal.Open(self.img_path, gdal.GA_ReadOnly)

        self.set_chunk_size()

        y_steps = self.chunk_size[0]
        x_steps = self.chunk_size[1]

        y_starts = self._calculate_start_pt_chunk(y_steps)
        x_starts = self._calculate_start_pt_chunk(x_steps)

        # reshape to account for grayscale images (i.e. no channel dim)
        lazy_array_List = []
        for j, (y_start, y_step) in enumerate(zip(y_starts, y_steps)):
            for i, (x_start, x_step) in enumerate(zip(x_starts, x_steps)):

                lazy_array = dask.delayed(
                    img_data.ReadAsArray(
                        float(x_start), float(y_start), x_step, y_step
                    ).reshape(
                        -1, y_step, x_step
                    )  # auto-populate channels
                )

                # create tile
                lazy_array_List.append(lazy_array)

        # close gdal dataset
        img_data = None

        ###########################################################
        # Implementation Footnote: Arranging chunk_shape elements #
        ###########################################################

        # [Cause]
        # self.chunk_size is a nested Tuple (Tuple(chunk height),
        # Tuple(chunk width), Tuple(channel))

        # [Problem]
        # nested Tuple is immutable thus unable to change order via
        # conventional methods

        # [Solution]
        # convert into List (mutable) thus able to arrange into required
        # format (channel, height, width)

        chunk_shape = [i[0] for i in self.chunk_size]
        chunk_shape = chunk_shape[-1:] + chunk_shape[:-1]

        lazy_array_List = [
            da.from_delayed(x, chunk_shape, dtype="uint8") for x in lazy_array_List
        ]

        chunk_count, chunk_count_x = self._calculate_chunk_count()

        data_index = []
        end_index_x = chunk_count - chunk_count_x + 1
        for i in range(0, end_index_x, chunk_count_x):
            # arranging into (channel count, height, width)
            # https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439
            row_i = lazy_array_List[i : i + chunk_count_x]
            data_index.append(row_i)

        self.lazy_loaded_img_arr = da.block(data_index)

        return self.lazy_loaded_img_arr

    def retrieve(
        self, top_left: PatchCoord, bottom_right: PatchCoord, selection: str = "orig"
    ) -> torch.Tensor:
        """retrieves image pixel data for specified patch based on patch
        coordinates

        Args:
            top_left (PatchCoord): patch top left coordinates
            bottom_right (PatchCoord): patch bottom right coordinates
            selection (str): the augmentation that will be performed
            "orig" - original image
            "fliph" - horizontally flipped
            "flipv" - vertically flipped
            "flipvh" - vertically + horizontally flipped

        Returns:
            torch.Tensor: patch based on selection args (i.e. original image,
            horizontally flipped image, vertically flipped image,
            vertically + horizontally flipped image)
        """
        cache = self.load()

        # to ensure user specified PATCH_SIZE =< CHUNK_SIZE
        chunk_area = self.chunk_size[0][0] * self.chunk_size[1][0]
        patch_height = bottom_right.coord_y - top_left.coord_y
        patch_width = bottom_right.coord_x - top_left.coord_x
        patch_area = patch_height * patch_width

        if patch_area > chunk_area:
            raise Exception("Please select a smaller patch_size")

        # augmentations - flip updown, flip leftright, flip udlr
        if selection == "fliph":
            cache = da.flip(cache, axis=2).copy()
        elif selection == "flipv":
            cache = da.flip(cache, axis=1).copy()
        elif selection == "flipvh":
            cache = da.flip(cache, axis=(1, 2)).copy()

        bottom_right.coord_x = min(bottom_right.coord_x, self.image_width)
        bottom_right.coord_y = min(bottom_right.coord_y, self.image_height)

        top_right = top_left.calculate_top_right_coord(bottom_right)
        bottom_left = top_left.calculate_bottom_left_coord(bottom_right)

        # create patch of zeros
        patch_frame = np.zeros(shape=(self.chunk_size[2][0], patch_height, patch_width))

        patch = cache[
            :,
            top_left.coord_y : bottom_left.coord_y,
            top_left.coord_x : top_right.coord_x,
        ].compute()

        patch_frame[
            :,
            0 : (bottom_left.coord_y - top_left.coord_y),
            0 : (top_right.coord_x - top_left.coord_x),
        ] = patch

        # to normalize images with non 0 - 255 range pixel values
        if np.array_equal(np.unique(patch_frame), [0, 1]):
            segments = len(np.unique(patch_frame))  # 256
            steps = int(255 / (segments - 1))  # 1
            normalized_img_arr = (patch_frame - patch_frame.min()) / (
                patch_frame.max() - patch_frame.min()
            )  # [0,1]
            scaled_img_arr = normalized_img_arr * steps  # element-wise #[0,1]
            patch_frame = scaled_img_arr.astype(np.uint8)

            # print(f"scaled: {np.unique(patch_frame)}")
        patch_frame = patch_frame.astype(np.uint8)
        patch_frame = torch.from_numpy(patch_frame)

        return patch_frame

    def export(self, trained: "PatchedImage", dst_filename: str) -> str:
        """writing patched image data to file using gdal

        Args:
            trained (PatchedImage): full / all patches data image (da.block)
            output by both models
            dst_filename (str): filename for output image file

        Returns:
            str: file path to output image file
        """
        # create image file (GTiff) for saving image data
        format = "GTiff"
        driver = gdal.GetDriverByName(format)

        # setting properties for image file
        ex_chunk_channels, ex_img_height, ex_img_width = trained.shape

        dst_ds = driver.Create(
            dst_filename, ex_img_width, ex_img_height, ex_chunk_channels, gdal.GDT_Byte
        )

        ex_chunk_size = trained.chunks

        y_steps = ex_chunk_size[1]  # eg. (3500,3500,3500,3500)
        x_steps = ex_chunk_size[2]  # eg. (5000,5000,5000)

        y_starts = self._calculate_start_pt_chunk(y_steps)  # eg. (0, 3500, 7000, 10500)
        x_starts = self._calculate_start_pt_chunk(x_steps)  # eg. (0, 5000, 10000)

        for j, (y_start, y_step) in enumerate(zip(y_starts, y_steps)):
            for i, (x_start, x_step) in enumerate(zip(x_starts, x_steps)):

                curr_chunk = trained[
                    # full range of channels
                    :,
                    # eg. 3500:(3500 + 3500 step)
                    y_start : (y_start + y_step),
                    # eg. 5000:(5000 + 5000 step)
                    x_start : (x_start + x_step)
                    # computing chunk by chunk
                ].compute()

                curr_col_start = x_step * i
                curr_row_start = y_step * j

                dst_ds.WriteArray(curr_chunk, curr_col_start, curr_row_start)

        # close dst_ds gdal
        dst_ds = None

        # export file path
        print("Saved file at: " + os.getcwd() + "/" + dst_filename)


#################################################################
# Lazy Loading Class - LazyDataset (retrieve data from mapping) #
#################################################################


class LazyDataset(Dataset):
    """
    class containing operations to retrieve data from filepath
    stated in given mapping

    Attributes:
        data_dir (str): Directory where information is stored
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_mappings(self) -> pd.DataFrame:
        """load given mapping.csv into dataframe for retrieval

        Returns:
            pd.DataFrame: mapping of filepath
        """
        return pd.read_csv(self.data_dir + "/mappings.csv")

    def __len__(self) -> int:
        """retrieve the number of rows of mapping

        Returns:
            int: number of rows of mapping
        """
        mappings = self.load_mappings()
        return mappings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[PatchedImage, PatchedImage]:
        """retrieves the data from respective column and filepath stated in
        mapping

        Args:
            idx (int): index number

        Returns:
            Tuple[PatchedImage, PatchedImage, json, int]: source_image,
            mask_image, annotation_json, target_value
        """
        row = self.load_mappings().iloc[idx, :]

        # retrieve source paths corresponding to the idx
        source_path = row["Sources"]
        source_image = PatchedImage(source_path)

        # retrieve mask paths corresponding to the 9o9⁹99idx
        mask_path = row["Masks"]
        mask_image = PatchedImage(mask_path)

        # # retrieve annotation paths corresponding to the idx
        # annotation_path = row["Annotations"]
        # with open(annotation_path, 'r') as json_file:
        #     annotation_json = json.load(json_file)

        # # retrieve Target value corresponding to the idx
        # target_value = row["Target"]

        return source_image, mask_image
        # return source_image, mask_image, annotation_json, target_value


class BoundaryDataset:
    pass


###############################################################################
#
#   This is a placeholder to for simulation. To be deleted
#
###############################################################################


class TrainPredLazyDataset(Dataset):
    """
    class containing operations to retrieve data from filepath
    stated in given mapping

    Attributes:
        data_dir (str): Directory where information is stored
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_mappings(self) -> pd.DataFrame:
        """load given mapping.csv into dataframe for retrieval

        Returns:
            pd.DataFrame: mapping of filepath
        """
        return pd.read_csv(self.data_dir + "/train_pred_mapping.csv")

    def __len__(self) -> int:
        """retrieve the number of rows of mapping

        Returns:
            int: number of rows of mapping
        """
        mappings = self.load_mappings()
        return mappings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[PatchedImage, PatchedImage]:
        """retrieves the data from respective column and filepath stated in
        mapping

        Args:
            idx (int): index number

        Returns:
            Tuple[PatchedImage, PatchedImage, json, int]: source_image,
            mask_image, annotation_json, target_value
        """
        row = self.load_mappings().iloc[idx, :]

        # retrieve source paths corresponding to the idx
        source_path = row["Sources"]
        source_image = PatchedImage(source_path)

        # retrieve mask paths corresponding to the 9o9⁹99idx
        mask_path = row["Masks"]
        mask_image = PatchedImage(mask_path)

        # # retrieve annotation paths corresponding to the idx
        # annotation_path = row["Annotations"]
        # with open(annotation_path, 'r') as json_file:
        #     annotation_json = json.load(json_file)

        # # retrieve Target value corresponding to the idx
        # target_value = row["Target"]

        return mask_image
        # return source_image, mask_image, annotation_json, target_value


class TestPredLazyDataset(Dataset):
    """
    class containing operations to retrieve data from filepath
    stated in given mapping

    Attributes:
        data_dir (str): Directory where information is stored
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_mappings(self) -> pd.DataFrame:
        """load given mapping.csv into dataframe for retrieval

        Returns:
            pd.DataFrame: mapping of filepath
        """
        return pd.read_csv(self.data_dir + "/predict_mapping.csv")

    def __len__(self) -> int:
        """retrieve the number of rows of mapping

        Returns:
            int: number of rows of mapping
        """
        mappings = self.load_mappings()
        return mappings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[PatchedImage, PatchedImage]:
        """retrieves the data from respective column and filepath stated in
        mapping

        Args:
            idx (int): index number

        Returns:
            Tuple[PatchedImage, PatchedImage, json, int]: source_image,
            mask_image, annotation_json, target_value
        """
        row = self.load_mappings().iloc[idx, :]

        # retrieve source paths corresponding to the idx
        source_path = row["Sources"]
        source_image = PatchedImage(source_path)

        # retrieve mask paths corresponding to the 9o9⁹99idx
        mask_path = row["Masks"]
        mask_image = PatchedImage(mask_path)

        # # retrieve annotation paths corresponding to the idx
        # annotation_path = row["Annotations"]
        # with open(annotation_path, 'r') as json_file:
        #     annotation_json = json.load(json_file)

        # # retrieve Target value corresponding to the idx
        # target_value = row["Target"]

        return source_image, mask_image
        # return source_image, mask_image, annotation_json, target_value


class PlaceHolderLazyDataset(Dataset):
    """
    class containing operations to retrieve data from filepath
    stated in given mapping

    Attributes:
        data_dir (str): Directory where information is stored
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_mappings(self) -> pd.DataFrame:
        """load given mapping.csv into dataframe for retrieval

        Returns:
            pd.DataFrame: mapping of filepath
        """
        return pd.read_csv(self.data_dir + "/placeholder_mapping.csv")

    def __len__(self) -> int:
        """retrieve the number of rows of mapping

        Returns:
            int: number of rows of mapping
        """
        mappings = self.load_mappings()
        return mappings.shape[0]

    def __getitem__(self, idx: int) -> Tuple[PatchedImage, PatchedImage]:
        """retrieves the data from respective column and filepath stated in
        mapping

        Args:
            idx (int): index number

        Returns:
            Tuple[PatchedImage, PatchedImage, json, int]: source_image,
            mask_image, annotation_json, target_value
        """
        row = self.load_mappings().iloc[idx, :]

        # retrieve source paths corresponding to the idx
        source_path = row["Sources"]
        source_image = PatchedImage(source_path)

        # retrieve mask paths corresponding to the 9o9⁹99idx
        mask_path = row["Masks"]
        mask_image = PatchedImage(mask_path)

        # # retrieve annotation paths corresponding to the idx
        # annotation_path = row["Annotations"]
        # with open(annotation_path, 'r') as json_file:
        #     annotation_json = json.load(json_file)

        # # retrieve Target value corresponding to the idx
        # target_value = row["Target"]

        return mask_image
        # return source_image, mask_image, annotation_json, target_value
