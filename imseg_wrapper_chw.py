#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import math
from typing import Callable, Generator, List, TypeVar, Tuple
import itertools

# Libs
# import dask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorly.tenalg import tensor_dot
from torch.utils.checkpoint import checkpoint_sequential
import tntorch as tnt
from torch.utils.data import DataLoader

# Custom
from unet.unet import UNet
from autoencoder.cnnautoencoder_asym import AE
from preprocessing import PatchedImage, PatchCoord

##################
# Configurations #
##################

FedSegType = TypeVar("FedSegType", bound="Memoize")

#################################
# IMSEG Helper Class - Memorize #
#################################


class Memoize:
    """
    Wraps around factory function to store prior dynamic class creations. This
    reduces the generation of multiple "same" classes being considered as
    unique.

    References:
    - https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python
    - https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python

    Attributes:
        f (Callable): Factory function to apply memorization to
        memo (dict): Cache to store wrapped factory function uniquely
    """

    def __init__(self, f: Callable):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        return self.memo.setdefault(args, self.f(*args))


@Memoize
def apply_patch_wrapper(model: torch.nn.Module) -> FedSegType:
    """Factory generator for dynamically creating wrapped models compatible
    with patch-based workflows for federated image segmentation
    """

    class FedSeg(model):
        ...

    return FedSeg


# How to use:
# model = apply_patch_wrapper(UNet)

##########################
# IMSEG Wrapper - FedSeg #
##########################


# main class code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FedSeg(
    UNet
):  # accepts Unet during runtime (create factory --> create classes during runtime)(metaprogramming)
    def __init__(
        self,
        unet_channels: int,
        unet_classes: int,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        AE: torch.nn.Module,
        AD: torch.nn.Module,
        unet: UNet,
        unet_input: int = 32,
        map_type: str = "intensity",
        tensor_batch_size: int = 500,
        tt_decomp_max_rank: int = 3,
        tt_threshold: int = 8,
    ):
        super(FedSeg, self).__init__(unet_channels, unet_classes)
        super().float()  # <-- what's this for? If it is for casting weights to float, why is it here inside the model?
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.tt_decomp_max_rank = tt_decomp_max_rank
        self.tensor_batch_size = tensor_batch_size

        self.autoencoder = None
        self.unet_channels = unet_channels
        self.unet_classes = unet_classes
        self.unet_input = unet_input
        self.unet = unet(self.unet_channels, self.unet_classes)
        self.autodecoder = None
        self.map_type = map_type
        self.tt_threshold = tt_threshold

    ###########
    # Setters #
    ###########
    @property
    def tt_decomp_max_rank(self):
        return self._tt_decomp_max_rank

    @property
    def patch_height(self):
        return self._patch_height

    @property
    def patch_width(self):
        return self._patch_width

    @property
    def tt_threshold(self):
        return self._tt_threshold

    @property
    def unet_channels(self):
        return self._unet_channels

    @property
    def unet_classes(self):
        return self._unet_classes

    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def unet_input(self):
        return self._unet_input

    @property
    def map_type(self):
        return self._map_type

    @property
    def tensor_batch_size(self):
        return self._tensor_batch_size

    @tt_decomp_max_rank.setter
    def tt_decomp_max_rank(self, tt_decomp_max_rank):
        if tt_decomp_max_rank > 0:
            self._tt_decomp_max_rank = tt_decomp_max_rank
        else:
            raise ValueError("tt_decomp_max_rank must be more than zero")

    @patch_height.setter
    def patch_height(self, patch_height):
        if patch_height <= self.image_height and patch_height > 0:
            self._patch_height = patch_height
        elif patch_height <= 0:
            raise ValueError("patch_height must be more than zero")
        else:
            raise ValueError("patch_height must be smaller than image height")

    @patch_width.setter
    def patch_width(self, patch_width):
        if patch_width <= self.image_width and patch_width > 0:
            self._patch_width = patch_width
        elif patch_width <= 0:
            raise ValueError("patch_width must be more than zero")
        else:
            raise ValueError("patch_width must be smaller than image width")

    @tt_threshold.setter
    def tt_threshold(self, tt_threshold):
        if tt_threshold > 3 and tt_threshold <= 100:
            self._tt_threshold = tt_threshold
        else:
            raise ValueError("tt_threshold must be between 4 and 100")

    @unet_input.setter
    def unet_input(self, unet_input):
        if unet_input >= 32:
            self._unet_input = unet_input
        else:
            raise ValueError("unet_input must be 32 and more")

    @map_type.setter
    def map_type(self, map_type):
        if map_type in ["intensity", "fourier", "sinusoidal"]:
            self._map_type = map_type
        else:
            raise ValueError("map_type must be 'intensity', 'fourier' or 'sinusoidal'")

    @tensor_batch_size.setter
    def tensor_batch_size(self, tensor_batch_size):
        if tensor_batch_size > 0:
            self._tensor_batch_size = tensor_batch_size
        else:
            raise ValueError("tensor_batch_size must be more than zero")

    @image_height.setter
    def image_height(self, image_height):
        if image_height > 0:
            self._image_height = image_height
        else:
            raise ValueError("image_height must be more than zero")

    @image_width.setter
    def image_width(self, image_width):
        if image_width > 0:
            self._image_width = image_width
        else:
            raise ValueError("image_width must be more than zero")

    @unet_channels.setter
    def unet_channels(self, unet_channels):
        if unet_channels == 1 or unet_channels == 3:
            self._unet_channels = unet_channels
        else:
            raise ValueError("unet_channels must be either 1 or 3")

    @unet_classes.setter
    def unet_classes(self, unet_classes):
        if unet_classes > 0:
            self._unet_classes = unet_classes
        else:
            raise ValueError("unet_classes must be more than zero")

    ###########
    # Helpers #
    ###########

    def _perform_patching(self, image: PatchedImage) -> List[PatchCoord]:
        """Helper function that calculates the patch points from a PatchedImage based on
        given image height and image width

        Args:
            image (PatchedImage): image to be passed to forward function
            image_height (int): height of forwarded image
            image_width (int): width of forwarded image

        Returns:
            List[PatchCoord]: list of patch coordinates of image
        """

        patch_points = []
        num_patch_height = math.ceil(self.image_height / self.patch_height)
        num_patch_width = math.ceil(self.image_width / self.patch_width)
        # max_height_idx = self.image_height
        # max_width_idx = self.image_width

        for i in range(num_patch_height):
            for j in range(num_patch_width):

                patch_row_start_idx = i * self.patch_height
                patch_col_start_idx = j * self.patch_width

                patch_row_end_idx = patch_row_start_idx + self.patch_height

                patch_col_end_idx = patch_col_start_idx + self.patch_width

                patch_points.append(
                    [
                        PatchCoord(patch_col_start_idx, patch_row_start_idx),
                        PatchCoord(patch_col_end_idx, patch_row_end_idx),
                    ]
                )
        print(f"NUM PATCH HEIGHT IS {num_patch_height} AND NUM PATCH WIDTH IS {num_patch_width}")
        print(f"TOTAL NUM OF PATCHES IS {len(patch_points)}")
        return patch_points

    def load_patches(
        self, image: PatchedImage
    ) -> Generator[torch.Tensor, None, None]:  # to check
        """Loads a generator of patches based on given PatchedImage using _perform_patching
        helper function

        Args:
            image (PatchedImage): image to perform patching on

        Yields:
            Generator[torch.Tensor, None, None]: a generator of patches
        """
        self.patch_coords = self._perform_patching(image)
        for paired_points in self.patch_coords:
            yield image.retrieve(*paired_points)

    def images(
        self, dataloader: DataLoader
    ) -> Generator[Tuple[PatchedImage, PatchedImage], None, None]:
        """Loads in source and mask from dataloader as a generator

        Args:
            dataloader (DataLoader): a dataloader

        Yields:
            Generator[Tuple[PatchedImage, PatchedImage], None, None]: a generator of source and mask
        """
        for batch in dataloader:
            for source, mask in batch:
                yield (source, mask)

    def convert_to_1d(self, patch: torch.Tensor) -> torch.Tensor:
        """Takes an image patch of shape(C,H,W) and ravels it .

        Args:
            patch (torch.Tensor): tensor of pixel values of 1 image patch

        Returns:
            torch.Tensor: tensor of pixels obtained by reshaping the input patch
        """
        patch_channels = patch.shape[0]
        # patch_channels = patch.shape[2]
        total_pixels = self.patch_height * self.patch_width
        return patch.reshape(total_pixels, patch_channels)

    def _calculate_fourier_transform(self, pixels: np.ndarray) -> np.ndarray:
        """ """
        raise NotImplementedError

    def _calculate_intensity_transform(self, pixels: torch.Tensor) -> torch.Tensor:
        """Transforms specified pixels by executing vectorized math operations
            corresponding to intensity transform introduced in Efthymiou et al.
            (2019).

        Args:
            pixles (torch.Tensor): 1d (3) tensor of a pixel

        Returns:
            torch.Tensor: 2d (2,3) tensor

        """
        normalized_pixels = pixels / 255
        inverted_pixels = 1 - normalized_pixels
        transformed_pixels = torch.stack((normalized_pixels, inverted_pixels), axis=1)
        return transformed_pixels

    def _calculate_sinusoidal_transform(self, pixels: np.ndarray) -> np.ndarray:
        """ """
        raise NotImplementedError

    def calculate_local_maps(
        self, pixels: torch.Tensor, map_type: str = "intensity"
    ) -> Generator[torch.Tensor, None, None]:
        """Computes the local map of individual pixels based on selected map_type.

        Args:
            pixels (torch.Tensor): 1d (3) tensor of a pixel
            map_type (str, optional): Choose from "intensity", "fourier" and "sinusoidal"
            map types. Defaults to "intensity".

        Yields:
            Generator[torch.Tensor, None, None]: A generator of local_maps. Each local
            map is a 2d tensor.
        """
        MAPPING_FUNCTIONS = {
            "intensity": self._calculate_intensity_transform,
            "fourier": self._calculate_fourier_transform,
            "sinusoidal": self._calculate_sinusoidal_transform,
        }
        local_maps = MAPPING_FUNCTIONS.get(map_type)(pixels)

        for local_map in local_maps:
            yield local_map

    def _iter_to_generator(self, iterable: list) -> Generator:
        """Converts an interable to a generator.

        Returns:
            Generator: A generator of elemnets of the given list.
        """
        return (i for i in iterable)

    def tt_decomp(self, node: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        """Conducts tensor train decomposition of a tensor into a list of tensors.

        Args:
            node (torch.Tensor): Global maps to decompose 3d tensor train arrays.

        Yields:
            Generator[torch.Tensor, None, None]: A generator of tensors from tensor train decomposition.
        """
        tt = tnt.Tensor(node, ranks_tt=self.tt_decomp_max_rank)
        tt_nodes = tt.cores
        for tt_node in tt_nodes:
            yield tt_node

    def calculate_global_map(
        self, local_maps: Generator[torch.Tensor, None, None]
    ) -> Generator[torch.Tensor, None, None]:
        """Calculates global map by multiplying each local map to higher dimensions.
        Global map undergoes tensor train decomposition when the dimension of global map
        is bigger or equal to the tt_threshold. Process repeats until all local maps have
        been multiplied.

        Args:
            local_maps (Generator[torch.Tensor, None, None]): Generates tensors of local_maps
            tt_threshold (int, optional): Maximum dimension allowed for global map
            before decomposition. Defaults to 8.

        Returns:
            Generator[torch.Tensor, None, None]: A generator of decomposed global maps
        """

        curr_local_maps = self._iter_to_generator([next(local_maps)])
        counter = 0
        for next_map in local_maps:
            next_node = next_map
            curr_maps = (
                tensor_dot(local_map, next_node) for local_map in curr_local_maps
            )

            decomposed_maps = self._iter_to_generator([])
            sub_counter = 0
            for curr_map in curr_maps:

                map_dim = len(curr_map.shape)
                if map_dim >= self.tt_threshold:

                    tt_maps = self.tt_decomp(curr_map)
                    decomposed_maps = itertools.chain(decomposed_maps, tt_maps)
                    sub_counter += 1
                else:
                    decomposed_maps = itertools.chain(
                        self._iter_to_generator([curr_map]), curr_maps
                    )
                    break

            curr_local_maps = decomposed_maps
            counter += 1

        return curr_local_maps

    def convert_to_tt(
        self, tensor_list: Generator[torch.Tensor, None, None]
    ) -> Generator[torch.Tensor, None, None]:
        """Converts final global maps into 3d arrays to fit into forward function of
        the model.

        Args:
            tensor_list (Generator[torch.Tensor, None, None]): A generator of tensors of decomposed global maps

        Returns:
            Generator: A generator of 3d tensors
        """
        decomposed_maps = self._iter_to_generator([])

        for tensor in tensor_list:
            map_dim = len(tensor.shape)
            if map_dim >= 4:
                tt_maps = self.tt_decomp(tensor)
                decomposed_maps = itertools.chain(decomposed_maps, tt_maps)
            else:
                decomposed_maps = itertools.chain(
                    self._iter_to_generator([tensor]), tensor_list
                )
                break
        return decomposed_maps

    def pad_tensors(
        self, tensors: Generator[torch.Tensor, None, None]
    ) -> Generator[torch.tensor, None, None]:  # to check
        """Pads tensors to given dimensions to fit encoder input shape.

        Args:
            tensors (Generator[torch.Tensor, None, None]): A generator of 2d array.
            dimensions (Tuple[int,int,int]): A tuple of dimensions to pad array to.

        Yields:
            Generator[torch.tensor, None, None]: A generator that yields padded tensors of given dimensions.
        """

        difference = [0, 0, 0]
        for tensor in tensors:
            # Get amount to pad
            for i in range(len(tensor.shape)):
                difference[i] = self.tt_decomp_max_rank - tensor.shape[i]
                if difference[i] < 0:
                    raise ValueError(
                        "tt_decomp_max_rank must be bigger than tensor shape"
                    )
                else:
                    pass

            padded_tensor = F.pad(
                torch.tensor(tensor),
                (difference[2], 0, difference[1], 0, difference[0], 0),
                "constant",
                0,
            )
            yield padded_tensor

    def create_adapter_layers(self) -> None:
        """
        Initialises autoencoder:
        Takes a padded tensor of (H,W,C), num_output_features -> the number of features passed to the user defined model

        Initialises autodecoder:
        Decodes back to the original patch height
        num_output_features: Number of channels in the original image


        """

        if self.autoencoder == None:
            self.autoencoder = AE(
                input_height=self.tt_decomp_max_rank,
                input_width=self.tt_decomp_max_rank,
                encoded_size=2,
                decoded_size=self.unet_input,
                in_channels=self.tt_decomp_max_rank,
                out_channels=self.tt_decomp_max_rank,
                num_output_features=self.unet_channels,
                kernel_size=(3, 3),
            )
            self.autoencoder.to(device)
            self.unet.to(device)

        if self.autodecoder == None:
            self.autodecoder = AE(
                input_height=self.unet_input,
                input_width=self.unet_input,
                encoded_size=2,
                decoded_size=self.patch_height,
                in_channels=self.unet_classes,
                out_channels=self.unet_classes,
                num_output_features=self.unet_classes,
                kernel_size=(3, 3),
            )
            self.autodecoder.to(device)

    def _batch_tensors(
        self, tensors: Generator[torch.tensor, None, None], batch_size: int
    ) -> Generator[torch.tensor, None, None]:
        """Takes in a generator of tensors and outputs them in desired batch sizes.

        Args:
            tensors (Generator[torch.tensor, None, None]): A generator of tensors
            batch_size (int): Number of tensors in a batch

        Yields:
            Generator[torch.tensor, None, None]: A generator of tensors in batches.
        """
        batch_counter = 0
        batched_tensors = []

        for tensor in tensors:
            batched_tensors.append(tensor)
            batch_counter += 1
            if batch_counter == batch_size:
                yield torch.stack(batched_tensors)
                batched_tensors = []
                batch_counter = 0
        # yield the remainder
        if len(batched_tensors) != 0:
            yield torch.stack(batched_tensors)

    def _forward_padded_tensor(
        self, tensors: Generator[torch.tensor, None, None]
    ) -> Generator[torch.tensor, None, None]:
        """Forwards the padded tensors through the autoencoder, unet and autodecoder 1 at a time.

        Args:
            batched_tensors (Generator[torch.tensor, None, None]): A generator of batch of padded tensors

        Yields:
            Generator[torch.tensor, None, None]: A generator of decoded tensors
        """
        self.create_adapter_layers()

        tensor_count = 0

        batched_tensors = self._batch_tensors(
            tensors, batch_size=self.tensor_batch_size
        )
        for batch_tensor in batched_tensors:
            tensor_count += 1
            if not tensor_count % 10:
                print("TENSOR COUNT: ", tensor_count)

            float_tensor = batch_tensor.float().detach()
            print(f"forward_padded_tensor - float_tensor shape: {float_tensor.shape}")
            # permuted_tensor = torch.permute(float_tensor, (0, 3, 1, 2)).detach()
            # permuted_tensor_gpu = torch.tensor(permuted_tensor, requires_grad=True).to(
                # device
            # )
            float_tensor_gpu = torch.tensor(float_tensor, requires_grad=True).to(
                device
            )

            # encoder_logits = self.autoencoder.forward(permuted_tensor_gpu)
            encoder_logits = self.autoencoder.forward(float_tensor_gpu)

            unet_logits = self.unet(encoder_logits)
            unet_logits_grad = torch.tensor(unet_logits, requires_grad=True)

            autodecoder_modules = [
                module for k, module in self.autodecoder._modules.items()
            ]
            autodecoder_segments = len(autodecoder_modules)
            decoder_logits = checkpoint_sequential(
                autodecoder_modules, autodecoder_segments, unet_logits_grad
            )

            del float_tensor_gpu
            del encoder_logits
            del unet_logits
            del unet_logits_grad
            yield decoder_logits
            del decoder_logits

    def _forward_padded_tensor_full(
        self, tensors: Generator[torch.tensor, None, None]
    ) -> Generator[torch.tensor, None, None]:
        """Forwards the padded tensors through the autoencoder, unet and autodecoder all at once.

        Args:
            tensors (Generator[torch.tensor, None, None]): A generator of batch of padded tensors

        Yields:
            Generator[torch.tensor, None, None]: A generator of decoded tensors
        """

        self.create_adapter_layers()

        batched_tensors = self._batch_tensors(
            tensors, batch_size=self.tensor_batch_size
        )
        print("Running forward padded tensor full")
        float_tensors = (
            batch_tensor.float().detach() for batch_tensor in batched_tensors
        )
        print(f"forward_padded_tensor_full - float_tensor shape: {float_tensors.shape}")
        # permuted_tensors = (
        #     torch.permute(float_tensor, (0, 3, 1, 2)).detach()
        #     for float_tensor in float_tensors
        # )
        # encoder_logits = (
        #     self.autoencoder.forward(permuted_tensor.to(device))
        #     for permuted_tensor in permuted_tensors
        # )
        encoder_logits = (
            self.autoencoder.forward(float_tensor.to(device))
            for float_tensor in float_tensors
        )

        unet_logits = (self.unet(encoder_logit) for encoder_logit in encoder_logits)
        autodecoder_modules = [
            module for k, module in self.autodecoder._modules.items()
        ]
        autodecoder_segments = len(autodecoder_modules)
        counter = 0

        for unet_logit in unet_logits:
            print(f"counter:{counter}")
            decoder_logits = checkpoint_sequential(
                autodecoder_modules, autodecoder_segments, unet_logit
            )
            del unet_logit

            yield decoder_logits

            counter += 1
        del unet_logits

    def stack_tensors(
        self, decoded_tensors: Generator[torch.tensor, None, None]
    ) -> torch.tensor:
        """Sums the decoded tensors to get 1 tensor.

        Args:
            decoded_tensors (Generator[torch.tensor, None, None]): A generator of tensors.

        Returns:
            torch.tensor: A stacked tensor
        """
        stacked_tensor = sum(next(decoded_tensors))
        for tensor in decoded_tensors:
            stacked_tensor = stacked_tensor + sum(tensor)

        return stacked_tensor

    # @dask.delayed
    def forward_patch(
        self, patch: torch.Tensor, forward_type: str = "individual"
    ) -> torch.Tensor:
        """Forwards each patch in the forward function. Conducts these operations:
        1) Unravels image patch to array of pixels
        2) Calculates local maps of each pixel
        3) Generates global map of local maps
        4) Conducts tensor train decomposition of global map
        5) Pads tensors to desired shape
        6) Batches tensors to fit tensors into autoencoder, unet and autodecoder in batches
        7) Stacks all tensors to get 1 tensor
        8) Permutes tensor to (H*W*C) image

        Args:
            patch (torch.Tensor): A patch of image

        Returns:
            torch.Tensor: A patch of image after the unet model
        """
        FORWARD_FUNCTIONS = {
            "full": self._forward_padded_tensor_full,
            "individual": self._forward_padded_tensor,
        }
        print("Flattening patch...")
        pixels = self.convert_to_1d(patch)
        print("Calculating local maps...")
        local_maps = self.calculate_local_maps(pixels)
        print("Calculating global maps...")
        tensor_list = self.calculate_global_map(local_maps)
        print("Converting global map to TT...")
        tt_tensors = self.convert_to_tt(tensor_list)
        print("Padding tensors......")
        padded_tensors = self.pad_tensors(tt_tensors)
        print("Forward pass...")
        decoded_tensors = FORWARD_FUNCTIONS.get(forward_type)(padded_tensors)
        stacked_tensor = self.stack_tensors(decoded_tensors)
        # convert CHW to HWC
        # forwarded_tensor = torch.permute(stacked_tensor, (1, 2, 0)) // no need to convert to HWC
        print(f"Stacked tensor shape at the end of forward patch:{stacked_tensor.shape}")
        return torch.abs(stacked_tensor)

    # @dask.delayed
    def reassemble(
        self,
        patches: List[torch.Tensor],
    ) -> PatchedImage:
        """
        Takes a list of unravelled patches of shape(height,width,channels) and reassembles them into an image

        Args:
            image_height (int): Height of reassembled image
            image_width (int): Width of reassembled image
            patch_height (int): Height of each patch
            patch_width (int): Width of each patch
            patches (List(torch.Tensor)): A list of image patches

        Returns:
            torch.Tensor: A forwarded image after reassembling of patches
        """
        final_image = PatchedImage(image_channel = self.unet_channels, image_height = self.image_height, image_width = self.image_width) # C,H,W
        
        for patch_coord, patch in zip(self.patch_coords, patches):
            final_image.augment(patch_coord[0], patch_coord[1], patch.detach()) # this part got error 

        final_image.export('test_image.png') # placement for now

        return final_image
       

    def forward(self, image: PatchedImage) -> torch.Tensor:
        """This is the forward function of the model wrapper that forwards images in batches
        through the:
        1) patching of an image,
        2) forwarding of each patch,
        3) reassembling of patches into an image and,
        4) stacking of batches of images

        Args:
            image (torch.Tensor): image before forward function

        Returns:
            torch.Tensor: image after forward function
        """

        # self.image_height = image.shape[0]
        # self.image_width = image.shape[1]
        # image is C, H, W
        # image.load()
        # self.image_height = image.lazy_loaded_img_arr.shape[1]
        # self.image_width = image.lazy_loaded_img_arr.shape[2]

        print(f"Generating patches...\n")
        patches = self.load_patches(image)

        patch_counter = 0
        patch_cache = []

        for patch in patches:

            # permute (C, H, W) to (H, W, C) to match mask
            # hwc_image = torch.permute(patch, (1, 2, 0)) //no need because we will be using CHW
            print(f"PATCH shape at the start of forward:{patch.shape}")
            # stacked_tensor = dask.delayed(self.forward_patch)(self, patch)
            stacked_tensor = self.forward_patch(patch)
            print("stacked_tensor.shape: ", stacked_tensor.shape)
            patch_cache.append(stacked_tensor)
            patch_counter += 1
            print(f"patch counter: {patch_counter}")

        print("number of patches", len(patch_cache))

        # final_batch_image = dask.delayed(self.reassemble)(
        #     self,
        #     self.image_height,
        #     self.image_width,
        #     self.patch_height,
        #     self.patch_height,
        #     patch_cache,
        # )

        final_image = self.reassemble(patch_cache)  # H, W, C

        print("shape of final patch image", final_image.lazy_loaded_img_arr.shape)


        # images = dask.delayed(torch.stack)(batch_cache)

        return final_image
