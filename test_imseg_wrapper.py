#!/usr/bin/env python

####################
# Required Modules #
####################
# Libraries
import pytest
import torch

# Custom
from conftest import *
from unet.unet import UNet


################
# Fedseg Tests #
################


@pytest.mark.parametrize(
    "patch_height, patch_width, expected",
    [(5, 5, (3, 5, 5)), (5, 6, (3, 5, 6)), (6, 5, (3, 6, 5))],
)
def test_patch_size(fedseg, dataloader, patch_height, patch_width, expected):
    """Check that patch size is correct for different patch height and patch width"""

    fedseg.patch_height = patch_height
    fedseg.patch_width = patch_width

    image, _ = next(fedseg.images(dataloader))
    image.load()
    patches = fedseg.load_patches(image)
    for patch in patches:
        shape = patch.shape
    assert shape == expected  # C, H, W


def test_patch_count(fedseg, dataloader):
    """Check that patch count is correct with given patch size and image size"""

    fedseg.patch_height = 6
    fedseg.patch_width = 5
    fedseg.image_height = 55
    fedseg.image_width = 21

    image, _ = next(fedseg.images(dataloader))
    image.load()
    patches = fedseg.load_patches(image)
    assert len(list(patches)) == 50


def test_max_dim(fedseg):
    """Assert error if max dimension is less than tensor shape dimension."""

    with pytest.raises(ValueError) as e:
        fedseg.tt_decomp_max_rank = 2
        tensor_gen = (torch.randint(0, 255, (2, 3, 2)) / 255 for i in range(2))
        next(fedseg.pad_tensors(tensor_gen))
        assert "tt_decomp_max_rank must be bigger than tensor shape" in str(e.value)


def test_patch_image_size(fedseg):
    """Check that error is thrown when patch size is bigger than image size"""
    with pytest.raises(ValueError) as e:
        fedseg.patch_height = 30
        assert "patch_height must be smaller than image height" in str(e.value)
    with pytest.raises(ValueError) as e:
        fedseg.patch_width = 30
        assert "patch_width must be smaller than image width" in str(e.value)


@pytest.mark.parametrize(
    "patch_height, patch_width, expected",
    [(3, 3, 504), (4, 4, 52488)],
)
def test_tensor_count(fedseg, patch_height, patch_width, expected):
    """Check that the tensor count is expected for different patch size."""
    fedseg.patch_height = patch_height
    fedseg.patch_width = patch_width
    patch = torch.randint(0, 255, (patch_height, patch_width, 3))  # H, W, C
    pixels = fedseg.convert_to_1d(patch)
    local_maps = fedseg.calculate_local_maps(pixels)
    tensor_list = fedseg.calculate_global_map(local_maps)
    tt_tensors = fedseg.convert_to_tt(tensor_list)

    assert len(list(tt_tensors)) == expected


@pytest.mark.parametrize(
    "image_height, image_width, patch_height, patch_width, expected",
    [(20, 20, 5, 5, (20, 20, 3)), (20, 21, 5, 5, (20, 25, 3))],
)
def test_reassemble(
    fedseg, dataloader, image_height, image_width, patch_height, patch_width, expected
):
    """Check that shape of final_image is expected with given image size."""
    fedseg.image_height = image_height
    fedseg.image_width = image_width
    fedseg.patch_height = patch_height
    fedseg.patch_width = patch_width
    image, _ = next(fedseg.images(dataloader))
    image.load()
    patches = fedseg.load_patches(image)
    patch_count = len(list(patches))
    patches = [torch.randint(0, 255, (patch_height, patch_width, 3)) / 255] * int(
        patch_count
    )
    final_image = fedseg.reassemble(patches)

    assert final_image.shape == expected


def test_channel_size(fedseg, dataloader):
    """Check final image shape of grayscale image through the forward function"""
    fedseg.unet_channels = 1  # image channel
    fedseg.unet_classes = 1  # output classes for prediction
    fedseg.unet = UNet(1, 1)
    _, bw_image = next(fedseg.images(dataloader))
    final_image = fedseg.forward(bw_image)

    assert final_image.shape == (9, 9, 1)  # H, W, C


@pytest.mark.parametrize(
    "unet_input, expected",
    [(100, (1, 3, 3, 3)), (50, (1, 3, 3, 3)), (200, (1, 3, 3, 3)), (64, (1, 3, 3, 3))],
)
def test_unet_input(fedseg, unet_input, expected):
    """Check whether shape of decoder_logits is correct with given unet_input

    Args:
        fedseg (_type_): _description_
    """
    fedseg.unet_input = unet_input

    fedseg.create_adapter_layers()
    tensor = torch.randint(0, 255, (1, 3, 3, 3)) / 255

    encoder_logits = fedseg.autoencoder.forward(tensor)
    unet_logits = fedseg.unet(encoder_logits)
    decoder_logits = fedseg.autodecoder(unet_logits)

    assert decoder_logits.shape == expected


def test_unet_input_low(fedseg):
    """Check that unet_input is 32 and more"""
    with pytest.raises(ValueError) as e:
        fedseg.unet_input = 5
        assert "unet_input must be 32 and more" in str(e.value)


def test_tt_threshold(fedseg):
    """tt_threshold cannot be less than 3 and more than 22
    (Takes too long to run. At an even higher threshold, a RuntimeError will show.)

    """
    with pytest.raises(ValueError) as e:
        fedseg.tt_threshold = 3
        assert "tt_threshold must be between 4 and 100" in str(e.value)


@pytest.mark.parametrize(
    "unet_classes, expected",
    [(100, (1, 100, 3, 3)), (10, (1, 10, 3, 3))],
)
def test_class_size(fedseg, unet_classes, expected):
    """Check if code works with different class size"""
    fedseg.unet_classes = unet_classes
    fedseg.unet = UNet(3, unet_classes)
    fedseg.create_adapter_layers()
    tensor = torch.randint(0, 255, (1, 3, 3, 3)) / 255

    encoder_logits = fedseg.autoencoder.forward(tensor)
    unet_logits = fedseg.unet(encoder_logits)
    decoder_logits = fedseg.autodecoder(unet_logits)
    assert decoder_logits.shape == expected


def test_init_values(fedseg):
    """
    Check that all values cannot be zero
    """
    with pytest.raises(ValueError) as e:
        fedseg.unet_channels = 0
        assert "unet_channels must be either 1 or 3" in str(e.value)

    with pytest.raises(ValueError) as e:
        fedseg.unet_classes = 0
        assert "unet_classes must be more than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        fedseg.image_height = 0
        assert "image_height must be more than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        fedseg.image_width = 0
        assert "image_width must be more than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        fedseg.patch_height = 0
        assert "patch_height must be more than zero" in str(e.value)

    with pytest.raises(ValueError) as e:
        fedseg.patch_width = 0
        assert "patch_width must be more than zero" in str(e.value)
    with pytest.raises(ValueError) as e:
        fedseg.map_type = ""
        assert "map_type must be 'intensity', 'fourier' or 'sinusoidal'" in str(e.value)
    with pytest.raises(ValueError) as e:
        fedseg.tensor_batch_size = 0
        assert "tensor_batch_size must be more than zero" in str(e.value)
    with pytest.raises(ValueError) as e:
        fedseg.tt_decomp_max_rank = 0
        assert "tt_decomp_max_rank must be more than zero" in str(e.value)
    with pytest.raises(ValueError) as e:
        fedseg.tt_threshold = 0
        assert "tt_threshold must be between 4 and 100" in str(e.value)
