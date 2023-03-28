# #!/usr/bin/env python

# ####################
# # Required Modules #
# ####################
# # Libraries

# import pytest
# from torch.utils.data import DataLoader

# # Custom
# from preprocessing import LazyDataset
# from autoencoder.cnnautoencoder_asym import AE
# from preprocessing import LazyDataset
# from imseg_wrapper import FedSeg
# from unet.unet import UNet

# ########################
# # Fedseg Test Fixtures #
# ########################


# @pytest.fixture
# def dataloader():

#     lazy_train = LazyDataset("data/stare/examples/stare/datasets/partition3/train")

#     lazy_train_dataloader = DataLoader(
#         lazy_train, batch_size=4, collate_fn=lambda x: tuple(x)
#     )

#     return lazy_train_dataloader


# @pytest.fixture
# def fedseg(dataloader):
#     fedseg = FedSeg(
#         unet_channels=3,
#         unet_classes=3,
#         patch_height=3,
#         patch_width=3,
#         image_height=9,
#         image_width=9,
#         AE=AE,
#         AD=AE,
#         unet=UNet,
#     )

#     image, _ = next(fedseg.images(dataloader))
#     image.load()
#     fedseg.load_patches(image)

#     return fedseg
