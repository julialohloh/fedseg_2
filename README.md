# FedSeg

Main repo for prototyping federated image segmentation functionality in Synergos

## Description

Federated patch-based image segmentation with boundary refinement consisting of:

- Custom lazy dataloader that adopts DASK and GDAL library for preprocessing image data,
- Image segmentation wrapper that uses tensorly for training user-defined model,
- Boundary refinement model that uses mmcv for refining the predicted boundary.

## Visuals

![fedseg](/fedseg-visuals.png)

## Installation

### Dependencies

#### Worker

- dask==2022.5.0
- gdal=3.5.1
- imagesize==1.3.0
- numpy==1.23.1
- pandas=1.4.4
- psutil==5.7.0
- ray==1.13.0
- torch==1.12.0
- torchvision==0.13.0
- torchaudio==0.12.0

#### TTP

- dask==2022.7.0
- dask-core==2022.7.0
- natsort==8.2.0
- numpy==1.23.1
- opencv==4.5.5
- pillow==9.2.0
- psutil==5.7.0
- pyyaml==6.0
- ray==1.13.0
- torch==1.12.0
- torchvision==0.13.0
- torchaudio==0.12.0
- tqdm==4.64.0
- yacs==0.1.8
- mmcv-full==1.6.1
- tensorly==0.6.0

## Usage

### Chunking and Patching Image Data

Users populate image file paths and save them in the standardized ‘mappings.csv’.

Users then define the desired patch size to be processed from the raw image.

Coordinates are used to extract patches which are used as input for subsequent model processing.

Functions available to users are load (to load raw image), retrieve (to retrieve the patch), export (to save data as image file).

Users can define augmentation (i.e. flipping) when using the retrieve function.

### Boundary Refinement

#### Requirements

1. Data

- Predicted Masks and Ground Truth Masks MUST be declared in grayscale (1 channel).

\
2. Initializing BoundaryPatcher Class

- mask_height refers to the mask height.
- mask_width refers to the mask width.
- patch_size refers to the boundary box size. (square)
- iou_threshold refers to the threshold that nms operation will use to filter the appropriate boundary boxes.

```
BP = BoundaryPatcher(
    mask_height=960,
    mask_width=999,
    patch_size=64,
    iou_threshold=0.1
    )
```

\
3. Using extract_load to obtain crop_mask_dict, mask_coord_dict, nms_dataset

- test

```
top_left = PatchCoord(0,0)
bottom_right = PatchCoord(999,960)

_, _, nms_dataset = BP.extract_load(
                    raw_dataloader= raw_data_loader,
                    predicted_dataloader= pred_data_loader,
                    augmentation=True,
                    top_left=top_left,
                    bottom_right=bottom_right
                                )
```

\
4. Obtain model configs for HRNet and relevant config files for model training. The hyperparameters within these config files can be tuned during optimization.

- Model_configs for HRNet18 adopted from - https://github.com/tinyalpha/BPR/tree/main/configs
- Train/ Inference/ Predict configs tunable hyperparameters such as EPOCHS, LEARNING_RATE

\
5. HRNet model within HRNetRefine class takes on stacked image and mask (total: 4 channels) and conducts convolutions separately for image and mask. After stages of high-resolution convolutions with multi-resolution fusions, concatenation of the upsampled representations from all the resolutions is carried out with the last layer passed to a sigmoid function to obtain a probability score.

\*Deep High-Resolution Representation Learning for Visual Recognition https://arxiv.org/pdf/1908.07919v2.pdf

\
6. Using model_pipeline function, train and test model functions will be called. During the train and test process, several metrics will be computed which include train accuracy, train loss, F1 score etc.

- Model_pipeline takes in hyperparameters within config files to instantiate HRNet model in (5)
- Instantiated HRNet model, train_loader and val_loader are then fed into the training pipeline and testing pipeline.
- Based on user-defined number of epochs, training and testing will be conducted, with the best model saved under SAVED_PATH
- Metric functions stored within metrics.py are called in the training and testing process
- Input channel in train_loader must be 4

```
	model, best_iou = model_pipeline(
                        modelcfg,
                        lr= cfg['LEARNING_RATE'],
                        epochs=cfg['EPOCHS'],
                        train_dataset=train_loader,
                        valid_dataset=val_loader,
                        SAVE_PATH = cfg['SAVE_PATH'])
```

\
7. Model inference can also be performed by calling the infer function.

- where infer_loader contains the inference dataset for loading into the HRNet model
- Masks declared MUST be specified in grayscale In an inference dataset

```
test_loss, mean_iou= infer(model, infer_loader)

```

\
8. Model prediction can also be performed by calling the predict function.

- where test_pred_loader contains the predict dataset for loading into the HRNet model
- path takes on the filepath of saved model in string format
- model is initiated using the hyperparameters within the model config file]
- Masks declared MUST be specified in grayscale In an inference dataset

```
path = 'worker/outputs/saved_model.pth'
predictions, image_list = predict(path, test_pred_loader, modelcfg)

```

\
9. Using merge_patch to merge the refined patch onto the coarse mask output a refined mask.

- Important: augmentation MUST be set to False.
- Important: batch_size for the dataloader MUST be set to 1.
- coord_dict from extract_load is a dictionary with mask index and coordinates of the patches for that mask.
- predictions is the list of predicted patches(refined patch)
- saved_path is the directory where the refined mask will be saved.
- To be Updated: details on the dataloader.

```
# Extract crop patches into test_pred_dataset with the co-ord

# dictionary and crop mask patch list

top_left = PatchCoord(0,0)
bottom_right = PatchCoord(999,960)

crop_mask_patch, coord_dict, test_pred_dataset = BP.extract_load(
   raw_dataloader= test_pred_data_loader,
   predicted_dataloader= placeholder_data_loader,
   augmentation=False,
   top_left=top_left,
   bottom_right=bottom_right
)

# Create dataloader that send the crop patches into the model for

# predictions

BATCH_SIZE = 1

test_pred_loader = DataLoader(test_pred_dataset, batch_size=BATCH_SIZE, shuffle=False)

path = 'worker/outputs/saved_model.pth'

# predictions is the list of predicted patches(refined patch)

predictions, image_list = predict(path, test_pred_loader, modelcfg)

# Create a path for output of refined mask

saved_path = os.path.join(os.getcwd(),'worker/data/test_data/refined_mask')

BP.merge_patch(
   coord_dict,
   predictions,
   test_pred_data_loader,
   saved_path,
   top_left,
   bottom_right
)

```

### Image segmentation wrapper

#### Requirements

1. Data

- Input image of PatchedImage class into the forward function

\
2. Processing of image in forward function of image segmentation wrapper

- patching of an image,
- forwarding of each patch,
  - patching of an image,
  - forwarding of each patch,
  - Unravels image patch to array of pixels
  - Calculates local maps of each pixel
  - Generates global map of local maps
  - Conducts tensor train decomposition of global map
  - Pads tensors to desired shape
  - Batches tensors to fit tensors into autoencoder, unet and autodecoder in batches
  - Stacks all tensors to get 1 tensor
  - Permutes tensor to (H*W*C) image
- reassembling of patches into an image and,
- stacking of batches of images

\
3. Initializing FedSeg Class

- Unet_channels: Number of channels of the tensors input into the unet
- Unet_classes: Number of classes of the tensors to be generated from the unet
- Image_height: Height of image
- Image_width: Width of image
- patch_height: Height of patch
- patch_width: Width of patch
- Tt_decomp_max_rank: Max rank of tensors resulting from tensor train decomposition
- Tensor_batch_size: Batch size of tensors to be passed into the unet model
- AE: autoencoder before the unet
- AD: autodecoder after the unet
- Unet: Custom unet defined by the user
- Map_type: Type of local map. Choose from "intensity", "fourier", "sinusoidal".
- Tt_threshold: Maximum dimension allowed for global map before tensor train decomposition

```
# Initializing FedSeg Model

fedseg = FedSeg(
   unet_channels=3,
   unet_classes=1,
   image_height=8,
   image_width=4,
   patch_height=4,
   patch_width=4,
   AE=AE,
   AD=AE,
   unet=UNet,
   map_type: str = "intensity",
   tensor_batch_size: int = 500,
   tt_decomp_max_rank: int = 3,
   tt_threshold: int = 8,
)
```

\
4. Preparing Lazy Dataloaders

```
lazy_train = LazyDataset("data/stare/examples/stare/datasets/partition3/train")
lazy_test = LazyDataset("data/stare/examples/stare/datasets/partition3/evaluate")

lazy_train_dataloader = DataLoader(
   lazy_train, batch_size=4, collate_fn=lambda x: tuple(x)
)
lazy_test_dataloader = DataLoader(
   lazy_test, batch_size=4, collate_fn=lambda x: tuple(x)
)
```

\
5. Model Training

```
   for i, data in enumerate(tqdm(fedseg.images(lazy_train_dataloader))):
   image, mask = data
   output = model(image)
```

\
6. Model Evaluation

```
   for i, data in enumerate(tqdm(fedseg.images(lazy_test_dataloader))):
   image, mask = data
   output = model(image)
```

## Authors and acknowledgment

Show your appreciation to those who have contributed to the project.
\
https://github.com/tinyalpha/BPR
\
https://github.com/HRNet/HRNet-Semantic-Segmentation

## License

For open source projects, say how it is licensed.
\
MIT License
\
Apache-2.0 license
