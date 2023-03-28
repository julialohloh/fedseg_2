####################
# REQURIED MODULES #
####################

# Libs
import torch
from  torchmetrics.classification.accuracy import MulticlassAccuracy
from scipy.spatial.distance import directed_hausdorff
import numpy as np
###################
#  Pixel accuracy #
###################

def multi_accuracy(outputs: torch.tensor, labels: torch.tensor, num: int) -> float:
    """Compute multiclass pixel count accuracy between predicted and groundtruth

    Args:
        outputs (torch.tensor): predicted mask from HRNET model
        labels (torch.tensor): ground truth mask
        num (int): num of classes

    Returns:
        accuracy score (float)
    """
    metric =  MulticlassAccuracy(num,  average = 'weighted')
    multiacc = metric(outputs, labels)
    return round(float(multiacc), 2)

def pixel_accuracy(outputs: torch.tensor, labels: torch.tensor) -> float:
    """Compute pixel count accuracy between predicted and groundtruth

    Args:
        outputs (torch.tensor): predicted mask from HRNET model
        labels (torch.tensor): ground truth mask

    Returns:
        accuracy score (float) 
    """
    with torch.no_grad():
        correct = torch.eq(outputs, labels).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return round(accuracy, 2)


def multiDICE(outputs: torch.tensor, labels: torch.tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute multiclass Sorensen Dice coefficient score between predicted and groundtruth

    Args:
        outputs (torch.tensor): predicted mask from HRNET model
        labels (torch.tensor): ground truth mask

    Returns:
        dice_score (torch.tensor)

    """
    # dice = torch.mean(dice)

    if num_classes <= 0:
        raise ValueError("num_classes must be more than zero")

    if outputs.shape != labels.shape:
        raise ValueError("outputs and labels should be of the same shape")

    dice_score = []
    for num in range(0, num_classes):
        intersection = ((outputs == num) * (labels == num)).sum()
        dice_sum = ((outputs == num).sum() + (labels == num).sum())
        if dice_sum == 0:
            dice_score.append(float("nan"))
        else:
            dice_score.append((2 * intersection)/dice_sum)
    return torch.tensor(dice_score),torch.nanmean(torch.tensor(dice_score))

##############
# MIOU Metric#
##############


def miou(
    outputs: torch.tensor, labels: torch.tensor, num_classes: int = 1
) -> tuple[torch.tensor, torch.tensor]:
    """Returns iou score by comparing outputs (prediction) and labels (groundtruth)

    Args:
        outputs (torch.tensor): predicted mask
        labels (torch.tensor): ground truth mask
        num_classes (int, optional): number of classes. Defaults to 1.

    Returns:
        tuple(torch.tensor, torch.tensor): a tuple of all the ious of each class
        in a tensor and the mean of all the ious
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be more than zero")

    if outputs.shape != labels.shape:
        raise ValueError("outputs and labels should be of the same shape")

    ious = []
    for num in range(1, num_classes + 1):
        intersection = ((outputs == num) * (labels == num)).sum()
        union = (outputs == num).sum() + (labels == num).sum() - intersection
        if union == 0:
            ious.append(
                float("nan")
            )  # if there is no class in ground truth, do not include in evaluation
        else:
            ious.append((intersection + 1e-6) / (union + 1e-6))

    return torch.tensor(ious), torch.nanmean(torch.tensor(ious))


def hd(truth, pred):
    """
    _summary_

    Args:
        truth (_type_): _description_
        pred (_type_): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    num = np.intersect1d(np.unique(truth), np.unique(pred))
    if len(num) == 0:
        raise Exception('Invalid image')

    hd_max = 0
    for n in num:
        y_truth, x_truth = np.where(truth==n)
        y_pred, x_pred = np.where(pred==n)

        u = list(zip(x_truth, y_truth))
        v = list(zip(x_pred, y_pred))
        hd_value = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        print(f'For class {n}, Hausdorff Distance Value is {hd_value}')
        hd_max = max(hd_max, hd_value)
    return hd_max
