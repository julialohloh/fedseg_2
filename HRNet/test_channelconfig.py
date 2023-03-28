from random import random
import pytest
import torch
import numpy as np
from .utils.auxillary import tensor2np
from .utils.load_data_4c import get_dataset
from .utils.metrics import f1, get_iou_score, separate_count, precision, recall, DSC, divide
from .src.modelling4c import _hrnet
from .configs.channelconfig import get_cfg_defaults


################
# load_data.py #
################
@pytest.fixture
def example_valid():
    # valid_ratio = 0.3
    # train_ratio = 0.6
    dataset = torch.rand(50, 500,300)
    random_seed = 42
    return dataset, random_seed


def test_fail_train_ratio(example_valid):
    """
    C1: Check that train_ratio cannot be 0, else raise ValueError
    """
    with pytest.raises(ValueError) as e:
        train_ratio = 0
        valid_ratio = 0.6
        train_dataset,valid_dataset, test_dataset = get_dataset(*example_valid, valid_ratio, train_ratio)
        assert 'valid_ratio or train_ratio should be in the range [0, 1]' in str(e.value)

def test_fail_valid_ratio(example_valid):
    """
    C1: Check that valid_ratio cannot be 0, else raise ValueError
    """
    with pytest.raises(ValueError) as e:
        valid_ratio = 0
        train_ratio = 0.5
        train_dataset,valid_dataset, test_dataset = get_dataset(*example_valid, valid_ratio, train_ratio)
        assert 'valid_ratio or train_ratio should be in the range [0, 1]' in str(e.value)

def test_fail_getdataset(example_valid):
    """
    C1: Check that sum of valid_ratio and train_ratio cannot be more than 0, else raise ValueError
    """
    with pytest.raises(ValueError) as e:
        valid_ratio = 0.8
        train_ratio = 0.5   
        train_dataset,valid_dataset, test_dataset = get_dataset(*example_valid, valid_ratio, train_ratio)
        assert 'sum of valid ratio and train ratio must be less than 1' in str(e.value)

def test_getdataset(example_valid):
    """
    C1: Check that the ratio of validation dataset over dataset is same as valid_ratio.
    C2: Check that the ratio of training dataset over dataset is same as train_ratio.
    C3: Check that the ratio of test dataset over dataset is correct.
    """
    train_ratio = 0.5
    valid_ratio = 0.3
    train_dataset,valid_dataset, test_dataset = get_dataset(*example_valid, valid_ratio, train_ratio)
    assert len(valid_dataset)/len(example_valid[0])==round(valid_ratio,1)
    assert len(train_dataset)/len(example_valid[0])==round(train_ratio,1)
    assert len(test_dataset)/len(example_valid[0]) == round((1- valid_ratio - train_ratio),1)

###################################################################

# @pytest.mark.parametrize("valid_ratio, train_ratio, dataset", [
#     # (0,0,torch.rand(50, 500,300)),
#     (0.1,0.6,torch.rand(20, 500,300)),
#     (0.3,0.6,torch.rand(30, 500,300)),
# ])
# def test_get_dataset(valid_ratio, train_ratio, dataset, random_seed =42):
#     """
#     C1: Check that the ratio of validation dataset over dataset is same as valid_ratio.
#     C2: Check that the ratio of training dataset over dataset is same as train_ratio.
#     C3: Check that the ratio of test dataset over dataset is correct.
#     """
#     train_dataset,valid_dataset, test_dataset = get_dataset(dataset, random_seed, valid_ratio, train_ratio)
#     assert len(valid_dataset)/len(dataset)==round(valid_ratio,1)
#     assert len(train_dataset)/len(dataset)==round(train_ratio,1)
#     assert len(test_dataset)/len(dataset) == round((1- valid_ratio - train_ratio),1)

##############
# metrics.py #
##############

@pytest.fixture(scope = 'session')
def example_y_data():
    y_pred = torch.Tensor(
        [[3,3, 1],
        [2, 1, 3],
        [1, 3, 1]])
    y_true =  torch.Tensor(
        [[3,3, 1],
        [2, 2, 3],
        [1, 3, 1]])
    return y_pred, y_true

def test_separate_count(example_y_data):
    """
    C1: Check that computation of confusion matrix is accurate.
    """
    assert separate_count(*example_y_data) ==  (3, 0, 0, 0)

def test_precision(example_y_data):
    """
    C1: Check that computation of precision is accurate
    """
    assert precision(*example_y_data) == 1

def test_recall(example_y_data):
    """
    C1: Check that computation of recall is accurate.
    """
    assert recall(*example_y_data) == 1

def test_f1(example_y_data):
    """
    C1: Check that computation of f1 is accurate.
    """
    assert f1(*example_y_data) == 1

def test_dsc(example_y_data):
    """
    C1: Check that computation of dsc score is accurate.
    """
    assert DSC(*example_y_data) == (2 * (example_y_data[1] * example_y_data[0]).sum())/ (example_y_data[1].sum() + example_y_data[0].sum())


@pytest.mark.parametrize("dividend, divisor, expected", 
[
    [0,1,0.0],
    [2, 0, 0.0],
    [10, 2, 5.0]
])
def test_divide(dividend, divisor, expected):
    """
    C1: Check that there is no divsion by 0 error
    """
    result = divide(dividend, divisor)
    assert result == expected


# ################
# # auxillary.py #
# ################

def test_tensor2np(example_y_data):
    """
    C1: Check that dtype of output is numpy
    C2: Check that the array values remains consistent during the conversion
    """
    expected_mask = np.array([[3,3, 1],
        [2, 1, 3],
        [1, 3, 1]])
    mask = tensor2np(example_y_data[0])
    assert(np.array_equal(mask, expected_mask))

# @pytest.mark.parametrize("y_pred, y_true, test_confusion_results, pre_rec_expected", 
# [
#     (torch.Tensor(
#         [[3,3, 1],
#         [2, 1, 3],
#         [1, 3, 1]]), torch.Tensor(
#         [[3,3, 1],
#         [2, 2, 3],
#         [1, 3, 1]]), (3, 0, 0, 0), 1),
# ])

# def test_metrics(
#     y_pred, y_true, test_confusion_results, 
#     pre_rec_expected
#     ):
#     """
#     C1: Check that computation of confusion matrix is accurate.
#     C2: Check that computation of precision is accurate.
#     C3: Check that computation of recall is accurate.
#     C4: Check that computation of f1 is accurate.
#     C5: Check that computation of dsc score is accurate.
#     C6: Check that the appending of results into result list is accurate.
#     """
#     confusion_results = separate_count(y_pred, y_true)
#     precision_results = precision(y_pred, y_true)
#     recall_results = recall(y_pred, y_true)
#     f1_results = f1(y_pred, y_true)
#     dsc_results = DSC(y_pred, y_true)
#     result_list = get_all_metrics(y_pred,y_true)
#     assert confusion_results == test_confusion_results
#     assert precision_results == pre_rec_expected
#     assert recall_results == pre_rec_expected
#     assert f1_results == pre_rec_expected
#     assert dsc_results == (2 * (y_true * y_pred).sum())/ (y_pred.sum() + y_true.sum())
#     assert result_list == [pre_rec_expected, pre_rec_expected, pre_rec_expected, dsc_results]

@pytest.fixture
def example_iou():
    y_pred = torch.ones(1,1,64,64)
    return y_pred

def test_iou_fail(example_iou):
    """
    C1: Check that y_pred must be the same height and width as y_true, else raise ValueError
    """
    with pytest.raises(ValueError) as e:
        y_true = torch.zeros(1,1,28,32)
        iou_results = get_iou_score(example_iou, y_true)
        assert 'valid_ratio or train_ratio should be in the range [0, 1]' in str(e.value)


# pytest for iou_score
@pytest.mark.parametrize("y_pred, y_true, expected",
[
    (torch.ones(1,1,64,64), torch.zeros(1,1,64,64) , 1e-6/((64*64)+1e-6)),
    (torch.zeros(1,1,16,16), torch.zeros(1,1,16,16) , 1e-6/1e-6),
    (torch.ones(1,1,28,32), torch.zeros(1,1,28,32) , 1e-6/((28*32)+1e-6)),
])
def test_get_iou_score(y_pred, y_true, expected):
    """
    C1: Check that computation of iou score is accurate.
    """
    iou_results = get_iou_score(y_pred, y_true)
    assert iou_results == expected


    

##############
# model4c.py #
##############

@pytest.fixture
def example_model():
    input = torch.rand(1,3,64,64)
    return input

def test_model_fail(example_model):
    """
    C1: Check that channels of input must be 4
    """
    with pytest.raises(ValueError) as e:
        modelcfg = get_cfg_defaults()
        model = _hrnet(modelcfg)
        # sample = get_sample()
        model.eval()
        with torch.no_grad():
            output = model(example_model)
            assert 'input channel must be 4' in str(e.value)

@pytest.mark.parametrize("input",
[
    (torch.ones([1, 4, 64, 64])),
    (torch.ones([1, 4, 28, 10])),
])
def test_forward(input):
    """
    C1: Check that output height and width is consistent with input height and width.
    C2: Check that the input channel is not the same as the output channel as model takes in a 4 channel data and gives a 1 channel output as result.
    """
    modelcfg = get_cfg_defaults()
    model = _hrnet(modelcfg)
    model.eval()
    with torch.no_grad():
        output = model(input)
        assert output.shape[2:] == input.shape[2:]
        assert output.size(dim=1) != input.size(dim=1)  


