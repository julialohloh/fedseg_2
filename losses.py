import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F


class DiceCoeff(nn.Module):
    """Dice coeff for individual examples"""

    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        inter = torch.dot(input, target) + 0.0001
        union = torch.sum(input ** 2) + torch.sum(target ** 2) + 0.0001

        t = 2 * inter.float() / union.float()
        return t


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceLoss(_Loss):
    def forward(self, output, target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            """
        eps = 0.0001

        output = output.exp()
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.

#     Args:
#          input: A tensor of shape [N, 1, *]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, *]
#     """
#     shape = np.array(input.shape)
#     shape[1] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, input.cpu(), 1)

#     return result


# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#         reduction: Reduction method to apply, return mean over batch if 'mean',
#             return sum if 'sum', return a tensor of shape [N,] if 'none'
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1, p=2, reduction='mean'):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction

#     def forward(self, predict, target):
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)

#         num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
#         den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

#         loss = 1 - num / den

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))


# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index

#     def forward(self, predict, target):
#         # print(f"predict shape:{predict.shape}")
#         # print(f"target shape:{target.shape}")
#         # target = torch.permute(target,(0,3,1,2))
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.softmax(predict, dim=1)

#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 dice_loss = dice(predict[:, i], target[:, i])
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weights[i]
#                 total_loss += dice_loss

#         return total_loss/target.shape[1]

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self): 
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight):
        # TODO: why?
        target = target.type(torch.LongTensor).cuda()
        # target = target.type(torch.FloatTensor).cuda()
        input_soft = F.softmax(input,dim=1)
        y2 = torch.mean(self.dice_loss(input_soft, target))
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        y = y1 + (0.5 * y2)
#         y = y1 + y2
        return y

