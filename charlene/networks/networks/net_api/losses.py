
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
    def forward(self, output, encoded_target, weights=None, ignore_index=None):
        """
            output : NxCxHxW Variable
            target :  NxHxW LongTensor
            weights : C FloatTensor
            ignore_index : int index to ignore from loss
            mean dice loss across channels
            """
        eps = 0.0001

#         output = output.exp()
#         encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
#         else:
#             encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        encoded_target = encoded_target.detach().float()
#         print(output.size())
#         print(encoded_target.size())
        intersection = output * encoded_target # N x C x H x W
        numerator = 2 * intersection.sum(0).sum(1).sum(1) # sum across batches, H and W
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps # sum across batches, height and width
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1) 

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()
        self.device = device

    def forward(self, X, target, weight):
        # TODO: why?
        target = target.type(torch.LongTensor).to(self.device)
#         
#         target = torch.unsqueeze(target, dim=1)
        
        input_soft = F.softmax(X,dim=1)
#         y2 = torch.mean(self.dice_loss(input_soft, target)) # get mean of all elements
        y2 = self.dice_loss(input_soft, target)
        _,target = torch.max(target, dim = 1)
#         print("x size")
#         print(X.size())
#         print("target size")
#         print(target.size())
#         print("weight size")
#         print(weight.size())
        y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(X, target), weight)) # combines logsoftmax and nllloss
        y = y1 + 0.5*y2
#         print('Y1')
#         print(y1)
#         print('y2')
#         print(y2)
        return y
    

