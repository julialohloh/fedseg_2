# List of APIs
import torch
import torch.nn as nn



class BasicBlock(nn.Module):


    def __init__(self, params):
        super(BasicBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
#         self.prelu = nn.PReLU()
        self.relu = nn.ReLU()

    def forward(self, X):
        out_conv = self.conv(X)
        out_bn = self.batchnorm(out_conv)
        out_prelu = self.relu(out_bn) # why is prelu used here
        return out_prelu


class EncoderBlock(BasicBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, X):
        out_block = super(EncoderBlock, self).forward(X)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(BasicBlock):
    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, X, out_block, indices):
        unpool = self.unpool(X, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        return out_block


class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])
#         self.softmax = nn.Softmax2d()

    def forward(self, X):
        out_conv = self.conv(X)
        #out_logit = self.softmax(out_conv)
        return out_conv