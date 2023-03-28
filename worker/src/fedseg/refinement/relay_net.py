"""ClassificationCNN"""
import torch
import torch.nn as nn

import sub_module as sm
from yacs.config import CfgNode
from build import BACKBONE_REGISTRY

class ReLayNet(nn.Module):
    """
    A PyTorch implementation of ReLayNet
    Coded by Shayan and Abhijit


    """
    

    def __init__(self, params):
        super(ReLayNet, self).__init__()
        params['num_channels'] = 1
        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        # params['num_channels'] = 64  # This can be used to change the numchannels for each block
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, X):
        e1, out1, ind1 = self.encode1.forward(X)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        bn = self.bottleneck.forward(e3)

        d3 = self.decode1.forward(bn, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)

        return prob

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

@BACKBONE_REGISTRY.register('mobilenet_v2')
def build_mobilenet_v2_backbone(backbone_cfg: CfgNode, **kwargs) -> nn.Module:
    """

    :param backbone_cfg: backbone config node
    :param kwargs:
    :return: backbone module
    """
    model = ReLayNet(backbone_cfg,**kwargs)
    # if backbone_cfg.get('PRETRAINED_PATH'):
    #     pretrained_path = backbone_cfg['PRETRAINED_PATH']
    #     state_dict = torch.load(pretrained_path, map_location='cpu')
    #     model.load_state_dict(state_dict, strict=False)
    return model