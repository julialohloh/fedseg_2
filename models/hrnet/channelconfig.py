
####################
# REQURIED MODULES #
####################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN

##################
# MODEL CONNFIGS #
##################

HRNET_18 = CN()
HRNET_18.FINAL_CONV_KERNEL = 1
HRNET_18.BN_MOMENTUM = 1
HRNET_18.ALIGN_CORNERS = False
HRNET_18.STAGE1 = CN()
HRNET_18.STAGE1.NUM_MODULES = 1
HRNET_18.STAGE1.NUM_BRANCHES = 1
HRNET_18.STAGE1.NUM_BLOCKS = [2]
HRNET_18.STAGE1.NUM_CHANNELS = [64]
HRNET_18.STAGE1.BLOCK = 'BOTTLENECK'
HRNET_18.STAGE1.FUSE_METHOD = 'SUM'

HRNET_18.STAGE2 = CN()
HRNET_18.STAGE2.NUM_MODULES = 1
HRNET_18.STAGE2.NUM_BRANCHES = 2
HRNET_18.STAGE2.NUM_BLOCKS = [2, 2]
HRNET_18.STAGE2.NUM_CHANNELS = [18, 36]
HRNET_18.STAGE2.BLOCK = 'BASIC'
HRNET_18.STAGE2.FUSE_METHOD = 'SUM'

HRNET_18.STAGE3 = CN()
HRNET_18.STAGE3.NUM_MODULES = 3
HRNET_18.STAGE3.NUM_BRANCHES = 3
HRNET_18.STAGE3.NUM_BLOCKS = [2, 2, 2]
HRNET_18.STAGE3.NUM_CHANNELS = [18, 36, 72]
HRNET_18.STAGE3.BLOCK = 'BASIC'
HRNET_18.STAGE3.FUSE_METHOD = 'SUM'

HRNET_18.STAGE4 = CN()
HRNET_18.STAGE4.NUM_MODULES = 2
HRNET_18.STAGE4.NUM_BRANCHES = 4
HRNET_18.STAGE4.NUM_BLOCKS = [2, 2, 2, 2]
HRNET_18.STAGE4.NUM_CHANNELS = [18, 36, 72, 144]
HRNET_18.STAGE4.BLOCK = 'BASIC'
HRNET_18.STAGE4.FUSE_METHOD = 'SUM'


MODEL_CONFIGS = {
    'hrnet18': HRNET_18,
}

##################
# CORE FUNCTIONS #
##################

def get_cfg_defaults() -> CN():
  """Runs yacs CfgNode for HRNET_18
  
  Returns:
    HRNET_18 (dict)
  
  """
  return HRNET_18.clone()
