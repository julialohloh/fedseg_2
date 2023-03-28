# Libs
import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import gaussian_kde

#################
# Fisher Metric #
#################


def fisher(
    outputs: torch.tensor, labels: torch.tensor, num_classes: int = 1
) -> tuple[torch.tensor, torch.tensor]:

    pass

def fisher(gt:torch.tensor,preds:torch.tensor):
    ground_truths = gt
    predictions = preds
    combined = np.vstack([ground_truths,predictions])
    gaussian = gaussian_kde(combined) 
    
