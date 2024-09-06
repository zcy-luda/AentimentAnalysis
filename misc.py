'''
Author: Chengyu Zheng
Date: 2024-09-06
Description: 

'''
import random

import torch
import torch.nn

import numpy as np

def setup_seed(seed, cudnn_deterministic=False):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True