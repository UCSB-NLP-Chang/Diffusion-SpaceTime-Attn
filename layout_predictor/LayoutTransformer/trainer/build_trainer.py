import os
import sys
import torch
import logging
import random
import torch.utils.data as data
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data.dataloader import default_collate
import pickle
import sys
from . import Trainer, PretrainTrainer, RegTrainer
import logging

def build_trainer(cfg, model, dataloader, dataloader_r, opt):
    logger = logging.getLogger('dataloader')
    
    T = PretrainTrainer(model = model, dataloader = dataloader,
                        dataloader_r=dataloader_r, opt = opt, cfg = cfg)
    
    logger.info('Setup trainer {}.'.format(T.__class__.__name__))
    return T
