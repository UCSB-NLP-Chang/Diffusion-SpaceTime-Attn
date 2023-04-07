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

from .Encoder import RelEncoder
from .Decoder import BboxDecoder, BboxRegDecoder
from .Model import Rel2Layout, Rel2RegLayout, Rel2Bbox

logger = logging.getLogger('model')

def build_model(cfg):
    vocab_size = cfg['MODEL']['ENCODER']['VOCAB_SIZE']
    obj_classes_size = cfg['MODEL']['ENCODER']['OBJ_CLASSES_SIZE']
    hidden_size = cfg['MODEL']['ENCODER']['HIDDEN_SIZE']
    num_layers = cfg['MODEL']['ENCODER']['NUM_LAYERS']
    attn_heads = cfg['MODEL']['ENCODER']['ATTN_HEADS']
    dropout= cfg['MODEL']['ENCODER']['DROPOUT']

    model = Rel2Bbox(vocab_size=vocab_size, obj_classes_size=obj_classes_size,
                           hidden_size=hidden_size, num_layers=num_layers, 
                           attn_heads=attn_heads, dropout=dropout, cfg=cfg)

    logger.info('Setup model {}.'.format(model.__class__.__name__))
    logger.info('Model structure:')
    logger.info(model)
    return model
