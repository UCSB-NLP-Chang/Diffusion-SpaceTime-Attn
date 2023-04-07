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

from .inference_coco import Inference_COCO
from .inference_vg import Inference_VG
from .inference_vg_msdn import Inference_VG_MSDN

logger = logging.getLogger('inference')

def build_inference(cfg):
    data_dir = cfg['DATASETS']['DATA_DIR_PATH']
    
    test_output = cfg['TEST']['OUTPUT_DIR']
    
    if cfg['DATASETS']['NAME'] == 'coco':
        vocab_dic_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        with open(vocab_dic_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        infer = Inference_COCO(save_dir = test_output, vocab_dict = vocab_dict)
        
    elif cfg['DATASETS']['NAME'] == 'vg_msdn':
        vocab_dic_path = os.path.join(data_dir, 'object_pred_idx_to_name.pkl')
        with open(vocab_dic_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        infer = Inference_VG_MSDN(save_dir = test_output, vocab_dict = vocab_dict)
        
    elif cfg['DATASETS']['NAME'] == 'vg':
        vocab_dic_path = os.path.join(data_dir, cfg['DATASETS']['REL_DICT_FILENAME'])
        cls_dic_path = os.path.join(data_dir, cfg['DATASETS']['CLS_DICT_FILENAME'])
        anns_path = os.path.join(data_dir, cfg['DATASETS']['ANNS_FILENAME'])
        with open(vocab_dic_path, 'rb') as file:
            vocab_dict = pickle.load(file)
        with open(cls_dic_path, 'rb') as file:
            cls_dict = pickle.load(file)
        with open(anns_path, 'rb') as file:
            all_anns = pickle.load(file)
        infer = Inference_VG(save_dir=test_output, vocab_dict=vocab_dict,
                             cls_dict=cls_dict, all_anns=all_anns)
    else:
        raise Exception("Sorry, we only support vg, vg_msdn and coco datasets.")
            
    logger.info('Setup inference {}.'.format(infer.__class__.__name__))
    logger.info('Start Inference.')
    return infer
