from loader import build_loader
from model import build_model
from trainer import build_trainer
from inference import build_inference, Inference_COCO
from utils import ensure_dir
import logging, coloredlogs
import argparse
import yaml
import os
import torch
import random
import json
from os import listdir
from os.path import isfile, join
import pickle
from tqdm import tqdm

# setting parser
parser = argparse.ArgumentParser()
parser.add_argument('--cfg_path', type=str, default='configs/')
parser.add_argument('--default_cfg_path', type=str, default='configs/default.yaml')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--test_output_dir', type=str, default='./images/Appendix/output_4')
parser.add_argument('--layout_output_dir', type=str, default='./images/Appendix/output_4/layout')
parser.add_argument('--test_scene_graphs_dir', type=str, 
       default='./images/vg_msdn_v24_sg')
parser.add_argument('--image_id', type=str, default='')
parser.add_argument('--repeat', type=int, default=1)
opt = parser.parse_args()

# setting config file
with open(opt.default_cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
with open(opt.cfg_path, 'r') as f:
    cfg.update(yaml.safe_load(f))

# handle dir for saving
ensure_dir(cfg['OUTPUT']['OUTPUT_DIR'])
ensure_dir(cfg['TEST']['OUTPUT_DIR'])

# setting logger
handlers = [logging.FileHandler(os.path.join(cfg['OUTPUT']['OUTPUT_DIR'],
                               'output_eval_from_sg.log'), mode = 'w'), 
logging.StreamHandler()]
logging.basicConfig(handlers = handlers, level=logging.INFO)
logger = logging.getLogger('root')
coloredlogs.install(logger = logger, fmt='%(asctime)s [%(name)s] %(levelname)s %(message)s')
logger.info('Setup output directory - {}.'.format(cfg['OUTPUT']['OUTPUT_DIR']))

if __name__ == '__main__':
    model = build_model(cfg)
    
    assert opt.checkpoint is not None, 'Please provide model ckpt for testing'
    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    ly_save_path = opt.layout_output_dir
    
    data_dir = cfg['DATASETS']['DATA_DIR_PATH']
    vocab_dic_path = os.path.join(data_dir,  'object_pred_idx_to_name.pkl')
    with open(vocab_dic_path, 'rb') as file:
        vocab_dict = pickle.load(file)
        
    infer = Inference_COCO(save_dir = opt.test_output_dir, vocab_dict = vocab_dict)
    
    # Load the scene graphs
    onlyfiles = [f for f in listdir(opt.test_scene_graphs_dir) if \
                   isfile(join(opt.test_scene_graphs_dir, f))]
    
    if opt.image_id != '':
        iter_idx = onlyfiles.index(opt.image_id+'.json')
#         print(iter_idx)
    for dd in range(opt.repeat):
        for idx, filename in tqdm(enumerate(onlyfiles)):
            if opt.image_id != '' and iter_idx != idx:
                continue
            file_path = os.path.join(opt.test_scene_graphs_dir, filename)
            with open(file_path, 'r') as f:
                scene_graphs = json.load(f)
#             print(scene_graphs)
            input_dict = dict()
            input_dict['image_id'] = str(scene_graphs['image_id'])+'_{}'.format(dd)
            input_dict['dataset_idx'] = scene_graphs['dataset_idx']
            input_dict['tensor_list'] = []

            print(scene_graphs['objects'])
            print(scene_graphs['relationships'])
            # random suffle
            for jj in range(3):
                num_objs = len(scene_graphs['objects'])
                sampling = random.choices([i for i in range(num_objs)], k=2)
                sour_idx, targ_idx = sampling[0], sampling[1]
                sour_obj = scene_graphs['objects'][sour_idx]
                targ_obj = scene_graphs['objects'][targ_idx]
                scene_graphs['objects'][targ_idx] = sour_obj
                scene_graphs['objects'][sour_idx] = targ_obj
                for i in range(len(scene_graphs['relationships'])):
                    if scene_graphs['relationships'][i][0] == sour_idx:
                        scene_graphs['relationships'][i][0] = targ_idx
                    elif scene_graphs['relationships'][i][0] == targ_idx:
                        scene_graphs['relationships'][i][0] = sour_idx
                    if scene_graphs['relationships'][i][2] == sour_idx:
                        scene_graphs['relationships'][i][2] = targ_idx
                    elif scene_graphs['relationships'][i][2] == targ_idx:
                        scene_graphs['relationships'][i][2] = sour_idx
            print(scene_graphs['objects'])
            print(scene_graphs['relationships'])
            
            tmp_name_to_idx = dict()
            for i, (idx, name) in enumerate(infer.vocab_dict.items()):
                tmp_name_to_idx[name] = idx
#             tmp_name_to_idx['None'] = 2
            # [1, 128]
            # create obj_class_id list
            class_ids_list = []
            for obj_name in scene_graphs['objects']:
                class_ids_list.append(tmp_name_to_idx[obj_name])
            obj_ids_list = []
            for i in class_ids_list:
                obj_ids_list.append(0)

            # create input_token
            input_token  = torch.zeros(128).type(torch.int64)
            input_obj_id = torch.zeros(128).type(torch.int64)
            segment_label = torch.zeros(128).type(torch.int64)
            token_type = torch.zeros(128).type(torch.int64)
            input_token[0] = 1
            segment_label[0] = 1

            sent_p = 1
            random.shuffle(scene_graphs['relationships'])
            for i in range(len(scene_graphs['relationships'])):
                pair = scene_graphs['relationships'][i]

                input_token[sent_p]   = class_ids_list[pair[0]]
                input_token[sent_p+1] = tmp_name_to_idx[pair[1]]
                input_token[sent_p+2] = class_ids_list[pair[2]]
                input_token[sent_p+3] = 2

                # save obj_id
                obj_ids_list[pair[0]] = pair[0] + 1
                obj_ids_list[pair[2]] = pair[2] + 1
                if class_ids_list[pair[0]] == 3: obj_ids_list[pair[0]] = 0
                if class_ids_list[pair[2]] == 3: obj_ids_list[pair[2]] = 0

                input_obj_id[sent_p]   = pair[0] + 1
                input_obj_id[sent_p+1] = 0
                input_obj_id[sent_p+2] = pair[2] + 1
                input_obj_id[sent_p+3] = 0
                if class_ids_list[pair[0]] == 3: input_obj_id[sent_p]   = 0
                if class_ids_list[pair[2]] == 3: input_obj_id[sent_p+2] = 0

                segment_label[sent_p] = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+1] = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+2] = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+3] = int((sent_p - 1) / 4 + 1)

                token_type[sent_p] = 1
                token_type[sent_p+1] = 2
                token_type[sent_p+2] = 3

                sent_p += 4

            # __image__
            __image__ = tmp_name_to_idx['__image__']
            __in_image__ = tmp_name_to_idx['__in_image__']
            for i in range(len(class_ids_list)):
                O = class_ids_list[i]
                input_token[sent_p]   = O
                input_token[sent_p+1] = __in_image__
                input_token[sent_p+2] = __image__
                input_token[sent_p+3] = 2

                input_obj_id[sent_p]   = obj_ids_list[i]
                input_obj_id[sent_p+1] = 0
                input_obj_id[sent_p+2] = 0
                input_obj_id[sent_p+3] = 0

                segment_label[sent_p]   = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+1] = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+2] = int((sent_p - 1) / 4 + 1)
                segment_label[sent_p+3] = int((sent_p - 1) / 4 + 1)

                token_type[sent_p]   = 1
                token_type[sent_p+1] = 2
                token_type[sent_p+2] = 3

                sent_p += 4

            tmp_list = []
            tmp_list.append(input_token)
            tmp_list.append(input_obj_id)
            tmp_list.append(segment_label)
            tmp_list.append(token_type)
#             print(tmp_list[0])
            input_dict['tensor_list'] = tmp_list

            infer.check_from_sg(input_dict, model, layout_save=ly_save_path)