import sys
sys.path.append("../")
sys.path.append("./")
from loader import build_loader, RelDataset
import torch
import logging
import random
from model import Rel2Layout, RelEncoder, Rel2Bbox, Rel2RegLayout
from trainer import Trainer, PretrainTrainer, RegTrainer
from utils import ensure_dir
import argparse
import cv2
from bounding_box import bounding_box as bb
import os
import numpy as np
import pickle
from random import randrange

logger = logging.getLogger('inference')

class Inference_VG():
    def __init__(self, save_dir, vocab_dict, cls_dict, all_anns):
        self.save_dir = save_dir
        self.device = self._prepare_gpu()
        self.all_anns = all_anns
        self.vocab_dict = vocab_dict
        self.cls_dict = cls_dict
            
    def check_GT(self, idx, dataset=None):
        all_anns = self.all_anns
        image_id = all_anns['image_id'][idx]
        log_file_name = os.path.join(self.save_dir,str(image_id)+'.png')
        image_wh = [all_anns['image_wh'][idx][0],all_anns['image_wh'][idx][1]]
        image_size = [all_anns['image_wh'][idx][1],all_anns['image_wh'][idx][0],3]
        box_mask = np.array(all_anns['rel_box'][idx]) != 2.
        boxes = np.array(all_anns['rel_box'][idx])[box_mask].reshape(-1,4)
        boxes = self.xywh2xyxy(boxes, image_wh)
        id_mask = np.array(all_anns['id'][idx]) != 0.
        ids = np.array(all_anns['id'][idx])[id_mask]
        clss = self.idx2vocab(all_anns['rel'][idx][1::2][all_anns['rel'][idx][1::2] != 0],'text')
        
#         log_file_name = os.path.join(self.save_dir, str(image_id)+'_gt.txt')
#         self.write_log(all_anns['rel'][idx], all_anns['id'][idx], log_file_name)
        self.draw_img(image_size = image_size, boxes=boxes, labels=clss,
                      label_ids = np.zeros(len(clss)),
                      save_dir = self.save_dir, name=str(image_id)+'_gt')
        
    def check_from_model(self, idx, dataset, model, random=False):
        model.to(self.device)
        model.eval()
        if random == True:
            idx = randrange(len(dataset.data['image_id']))
        image_id = dataset.data['image_id'][idx]
        single_data = dataset[idx]
        image_wh = dataset.data['image_wh'][idx]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        
        input_token = single_data[0].unsqueeze(0).to(self.device)
        input_obj_id = single_data[1].unsqueeze(0).to(self.device)
        segment_label = single_data[4].unsqueeze(0).to(self.device)
        token_type = single_data[5].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)

        vocab_logits, token_type_logits, output_box, refine_box = \
            model(input_token, input_obj_id, segment_label, token_type, src_mask)
        if type(refine_box) != None:
            output_box[:,1::2,:] = refine_box
        # construct mask
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))

        for i in range(1, int(max(input_obj_id_list))+1):
            mask[0, input_obj_id_list.index(i)] = 1
        mask = mask.bool()

        pred_vocab = vocab_logits.argmax(2)
#         pred_mask = (pred_vocab >= 4) * (pred_vocab < 176)
        pred_vocab = pred_vocab[mask].detach()
        output_boxes = output_box[mask].detach()
        output_class_ids = input_obj_id[mask].detach()
        
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)
        pred_classes = self.idx2vocab(pred_vocab, 'text')
        
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
#         print(pred_classes)
#         print(output_sentence)
        
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id)
        self.write_log(output_sentence, input_obj_id_list, log_file_name)
        
#     def predict_val(self, )
    def draw_img(self, image_size, boxes, labels, label_ids, save_dir, name):
        # setting color
        color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
                'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
        random.shuffle(color)
        labels_no_repeat = list(set(labels))
        image = np.full(image_size, 200.)
        for i in range(len(boxes)):
            bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                   str(labels[i]+'[{}]'.format(label_ids[i])), 
                   color=color[labels_no_repeat.index(labels[i])])
        self.show_and_save(image, os.path.join(save_dir,str(name)+'.png'))
        logger.info("Save image in {}".format(os.path.join(save_dir,str(name)+'.png')))
        
    def write_log(self, sentence, class_ids, log_file_name):
        f = open(log_file_name, 'w')

        for i in range(1, len(sentence), 4):
            if sentence[i] == '[SEP]' and sentence[i+1] == '[SEP]':
                break
            single_pair = ''
            single_pair += sentence[i] + '[{}]'.format(class_ids[i]) + ' '
            single_pair += sentence[i+1] + ' '
            single_pair += sentence[i+2] + '[{}]'.format(class_ids[i+2]) + ' '
            single_pair += sentence[i+3] + '\n'
            f.write(single_pair)
        logger.info("Save log file in {}".format(log_file_name))
        
    def xywh2xyxy(self, boxes, image_wh):
        boxes[:,:2] = boxes[:,:2] 
        boxes[:,2:] = boxes[:,:2] + boxes[:,2:]
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def xcycwh2xyxy(self, boxes, image_wh):
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        return boxes
    
    def _prepare_gpu(self):
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        return device
    
    def idx2vocab(self, idx, modality):
        sent = []
        for i in range(len(idx)):
            if modality == 'text':
                sent.append(self.vocab_dict[int(idx[i])])
            else:
                sent.append(self.cls_dict[idx[i]])
        return sent
    
    def show_and_save(self, image, path):
        cv2.imwrite(path, image)
        
    def run(self, cfg, model, dataset):
        if cfg['TEST']['MODE'] == 'gt':
            if cfg['TEST']['RANDOM']: logger.warning('Test gt mode do not support random.')
            self.check_GT(cfg['TEST']['SAMPLE_IDX'])
        elif cfg['TEST']['MODE'] == 'model':
            if cfg['TEST']['RANDOM']: 
                self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset, model, 
                                       random=True)
            else: self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset=dataset,
                                         model=model)
        else:
            logger.error('We only support gt and model test mode.')
        
if __name__ == '__main__':
    with open('./data/vg/rel_dict_45.pkl', 'rb') as file:
        vocab_dict = pickle.load(file)
    with open('./data/vg/cls_dict_45.pkl', 'rb') as file:
        cls_dict = pickle.load(file)
    data_dir = './data/vg/'
    anns_file_name = 'vg_anns_45.pkl'
    model_path = './saved/pretrained_vg_N_v0/checkpoint_50.pth'
    fn = './data/vg/vg_anns_45.pkl'
    with open(fn, 'rb') as file:
        all_anns = pickle.load(file)
        
    D = RelDataset(smart_sampling = True, data_dir=data_dir, anns_file_name=anns_file_name)
    
    model = RelEncoder(vocab_size=205, obj_classes_size=150,hidden_size=256, num_layers=4, attn_heads=4, dropout=0.1)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    infer = Inference_VG(save_dir = './', vocab_dict=vocab_dict, cls_dict=cls_dict,
                        all_anns=all_anns)
    infer.check_from_model(99, D, model)
#     infer.check_GT(99)