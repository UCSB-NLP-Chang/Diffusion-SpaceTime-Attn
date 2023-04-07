import os
import sys
import torch
import logging
import random
import torch.utils.data as data
import numpy as np
from .base_data_loader import BaseDataLoader
from torch.utils.data.dataloader import default_collate
import pickle
from .COCODataset import COCORelDataset, COCOLayoutDataset

class BboxDataset(data.Dataset):
    def __init__(self):

        fn = 'utils/val_input.pkl'

        with open(fn, 'rb') as file:
            self.data = pickle.load(file)
        
    def __getitem__(self, index):
        single_ann = self.data[index]
        cats_id = single_ann['categories']
        center_pos = single_ann['position']
        shape_centroid = single_ann['shape']
        caption = single_ann['caption']
        caption = "[CLS] " + caption + " [SEP]" 
        return cats_id, center_pos, shape_centroid, caption

    def __len__(self):
        return len(self.data)

class RelDataset(data.Dataset):
    def __init__(self, smart_sampling = False, data_dir=None, anns_file_name=None):
        
        fn = os.path.join(data_dir, anns_file_name)
        self.smart_sampling = smart_sampling
        with open(fn, 'rb') as file:
            self.data = pickle.load(file)
        self.rel_data = self.data['rel']
        self.id_data = self.data['id']
        self.rel_box = self.data['rel_box']
            
    def __getitem__(self, index):
        sent = self.rel_data[index]
        obj_id = self.id_data[index]
        box_xy = self.rel_box[index]
        
        if self.smart_sampling:
            input_token, input_obj_id, output_label, segment_label, token_type,\
            input_box_label = self.smart_random_word(sent, obj_id, box_xy)
        else:
            input_token, input_obj_id, output_label, segment_label, token_type,\
            input_box_label = self.random_word(sent, obj_id, box_xy)

        return torch.tensor(input_token), torch.tensor(input_obj_id),\
               torch.tensor(box_xy), torch.tensor(output_label), \
               torch.tensor(segment_label), torch.tensor(token_type)

    def __len__(self):
        return len(self.rel_data)

    def random_word(self, sentence, obj_id, box_xy):
        '''
        PAD = 0, CLS = 1, SEP = 2, MASK = 3
        Subj = 1, Rel = 2, Obj = 3
        Box -> 2. for ignore
        '''
        temp_sentence = sentence.copy()
        temp_obj_id = obj_id.copy()
        temp_box_xy = box_xy.copy()
        output_label = []
        output_box_label = []
        segment_label = []
        token_type = []
        segment_idx = 1
        assert len(temp_obj_id) == len(temp_sentence) == len(temp_box_xy)
        for i in range(len(temp_sentence)):
            prob = random.random()
            if prob < 0.15 and temp_sentence[i] > 0:
                prob /= 0.15
                label = temp_sentence[i].copy()
                label_box = temp_box_xy[i].copy()
                if prob < 0.8:
                    temp_sentence[i] = 3
                output_label.append(label)
                output_box_label.append(label_box)
            else:
                output_label.append(0)
                output_box_label.append([2.,2.,2.,2.])

            if temp_sentence[i] > 0:
                segment_label.append(segment_idx)
                token_type.append(i % 4)
                if temp_sentence[i] == 2:
                    segment_idx += 1
            else:
                token_type.append(0)
                segment_label.append(0)


        return temp_sentence, temp_obj_id, output_label, segment_label, token_type,\
                output_box_label
    
    def smart_random_word(self, sentence, obj_id, box_xy):
        '''
        PAD = 0, CLS = 1, SEP = 2, MASK = 3
        Subj = 1, Rel = 2, Obj = 3
        sentence : 1 o o o 2 o o o 2 o o o 2 o o o ...
        '''
#         print(sentence)
        temp_sentence = sentence.copy()
        temp_obj_id = obj_id.copy()
        temp_box_xy = box_xy.copy()
        output_label = []
        output_box_label = []
        segment_label = []
        token_type = []
        segment_idx = 1
        flag_is_mask = False
        num_pair = ((temp_sentence != 0) * (temp_sentence != 1) * \
                    (temp_sentence != 2) * (temp_sentence != 3)).sum() / 3
        assert len(temp_obj_id) == len(temp_sentence) == len(temp_box_xy)
        if num_pair > 2:
            for i in range(len(temp_sentence)):
                prob = random.random()
                if temp_sentence[i] == 0:
                    output_label.append(0)
                    output_box_label.append([2.,2.,2.,2.])
                elif prob < 0.15 and temp_sentence[i] > 0 and temp_sentence[i] < 4 and \
                    (i == 0 or ( i - 1 ) % 4 == 3):
                    prob /= 0.15
                    label = temp_sentence[i].copy()
                    if prob < 0.8:
                        temp_sentence[i] = 3
#                         temp_obj_id[i] = 0
                    output_label.append(label)
                    output_box_label.append([2.,2.,2.,2.])
                elif prob >= 0.15 and temp_sentence[i] > 0 and temp_sentence[i] < 4 and \
                    (i == 0 or ( i - 1 ) % 4 == 3):
                    output_label.append(0)
                    output_box_label.append([2.,2.,2.,2.])
                elif prob < 0.45 and temp_sentence[i] > 3 and ( i - 1 ) % 4 == 0:
                    label_box = temp_box_xy[i].copy()
                    output_box_label.append(label_box)
                    output_box_label.append([2.,2.,2.,2.])
                    label_box = temp_box_xy[i + 2].copy()
                    output_box_label.append(label_box)
                    
                    prob /= 0.45
                    if prob < 1/3.: 
                        prob = random.random()
                        label = temp_sentence[i].copy()
                        if prob < 0.8:
                            temp_sentence[i] = 3
#                             temp_obj_id[i] = 0
                        output_label.append(label)
                        output_label.append(0)
                        output_label.append(0)
                    elif prob >= 1/3. and prob < 2/3.: 
                        prob = random.random()
                        label = temp_sentence[i+1].copy()
                        if prob < 0.8:
                            temp_sentence[i+1] = 3
#                             temp_obj_id[i+1] = 0
                        output_label.append(0)
                        output_label.append(label)
                        output_label.append(0)
                    else: 
                        prob = random.random()
                        label = temp_sentence[i+2].copy()
                        if prob < 0.8:
                            temp_sentence[i+2] = 3
#                             temp_obj_id[i+2] = 0
                        output_label.append(0)
                        output_label.append(0)
                        output_label.append(label)

                elif prob >= 0.45 and temp_sentence[i] > 3 and ( i - 1 ) % 4 == 0:
                    output_label.append(0)
                    output_label.append(0)
                    output_label.append(0)
                    label_box = temp_box_xy[i].copy()
                    output_box_label.append(label_box)
                    output_box_label.append([2.,2.,2.,2.])
                    label_box = temp_box_xy[i + 2].copy()
                    output_box_label.append(label_box)

                if temp_sentence[i] > 0:
                    segment_label.append(segment_idx)
                    token_type.append(i % 4)
                    if temp_sentence[i] == 2:
                        segment_idx += 1
                else:
                    token_type.append(0)
                    segment_label.append(0)
        else:
            # mask relationship only
            rel_index = (((temp_sentence != 0) * (temp_sentence != 1) * \
                         (temp_sentence != 2) * (temp_sentence != 3)) == True).nonzero()[0]\
                         + 1
            for i in range(len(temp_sentence)):
                prob = random.random()
                if prob < 0.45 and temp_sentence[i] > 0:
                    prob /= 0.45
                    label = temp_sentence[i].copy()
                    if prob < 0.8 and i == rel_index[0]:
                        temp_sentence[i] = 3
#                         temp_obj_id[i] = 0
                    output_label.append(label)
                    output_box_label.append([2.,2.,2.,2.])
                else:
                    output_label.append(0)
                    output_box_label.append([2.,2.,2.,2.])
                
                if temp_sentence[i] > 0:
                    segment_label.append(segment_idx)
                    token_type.append(i % 4)
                    if temp_sentence[i] == 2:
                        segment_idx += 1
                else:
                    token_type.append(0)
                    segment_label.append(0)

        return temp_sentence, temp_obj_id, output_label, segment_label, token_type,\
                output_box_label

class Rel2Layout_Dataset(data.Dataset):
    def __init__(self, data_dir=None, anns_file_name=None):

        fn = os.path.join(data_dir, anns_file_name)

        with open(fn, 'rb') as file:
            all_anns = pickle.load(file)

        self.word = all_anns['rel']
        self.id = all_anns['id']
        self.cls = all_anns['cls']
        self.pos = all_anns['pos']
        self.shape = all_anns['shape']
        self.box_xy = all_anns['box_xy']
        self.image_id = all_anns['image_id']
        self.image_wh = all_anns['image_wh']
        
        assert len(self.word) == len(self.cls), 'Word and cls has difference length!'
        assert len(self.word) == len(self.pos), 'Word and pos has difference length!'
        assert len(self.word) == len(self.shape), 'Word and shape has difference length!'
        assert len(self.word) == len(self.box_xy), 'Word and box_xy has difference length!'
        assert len(self.word) == len(self.id), 'Word and id has difference length!'

    def __getitem__(self, index):
        sent = self.word[index]
        obj_id = self.id[index]
        output_cls = self.cls[index]
        output_pos = self.pos[index]
        output_shape = self.shape[index]
        output_box_xy = self.box_xy[index]
        
        input_token, segment_label, token_type = self.process_word(sent)
        return torch.tensor(input_token), torch.tensor(segment_label), torch.tensor(token_type),\
               torch.tensor(output_cls), torch.tensor(output_box_xy).float(), torch.tensor(obj_id)

    def __len__(self):
        return len(self.word)

    def process_word(self, sentence):
        '''
        PAD = 0, CLS = 1, SEP = 2, MASK = 3
        Subj = 1, Rel = 2, Obj = 3
        '''
        segment_label = []
        token_type = []
        segment_idx = 1
        for i in range(len(sentence)):
            if sentence[i] > 0:
                segment_label.append(segment_idx)
                token_type.append(i % 4)
                if sentence[i] == 2:
                    segment_idx += 1
            else:
                token_type.append(0)
                segment_label.append(0)

        return sentence, segment_label, token_type



