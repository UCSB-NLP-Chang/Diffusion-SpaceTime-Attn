import numpy as np
import logging
import skimage as io
from tqdm import tqdm
import matplotlib as mpl
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import pylab
import urllib
from io import BytesIO
import requests as req
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle
import json, os, random, math
from collections import defaultdict
import torch
from torch.utils.data import Dataset

logger = logging.getLogger('VGmsdnRelDataset')

class VGmsdnRelDataset(Dataset):
    def __init__(self, instances_json_path, category_json_path, dict_save_path,
               sentence_size=128, is_mask=True, is_std=False, add_coco_rel=False, 
                 reverse=False):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        - 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        - [PAD], [CLS], [SEP], [MASK]
        """

        super(Dataset, self).__init__()
        self.is_std = is_std
        self.is_mask = is_mask
        self.reverse = reverse
        self.sentence_size = sentence_size
        self.add_coco_rel = add_coco_rel
        if add_coco_rel: logger.info("Using coco relationship plugin!")
        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f)
        with open(category_json_path, 'r') as f:
            category_data = json.load(f)


        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data:
            image_id = image_data['id']
            filename = image_data['path']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
          'object_name_to_idx': {},
          'object_idx_to_name': {},
          'pred_name_to_idx': {},
          'pred_idx_to_name': {},
          'object_pred_name_to_idx': {},
          'object_pred_idx_to_name': {},
        }
        # setting predictes
        self.snetence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']

        # pred_name_to_idx
        self.vocab['pred_name_to_idx'] = {}
        self.vocab['pred_name_to_idx']['__in_image__'] = 0
        for idx, name in enumerate(category_data['predicate']):
            self.vocab['pred_name_to_idx'][name] = idx + 1
        if add_coco_rel:
            before_len = len(self.vocab['pred_name_to_idx'])
            coco_rel = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']
            for idx, name in enumerate(coco_rel):
                self.vocab['pred_name_to_idx'][name] = idx + before_len

        # object_name_to_idx
        for idx, token in enumerate(self.snetence_token):
            self.vocab['object_name_to_idx'][token] = idx
        self.vocab['object_name_to_idx']['__image__'] = len(self.snetence_token)
        for idx, name in enumerate(category_data['object']):
            self.vocab['object_name_to_idx'][name] = idx + len(self.snetence_token) + 1

        # object_idx_to_name
        for name, idx in self.vocab['object_name_to_idx'].items():
            self.vocab['object_idx_to_name'][idx] = name

        # pred_idx_to_name
        for name, idx in self.vocab['pred_name_to_idx'].items():
            self.vocab['pred_idx_to_name'][idx] = name


        all_vocabs = []
        for idx, name in enumerate(self.vocab['object_name_to_idx'].keys()):
            all_vocabs.append(name)
        for idx, name in enumerate(self.vocab['pred_name_to_idx'].keys()):
            all_vocabs.append(name)
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_idx_to_name'][i] = all_vocabs[i]

        # Add object data from instances
        self.image_id_to_instances = defaultdict(list)
        for i in range(len(instances_data)):
            image_id = instances_data[i]['id']
            self.image_id_to_instances[image_id] = instances_data[i]
        
        with open(dict_save_path, 'wb+') as file:
            pickle.dump(self.vocab['object_pred_idx_to_name'], file)
            
        # boxes = [xc, yc, w, h] normalized
        all_boxes = []
        for i in range(len(instances_data)):
            W = instances_data[i]['width']
            H = instances_data[i]['height']
            for obj in instances_data[i]['objects']:
                x0, y0, x1, y1 = obj['box']
                all_boxes.append([(x0+x1)/(2*W), (y0+y1)/(2*H), (x1-x0)/W, (y1-y0)/H])

                
        all_boxes = np.array(all_boxes)
        self.x_mean, self.x_std = all_boxes[:,0].mean(), all_boxes[:,0].std()
        self.y_mean, self.y_std = all_boxes[:,1].mean(), all_boxes[:,1].std()
        self.w_mean, self.w_std = all_boxes[:,2].mean(), all_boxes[:,2].std()
        self.h_mean, self.h_std = all_boxes[:,3].mean(), all_boxes[:,3].std()
        sta_dict = {}
        sta_dict['x_mean'], sta_dict['x_std'] = self.x_mean, self.x_std
        sta_dict['y_mean'], sta_dict['y_std'] = self.y_mean, self.y_std
        sta_dict['w_mean'], sta_dict['w_std'] = self.w_mean, self.w_std
        sta_dict['h_mean'], sta_dict['h_std'] = self.h_mean, self.h_std
        
        sta_dict_path = os.path.dirname(instances_json_path)
        with open(os.path.join(sta_dict_path,'sta_dict.json'), 'w') as fp:
            json.dump(sta_dict, fp)

    def __len__(self):
        return len(self.image_ids)
    
    def sta_normalized(self, box):
        """
        (x-mean)/std
        """
        box[0] = (box[0]-self.x_mean)/self.x_std
        box[1] = (box[1]-self.y_mean)/self.y_std
        box[2] = (box[2]-self.w_mean)/self.w_std
        box[3] = (box[3]-self.h_mean)/self.h_std
        return box
        
    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]
        blank_box = [2., 2., 2., 2.]
        
        W, H = self.image_id_to_size[image_id]

        objs, obj_ids, boxes = [], [], []
        triples = []
        triples_ids = []
        triples_boxes = []
        instances = self.image_id_to_instances[image_id]
        exist_objs = []
        for rel in instances['relationships']:
            # triples part
            if not self.reverse:
                sub_id_raw = rel['sub_id']
                obj_id_raw = rel['obj_id']
            else:
                sub_id_raw = rel['obj_id']
                obj_id_raw = rel['sub_id']
            sub_name = instances['objects'][sub_id_raw]['class']
            pred_name = rel['predicate']
            obj_name = instances['objects'][obj_id_raw]['class']
            sub_idx = self.vocab['object_pred_name_to_idx'][sub_name]
            pred_idx = self.vocab['object_pred_name_to_idx'][pred_name]
            obj_idx = self.vocab['object_pred_name_to_idx'][obj_name]
            triples.append([sub_idx, pred_idx, obj_idx])
            
            # triples_ids part
            sub_id = sub_id_raw + 1
            obj_id = obj_id_raw + 1
            triples_ids.append([sub_id, 0, obj_id])
            
            
            # triples_boxes part
            # box = [xc, yc, w, h/w]
            x0, y0, x1, y1 = instances['objects'][sub_id_raw]['box']

            sub_box = [(x0+x1)/(2*W), (y0+y1)/(2*H), (x1-x0)/W, (y1-y0)/H]
            if self.is_std:
                sub_box = self.sta_normalized(sub_box)
            
            x0, y0, x1, y1 = instances['objects'][obj_id_raw]['box']

            obj_box = [(x0+x1)/(2*W), (y0+y1)/(2*H), (x1-x0)/W, (y1-y0)/H]
#                 sub_box = self.sta_normalized(sub_box)
            if self.is_std:
                obj_box = self.sta_normalized(obj_box)

            rel_box = list(np.array(sub_box) - np.array(obj_box))
            pred_box = rel_box
                
            triples_boxes.append([sub_box, pred_box, obj_box])
            if sub_id not in exist_objs:
                exist_objs.append(sub_id)
                objs.append(sub_idx)
                obj_ids.append(sub_id)
                boxes.append(sub_box)
            if obj_id not in exist_objs:
                exist_objs.append(obj_id)
                objs.append(obj_idx)
                obj_ids.append(obj_id)
                boxes.append(obj_box)
                
        #################################
        # Add other relationship
        #################################
        if self.add_coco_rel:
            obj_centers = []
            for i, obj_idx in enumerate(objs):
                xc, yc, w, h = boxes[i]
                mean_x = xc
                mean_y = yc
                obj_centers.append([mean_x, mean_y])
            obj_centers = torch.FloatTensor(obj_centers)

            num_objs = len(objs)
    #         __image__ = self.vocab['object_pred_name_to_idx']['__image__']
            real_objs = []
            if num_objs > 1:
                real_objs = (torch.LongTensor(objs) != 0).nonzero().squeeze(1)
            for cur in real_objs:
                choices = [obj for obj in real_objs if obj != cur]
                if len(choices) == 0:
                    break
                other = random.choice(choices)
                if random.random() > 0.5:
                    s, o = cur, other
                else:
                    s, o = other, cur

                # Check for inside / surrounding
                sxc, syc, sw, sh = boxes[s]
                oxc, oyc, ow, oh = boxes[o]
                sx0, sy0, sx1, sy1 = sxc-sw/2., syc-sh/2., sxc+sw/2., syc+sh/2.
                ox0, oy0, ox1, oy1 = oxc-ow/2., oyc-oh/2., oxc+ow/2., oyc+oh/2.
                d = obj_centers[s] - obj_centers[o]
                theta = math.atan2(d[1], d[0])

                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = 'surrounding'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = 'inside'
                elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                    p = 'left of'
                elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                    p = 'above'
                elif -math.pi / 4 <= theta < math.pi / 4:
                    p = 'right of'
                elif math.pi / 4 <= theta < 3 * math.pi / 4:
                    p = 'below'
                p = self.vocab['object_pred_name_to_idx'][p]

                if not self.reverse:
                    triples.append([objs[s], p, objs[o]])
                    triples_ids.append([obj_ids[s], 0, obj_ids[o]])
                    rel_box = list(np.array([(sx0+sx1)/2., (sy0+sy1)/2., sx1-sx0, sy1-sy0]) - np.array([(ox0+ox1)/2., (oy0+oy1)/2., ox1-ox0, oy1-oy0]))
                    triples_boxes.append([[(sx0+sx1)/2., (sy0+sy1)/2., sx1-sx0, sy1-sy0],
                                          rel_box, 
                                          [(ox0+ox1)/2., (oy0+oy1)/2., ox1-ox0, oy1-oy0]])
                else:
                    triples.append([objs[o], p, objs[s]])
                    triples_ids.append([obj_ids[o], 0, obj_ids[s]])
                    rel_box = list(np.array([(sx0+sx1)/2., (sy0+sy1)/2., sx1-sx0, sy1-sy0]) - np.array([(ox0+ox1)/2., (oy0+oy1)/2., ox1-ox0, oy1-oy0]))
                    triples_boxes.append([[(ox0+ox1)/2., (oy0+oy1)/2., ox1-ox0, oy1-oy0],
                                          rel_box, 
                                          [(sx0+sx1)/2., (sy0+sy1)/2., sx1-sx0, sy1-sy0]])
        # Add dummy __image__ object
#         objs.append(self.vocab['object_pred_name_to_idx']['__image__'])
#         obj_ids.append(0)
#         # box = [xc, yc, w, h/w]
#         boxes.append([0.5, 0.5, 1, 1])

#         O = len(objs)
#         in_image = self.vocab['object_pred_name_to_idx']['__in_image__']
#         for i in range(O - 1):
#             triples.append([objs[i], in_image, objs[O - 1]])
#             triples_ids.append([obj_ids[i], 0, obj_ids[O - 1]])
#             triples_boxes.append([boxes[i], blank_box, boxes[O - 1]])
            
        ############################################
        # To sentence, Language model
        # 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        # - [PAD], [CLS], [SEP], [MASK]
        START_TOKEN = 1
        SEPERATE_TOKEN = 2
        PAD_TOKEN = 0
        complete_sentence = []
        complete_object_ids = []
        complete_boxes = []
        complete_sentence.append(START_TOKEN)
        complete_object_ids.append(0)
        complete_boxes.append(blank_box)
        assert len(triples) == len(triples_ids) == len(triples_boxes)
        for i in range(len(triples)):
            for j in range(len(triples[i])):
                complete_sentence.append(triples[i][j])
                complete_object_ids.append(triples_ids[i][j])
                complete_boxes.append(triples_boxes[i][j])
            complete_sentence.append(SEPERATE_TOKEN)
            complete_object_ids.append(0)
            complete_boxes.append(blank_box)

        assert len(complete_sentence) == len(complete_object_ids) == len(complete_boxes)

        # padding part
        if self.sentence_size >= len(complete_sentence):
            for i in range(self.sentence_size - len(complete_sentence)):
                complete_sentence.append(PAD_TOKEN)
                complete_object_ids.append(0)
                complete_boxes.append(blank_box)
        else:
            complete_sentence = complete_sentence[:self.sentence_size]
            complete_object_ids = complete_object_ids[:self.sentence_size]
            complete_boxes = complete_boxes[:self.sentence_size]

        complete_sentence = np.array(complete_sentence)
        complete_object_ids = np.array(complete_object_ids)
        complete_boxes = np.array(complete_boxes)

        input_token, input_obj_id, output_obj_id, output_label, segment_label, token_type,\
        input_box_label = \
        self.smart_random_word(complete_sentence, complete_object_ids, complete_boxes, reverse=self.reverse, is_mask=self.is_mask)

        
        return torch.tensor(input_token), torch.tensor(input_obj_id),\
               torch.tensor(output_obj_id),\
               torch.tensor(complete_boxes).float(), torch.tensor(output_label), \
               torch.tensor(segment_label), torch.tensor(token_type)

    def smart_random_word(self, sentence, obj_id, box_xy, reverse=False, is_mask=True):
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
        output_obj_id = []
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
                    output_obj_id.append(0)
                    output_label.append(0)
                    output_box_label.append([2.,2.,2.,2.])
                elif prob < 0.15 and temp_sentence[i] > 0 and temp_sentence[i] < 4 and \
                    (i == 0 or ( i - 1 ) % 4 == 3):
                    prob /= 0.15
                    label = temp_sentence[i].copy()
                    obj_id = temp_obj_id[i].copy()
                    if prob < 0.8 and is_mask:
                        temp_sentence[i] = 3
                        temp_obj_id[i] = 0
                    output_obj_id.append(obj_id)
                    output_label.append(label)
                    output_box_label.append([2.,2.,2.,2.])
                elif prob >= 0.15 and temp_sentence[i] > 0 and temp_sentence[i] < 4 and \
                    (i == 0 or ( i - 1 ) % 4 == 3):
                    output_obj_id.append(0)
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
                        obj_id = temp_obj_id[i].copy()
                        if prob < 0.8 and is_mask:
                            temp_sentence[i] = 3
                            temp_obj_id[i] = 0
                        output_obj_id.append(obj_id)
                        output_obj_id.append(0)
                        output_obj_id.append(0)
                        output_label.append(label)
                        output_label.append(0)
                        output_label.append(0)
                    elif prob >= 1/3. and prob < 2/3.: 
                        prob = random.random()
                        label = temp_sentence[i+1].copy()
                        obj_id = temp_obj_id[i+1].copy()
                        if prob < 0.8 and is_mask:
                            temp_sentence[i+1] = 3
                            temp_obj_id[i+1] = 0
                        output_obj_id.append(0)
                        output_obj_id.append(obj_id)
                        output_obj_id.append(0)
                        output_label.append(0)
                        output_label.append(label)
                        output_label.append(0)
                    else: 
                        prob = random.random()
                        label = temp_sentence[i+2].copy()
                        obj_id = temp_obj_id[i+2].copy()
                        if prob < 0.8 and is_mask:
                            temp_sentence[i+2] = 3
                            temp_obj_id[i+2] = 0
                        output_obj_id.append(0)
                        output_obj_id.append(0)
                        output_obj_id.append(obj_id)
                        output_label.append(0)
                        output_label.append(0)
                        output_label.append(label)

                elif prob >= 0.45 and temp_sentence[i] > 3 and ( i - 1 ) % 4 == 0:
                    output_label.append(0)
                    output_label.append(0)
                    output_label.append(0)
                    output_obj_id.append(0)
                    output_obj_id.append(0)
                    output_obj_id.append(0)
                    output_box_label.append([2.,2.,2.,2.])
                    output_box_label.append([2.,2.,2.,2.])
                    output_box_label.append([2.,2.,2.,2.])

                if temp_sentence[i] > 0:
                    segment_label.append(segment_idx)
                    if reverse:
                        if i % 4 == 0:
                            token_type.append(0)
                        elif i % 4 == 1:
                            token_type.append(3)
                        elif i % 4 == 2:
                            token_type.append(2)
                        else:
                            token_type.append(1)
                    else:
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
                    obj_id = temp_obj_id[i].copy()
                    if prob < 0.8 and i == rel_index[0] and is_mask:
                        temp_sentence[i] = 3
                        temp_obj_id[i] = 0
                    output_label.append(label)
                    output_obj_id.append(obj_id)
                    output_box_label.append([2.,2.,2.,2.])
                else:
                    output_label.append(0)
                    output_obj_id.append(0)
                    output_box_label.append([2.,2.,2.,2.])

                if temp_sentence[i] > 0:
                    segment_label.append(segment_idx)
                    if reverse:
                        if i % 4 == 0:
                            token_type.append(0)
                        elif i % 4 == 1:
                            token_type.append(3)
                        elif i % 4 == 2:
                            token_type.append(2)
                        else:
                            token_type.append(1)
                    else:
                        token_type.append(i % 4)
                    if temp_sentence[i] == 2:
                        segment_idx += 1
                else:
                    token_type.append(0)
                    segment_label.append(0)

        return temp_sentence, temp_obj_id, output_obj_id, output_label, segment_label, token_type, output_box_label


class VGmsdnLayoutDataset(Dataset):
    def __init__(self, instances_json_path, category_json_path, dict_save_path,
               sentence_size=128, is_mask=True):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        - 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        - [PAD], [CLS], [SEP], [MASK]
        """

        super(Dataset, self).__init__()

        self.is_mask = is_mask
        self.sentence_size = sentence_size

        with open(instances_json_path, 'r') as f:
            instances_data = json.load(f)
        with open(category_json_path, 'r') as f:
            category_data = json.load(f)


        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data:
            image_id = image_data['id']
            filename = image_data['path']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
          'object_name_to_idx': {},
          'object_idx_to_name': {},
          'pred_name_to_idx': {},
          'pred_idx_to_name': {},
          'object_pred_name_to_idx': {},
          'object_pred_idx_to_name': {},
        }
        # setting predictes
        self.snetence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']

        # pred_name_to_idx
        self.vocab['pred_name_to_idx'] = {}
        self.vocab['pred_name_to_idx']['__in_image__'] = 0
        for idx, name in enumerate(category_data['predicate']):
            self.vocab['pred_name_to_idx'][name] = idx + 1
            

        # object_name_to_idx
        for idx, token in enumerate(self.snetence_token):
            self.vocab['object_name_to_idx'][token] = idx
        self.vocab['object_name_to_idx']['__image__'] = len(self.snetence_token)
        for idx, name in enumerate(category_data['object']):
            self.vocab['object_name_to_idx'][name] = idx + len(self.snetence_token) + 1

        # object_idx_to_name
        for name, idx in self.vocab['object_name_to_idx'].items():
            self.vocab['object_idx_to_name'][idx] = name

        # pred_idx_to_name
        for name, idx in self.vocab['pred_name_to_idx'].items():
            self.vocab['pred_idx_to_name'][idx] = name


        all_vocabs = []
        for idx, name in enumerate(self.vocab['object_name_to_idx'].keys()):
            all_vocabs.append(name)
        for idx, name in enumerate(self.vocab['pred_name_to_idx'].keys()):
            all_vocabs.append(name)
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_idx_to_name'][i] = all_vocabs[i]

        # Add object data from instances
        self.image_id_to_instances = defaultdict(list)
        for i in range(len(instances_data)):
            image_id = instances_data[i]['id']
            self.image_id_to_instances[image_id] = instances_data[i]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]
        blank_box = [2., 2., 2., 2.]
        
        W, H = self.image_id_to_size[image_id]

        objs, obj_ids, boxes = [], [], []
        triples = []
        triples_ids = []
        triples_boxes = []
        instances = self.image_id_to_instances[image_id]
        exist_objs = []
        for rel in instances['relationships']:
            # triples part
            sub_name = instances['objects'][rel['sub_id']]['class']
            pred_name = rel['predicate']
            obj_name = instances['objects'][rel['obj_id']]['class']
            sub_idx = self.vocab['object_pred_name_to_idx'][sub_name]
            pred_idx = self.vocab['object_pred_name_to_idx'][pred_name]
            obj_idx = self.vocab['object_pred_name_to_idx'][obj_name]
            triples.append([sub_idx, pred_idx, obj_idx])
            
            # triples_ids part
            sub_id = rel['sub_id'] + 1
            obj_id = rel['obj_id'] + 1
            triples_ids.append([sub_id, 0, obj_id])
            
            
            # triples_boxes part
            x0, y0, x1, y1 = instances['objects'][rel['sub_id']]['box']
            sub_box = [x0/W, y0/H, (x1-x0)/W, (y1-y0)/H]
            pred_box = blank_box
            x0, y0, x1, y1 = instances['objects'][rel['obj_id']]['box']
            obj_box = [x0/W, y0/H, (x1-x0)/W, (y1-y0)/H]
            triples_boxes.append([sub_box, pred_box, obj_box])
            if sub_id not in exist_objs:
                exist_objs.append(sub_id)
                objs.append(sub_idx)
                obj_ids.append(sub_id)
                boxes.append(sub_box)
            if obj_id not in exist_objs:
                exist_objs.append(obj_id)
                objs.append(obj_idx)
                obj_ids.append(obj_id)
                boxes.append(obj_box)
            
        # Add dummy __image__ object
        objs.append(self.vocab['object_pred_name_to_idx']['__image__'])
        obj_ids.append(0)
        boxes.append([0, 0, 1, 1])

        O = len(objs)
        in_image = self.vocab['object_pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([objs[i], in_image, objs[O - 1]])
            triples_ids.append([obj_ids[i], 0, obj_ids[O - 1]])
            triples_boxes.append([boxes[i], blank_box, boxes[O - 1]])
            
        ############################################
        # To snetence, Language model
        # 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        # - [PAD], [CLS], [SEP], [MASK]
        START_TOKEN = 1
        SEPERATE_TOKEN = 2
        PAD_TOKEN = 0
        complete_sentence = []
        complete_object_ids = []
        complete_boxes = []
        complete_sentence.append(START_TOKEN)
        complete_object_ids.append(0)
        complete_boxes.append(blank_box)
        assert len(triples) == len(triples_ids) == len(triples_boxes)
        for i in range(len(triples)):
            for j in range(len(triples[i])):
                complete_sentence.append(triples[i][j])
                complete_object_ids.append(triples_ids[i][j])
                complete_boxes.append(triples_boxes[i][j])
            complete_sentence.append(SEPERATE_TOKEN)
            complete_object_ids.append(0)
            complete_boxes.append(blank_box)

        assert len(complete_sentence) == len(complete_object_ids) == len(complete_boxes)

        # padding part
        if self.sentence_size >= len(complete_sentence):
            for i in range(self.sentence_size - len(complete_sentence)):
                complete_sentence.append(PAD_TOKEN)
                complete_object_ids.append(0)
                complete_boxes.append(blank_box)
        else:
            complete_sentence = complete_sentence[:self.sentence_size]
            complete_object_ids = complete_object_ids[:self.sentence_size]
            complete_boxes = complete_boxes[:self.sentence_size]

        complete_sentence = np.array(complete_sentence)
        complete_object_ids = np.array(complete_object_ids)
        complete_boxes = np.array(complete_boxes)

        image_boxes = []
        image_classes = []
        for i in range(1,complete_object_ids.max()+1):
            idx = np.where(complete_object_ids==i)[0][0]
            image_classes.append(complete_sentence[idx])
            image_boxes.append(complete_boxes[idx])

        image_classes = np.array(image_classes)
        image_boxes = np.array(image_boxes)

        ## padding image_classes
        image_classes = np.insert(image_classes, 0, 1)
        image_classes = np.append(image_classes, [2])
        if len(image_classes) < self.sentence_size:
            image_classes = np.pad(image_classes, ((0, self.sentence_size - len(image_classes))), 'constant', constant_values = 0)

        ## padding image_classes
        image_boxes = np.insert(image_boxes, 0, [1,1,1,1], 0)
        image_boxes = np.append(image_boxes, [[2,2,2,2]], 0)
        if len(image_boxes) < self.sentence_size:
            for i in range(self.sentence_size - len(image_boxes)):
                image_boxes = np.append(image_boxes, [[0,0,0,0]], 0)

        assert len(image_boxes) == len(image_classes)

        input_token, segment_label, token_type = self.process_word(complete_sentence)

        return torch.tensor(input_token), torch.tensor(segment_label), \
            torch.tensor(token_type), torch.tensor(image_classes), \
            torch.tensor(image_classes), torch.tensor(image_classes), \
            torch.tensor(image_boxes).float(), torch.tensor(complete_object_ids)

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


if __name__ == '__main__':
    instances_json_path = './data/vg_msdn/train.json'
    category_json_path = './data/vg_msdn/categories.json'
    dict_save_path = './data/vg_msdn/object_pred_idx_to_name.pkl'
    VGmsdn = VGmsdnRelDataset(instances_json_path, category_json_path, dict_save_path,
               sentence_size=128, is_mask=True)
    print(VGmsdn.vocab['object_pred_idx_to_name'])
    fn = './data/vg_msdn/object_pred_idx_to_name.pkl'
    with open(fn, 'wb+') as file:
        pickle.dump(VGmsdn.vocab['object_pred_idx_to_name'], file)
