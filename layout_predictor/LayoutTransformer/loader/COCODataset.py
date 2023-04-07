import numpy as np
from pycocotools.coco import COCO
import numpy as np
import pickle
import json, os, random, math
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from fairseq.models.roberta import alignment_utils
import random
random.seed(42)

class COCORelDataset(Dataset):
    def __init__(self, instances_json, stuff_json=None,
               stuff_only=True, normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               sentence_size=128, is_mask=True, is_std=False,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None,
               reverse=False, obj_id_v2=False):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Modify:
        Input: Text
        Output: bbox

        Inputs:
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
          
        - 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        - [PAD], [CLS], [SEP], [MASK]
        """
        
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')
            
        self.is_std = is_std
        self.is_mask = is_mask
        self.reverse = reverse
        self.max_samples = max_samples
        self.sentence_size = sentence_size
        self.include_relationships = include_relationships
        self.obj_id_v2 = obj_id_v2

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
          'object_name_to_idx': {},
          'pred_name_to_idx': {},
          'object_pred_name_to_idx': {},
          'object_pred_idx_to_name': {},
        }
        # setting predictes
        self.snetence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
        
        self.vocab['pred_idx_to_name'] = [
          '__in_image__',
          'left of',
          'right of',
          'above',
          'below',
          'inside',
          'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx
            
        object_idx_to_name = {}
        all_instance_categories = []
        for idx, token in enumerate(self.snetence_token):
            self.vocab['object_name_to_idx'][token] = idx
            
        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = len(self.snetence_token)
        
        for category_data in instances_data['categories']:
            category_id = category_data['id'] + len(self.snetence_token)
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
            
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_id = category_data['id'] + len(self.snetence_token)
                category_name = category_data['name']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id
        
        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        
        self.vocab['object_pred_name_to_idx']
        all_vocabs = []
        for idx, name in enumerate(self.vocab['object_name_to_idx'].keys()):
            all_vocabs.append(name)
        for idx, name in enumerate(self.vocab['pred_name_to_idx'].keys()):
            all_vocabs.append(name)
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_idx_to_name'][i] = all_vocabs[i]
        
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = \
                object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = \
                    object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        
        # boxes = [xc, yc, w, h] normalized
        all_boxes = []
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            W, H = self.image_id_to_size[image_id]
            x0, y0, w, h = object_data['bbox']
            xc, yc, w, h = (x0+w/2.)/W, (y0+h/2.)/H, w/W, h/H
            all_boxes.append([xc, yc, w, h])
        if stuff_data:
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                W, H = self.image_id_to_size[image_id]
                x0, y0, w, h = object_data['bbox']
                xc, yc, w, h = (x0+w/2.)/W, (y0+h/2.)/H, w/W, h/H
                all_boxes.append([xc, yc, w, h])

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
        
        sta_dict_path = os.path.dirname(instances_json)
        with open(os.path.join(sta_dict_path,'sta_dict.json'), 'w') as fp:
            json.dump(sta_dict, fp)
        with open("data/coco/parsed_caption_label_dict.pkl", "rb") as f:
            anno_text = pickle.load(f)
        self.text = anno_text
        # remove some ids from self.image_ids; those ids do not have parsed words
        all_possible_ids = anno_text.keys()
        real_ids = []
        for possible_id in self.image_ids:
            if possible_id in all_possible_ids:
                real_ids.append(possible_id)
        self.image_ids = real_ids
        # assign bbox for self.text
        self.image_ids_with_bbox = []
        for data_index, each in enumerate(self.image_ids):
            image_id = each
            image = self.text[image_id][0]
            cands = image[2:]
            found=False
            try:
                bboxs = self.image_id_to_objects[image_id]
                W, H = self.image_id_to_size[image_id]
                for cand_index, each_cand in enumerate(cands):
                    candidate_name = each_cand[1]
                    for bbox in bboxs:
                        if self.vocab['object_idx_to_name'][bbox['category_id'] + 4] == candidate_name:
                            x,y,w,h = bbox['bbox']
                            each_cand.append([(x+w/2.)/W, (y+h/2.)/H, w/W, h/H])
                            found=True
                            break
                if found:
                    self.image_ids_with_bbox.append(image_id)
            except:
                continue
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        self.tokenizer = alignment_utils.spacy_tokenizer()

        import pickle as pkl
        with open("data/gpt-3.pkl", "rb") as f:
            self.gpt3 = pkl.load(f)        
            
    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        return len(self.gpt3) * 2

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
        if index < len(self.gpt3):
            image = self.gpt3[index]
            caption = image[0]
            bpe_toks = self.roberta.encode(caption)
            sample_caption_tokens = image[1]
            alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, sample_caption_tokens)
            relation = image[3]
            object_index = image[2]
            real_object_index = [] # "real" means the index have been fixed to roberta index
            for each_index in object_index:
                real_object_index.append(alignment[each_index][0])
            real_relation = []
            for each_relation in relation:
                assert len(each_relation) == 3
                real_each_relation = []
                real_each_relation.append(alignment[each_relation[0]][0])
                real_each_relation.append(alignment[each_relation[1]][0])
                real_each_relation.append(each_relation[2])
                real_relation.append(real_each_relation)
            

            if self.sentence_size >= bpe_toks.shape[0]:
                padding = torch.ones(self.sentence_size - bpe_toks.shape[0]).int()
                bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
            else:
                bpe_toks = bpe_toks[:128]
            return [bpe_toks.unsqueeze(0), real_object_index, None, caption, real_relation]
        else:
            # random sample a self.text
            all_keys = list(self.image_ids_with_bbox)
            # all_keys = list(self.text.keys())
            sample_key = random.sample(range(len(all_keys)), k=1)[0]
            curr_index = all_keys[sample_key]
            image = self.text[curr_index][0] # always the first description
            caption = image[0]
            bpe_toks = self.roberta.encode(caption)
            sample_caption_tokens = image[1]
            alignment = alignment_utils.align_bpe_to_words(self.roberta, bpe_toks, sample_caption_tokens)
            real_object_index = []
            returnbboxs = []
            cands = image[2:]
            for each in cands:
                if len(each) == 4:
                    try:
                        real_object_index.append(alignment[each[0]][0])
                        returnbboxs.append(each[3])
                    except:
                        continue
            if self.sentence_size >= bpe_toks.shape[0]:
                padding = torch.ones(self.sentence_size - bpe_toks.shape[0]).int()
                bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
            else:
                bpe_toks = bpe_toks[:128]
            return [bpe_toks.unsqueeze(0), real_object_index, True, caption, returnbboxs]

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
    
    
class COCOLayoutDataset(Dataset):
    def __init__(self, instances_json, stuff_json=None,
               stuff_only=True, normalize_images=True, max_samples=None,
               include_relationships=True, min_object_size=0.02,
               sentence_size=128, is_mask=True,
               min_objects_per_image=3, max_objects_per_image=8,
               include_other=False, instance_whitelist=None, stuff_whitelist=None):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - instances_json: Path to a JSON file giving COCO annotations
        - stuff_json: (optional) Path to a JSON file giving COCO-Stuff annotations
        - stuff_only: (optional, default True) If True then only iterate over
          images which appear in stuff_json; if False then iterate over all images
          in instances_json.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        - stuff_whitelist: None means use all stuff categories. Otherwise a list
          giving a whitelist of stuff category names to use.
          
        - 0 for PAD, 1 for BOS, 2 for EOS, 3 for MASK
        - [PAD], [CLS], [SEP], [MASK]
        """
        
        super(Dataset, self).__init__()

        if stuff_only and stuff_json is None:
            print('WARNING: Got stuff_only=True but stuff_json=None.')
            print('Falling back to stuff_only=False.')
            
        self.is_mask = is_mask
        self.max_samples = max_samples
        self.sentence_size = sentence_size
        self.include_relationships = include_relationships

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)

        stuff_data = None
        if stuff_json is not None and stuff_json != '':
            with open(stuff_json, 'r') as f:
                stuff_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

        self.vocab = {
          'object_name_to_idx': {},
          'pred_name_to_idx': {},
          'object_pred_name_to_idx': {},
          'object_pred_idx_to_name': {},
        }
        # setting predictes
        self.snetence_token = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
        
        self.vocab['pred_idx_to_name'] = [
          '__in_image__',
          'left of',
          'right of',
          'above',
          'below',
          'inside',
          'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx
            

        object_idx_to_name = {}
        all_instance_categories = []
        for idx, token in enumerate(self.snetence_token):
            self.vocab['object_name_to_idx'][token] = idx
            
        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = len(self.snetence_token)
        
        for category_data in instances_data['categories']:
            category_id = category_data['id'] + len(self.snetence_token)
            category_name = category_data['name']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
            
        all_stuff_categories = []
        if stuff_data:
            for category_data in stuff_data['categories']:
                category_id = category_data['id'] + len(self.snetence_token)
                category_name = category_data['name']
                all_stuff_categories.append(category_name)
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id
        
        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        
        self.vocab['object_pred_name_to_idx']
        all_vocabs = []
        for idx, name in enumerate(self.vocab['object_name_to_idx'].keys()):
            all_vocabs.append(name)
        for idx, name in enumerate(self.vocab['pred_name_to_idx'].keys()):
            all_vocabs.append(name)
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_name_to_idx'][all_vocabs[i]] = i
        for i in range(len(all_vocabs)):
            self.vocab['object_pred_idx_to_name'][i] = all_vocabs[i]
        
        if instance_whitelist is None:
            instance_whitelist = all_instance_categories
        if stuff_whitelist is None:
            stuff_whitelist = all_stuff_categories
        category_whitelist = set(instance_whitelist) | set(stuff_whitelist)

        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            _, _, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = \
                object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
            category_ok = object_name in category_whitelist
            other_ok = object_name != 'other' or include_other
            if box_ok and category_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)

        # Add object data from stuff
        if stuff_data:
            image_ids_with_stuff = set()
            for object_data in stuff_data['annotations']:
                image_id = object_data['image_id']
                image_ids_with_stuff.add(image_id)
                _, _, w, h = object_data['bbox']
                W, H = self.image_id_to_size[image_id]
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = \
                    object_idx_to_name[object_data['category_id']+len(self.snetence_token)]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)

            if stuff_only:
                new_image_ids = []
                for image_id in self.image_ids:
                    if image_id in image_ids_with_stuff:
                        new_image_ids.append(image_id)
                self.image_ids = new_image_ids

                all_image_ids = set(self.image_id_to_filename.keys())
                image_ids_to_remove = all_image_ids - image_ids_with_stuff
                for image_id in image_ids_to_remove:
                    self.image_id_to_filename.pop(image_id, None)
                    self.image_id_to_size.pop(image_id, None)
                    self.image_id_to_objects.pop(image_id, None)

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

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

        W, H = self.image_id_to_size[image_id]
        objs, boxes, obj_ids = [], [], []
        for object_data in self.image_id_to_objects[image_id]:
            obj_name = self.vocab['object_idx_to_name'][object_data['category_id']+len(self.snetence_token)]
            objs.append(self.vocab['object_pred_name_to_idx'][obj_name])
            obj_ids.append(object_data['id'])
            x, y, w, h = object_data['bbox']
            x0 = x / W
            y0 = y / H
            x1 = (x+w) / W
            y1 = (y+h) / H
            boxes.append([x0, y0, x1, y1])
        
        # object ids transform
        obj_ids_no_repeat = list(set(obj_ids))
        for i in range(len(obj_ids)):
            obj_ids[i] = obj_ids_no_repeat.index(obj_ids[i]) + 1
        
        # Add dummy __image__ object
        objs.append(self.vocab['object_pred_name_to_idx']['__image__'])
        obj_ids.append(0)
        boxes.append([0, 0, 1, 1])
        blank_box = [2., 2., 2., 2.]

#         objs = torch.LongTensor(objs)
#         obj_ids = torch.LongTensor(obj_ids)
#         boxes = torch.stack(boxes, dim=0)

        # Compute centers of all objects
        obj_centers = []
        for i, obj_idx in enumerate(objs):
            x0, y0, x1, y1 = boxes[i]
            mean_x = (x0 + x1) / 2.
            mean_y = (y0 + y1) / 2.
            obj_centers.append([mean_x, mean_y])
        obj_centers = torch.FloatTensor(obj_centers)
        
        assert len(objs) == len(obj_ids) == len(boxes) == len(obj_centers)
        # Add triples
        triples = []
        triples_ids = []
        triples_boxes = []
        
        num_objs = len(objs)
        __image__ = self.vocab['object_pred_name_to_idx']['__image__']
        real_objs = []
        if num_objs > 1:
            real_objs = (torch.LongTensor(objs) != __image__).nonzero().squeeze(1)
        for cur in real_objs:
            choices = [obj for obj in real_objs if obj != cur]
            if len(choices) == 0 or not self.include_relationships:
                break
            other = random.choice(choices)
            if random.random() > 0.5:
                s, o = cur, other
            else:
                s, o = other, cur

            # Check for inside / surrounding
            sx0, sy0, sx1, sy1 = boxes[s]
            ox0, oy0, ox1, oy1 = boxes[o]
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
            triples.append([objs[s], p, objs[o]])
            triples_ids.append([obj_ids[s], 0, obj_ids[o]])
            
            triples_boxes.append([[sx0, sy0, sx1-sx0, sy1-sy0], blank_box, 
                                  [ox0, oy0, ox1-ox0, oy1-oy0]])
            
        # Add __in_image__ triples
        O = len(objs)
        in_image = self.vocab['object_pred_name_to_idx']['__in_image__']
        for i in range(O - 1):
            triples.append([objs[i], in_image, objs[O - 1]])
            triples_ids.append([obj_ids[i], 0, obj_ids[O - 1]])
            sx0, sy0, sx1, sy1 = boxes[i]
            ox0, oy0, ox1, oy1 = boxes[O - 1]
            triples_boxes.append([[sx0, sy0, sx1-sx0, sy1-sy0], blank_box, 
                                  [ox0, oy0, ox1-ox0, oy1-oy0]])
        
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
    ins_train_path = '../data/coco/instances_train2017.json'
    sta_train_path = '../data/coco/stuff_train2017.json'
    COCO = COCORelDataset(ins_train_path, sta_train_path)
    print(COCO.vocab)
    print(len(COCO))
