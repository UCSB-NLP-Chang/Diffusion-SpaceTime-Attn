import sys
sys.path.append("../")
sys.path.append("./")
from utils import ensure_dir
import argparse
import logging
import json
import cv2
from bounding_box import bounding_box as bb
import os
import numpy as np
import torch
import yaml
from model import build_model
from tqdm import tqdm
import time
from fairseq.models.roberta import alignment_utils
import spacy
from nltk.corpus import wordnet
import inflect
import pickle as pkl

logger = logging.getLogger('inference')

class Inference_COCO():
    def __init__(self, save_dir, vocab_dict):
        self.save_dir = save_dir
        self.device = self._prepare_gpu()
        self.vocab_dict = vocab_dict
            
    def check_GT(self, idx, dataset):
        """
        input_token = COCO[0][0]
        input_obj_id = COCO[0][1]
        input_box_label = COCO[0][2]
        output_label = COCO[0][3]
        segment_label = COCO[0][4]
        token_type = COCO[0][5]
        """
        image_id = dataset.image_ids[idx]
        single_data = dataset[idx]
        log_file_name = os.path.join(self.save_dir,str(image_id)+'.txt')
        image_wh = dataset.image_id_to_size[image_id]
        image_size = [image_wh[1], image_wh[0], 3]
        
        box_mask = np.array(single_data[2]) != 2.
        boxes = np.array(single_data[2])[box_mask].reshape(-1,4)
        boxes = self.xywh2xyxy(boxes, image_wh)
        
        id_mask = np.array(single_data[1]) != 0.
        ids = np.array(single_data[1])[id_mask]
        
        class_mask = (single_data[0][1::2] != 0)
        clss = single_data[0][1::2][class_mask]
        
        __image__mask = clss != 4
        clss = clss[__image__mask]
        clss = self.idx2vocab(clss, 'text')

        boxes = boxes[__image__mask]


        assert len(clss) == len(boxes)

        self.draw_img(image_size = image_size, boxes=boxes, labels=clss,
                      save_dir = self.save_dir,label_ids= torch.ones(len(clss)), name=str(image_id)+'_gt')
        return boxes
        
    def check_from_model(self, dataset_idx, dataset, model, random=False, layout_save=None):
        model.to(self.device)
        model.eval()
#         bb_gt = self.check_GT(idx, dataset)
        image_id = dataset.image_ids[dataset_idx]
        single_data = dataset[dataset_idx]
        image_wh = dataset.image_id_to_size[image_id]
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name = os.path.join(json_save_dir, str(image_id)+'.json')

        input_token = single_data[0].unsqueeze(0).to(self.device)
        input_obj_id = single_data[1].unsqueeze(0).to(self.device)
        segment_label = single_data[5].unsqueeze(0).to(self.device)
        token_type = single_data[6].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        
        vocab_logits, obj_id_logits, token_type_logits, output_box, _, refine_box, _ = \
            model(input_token, input_obj_id, segment_label, token_type, src_mask, inference=True, epoch=0, global_mask=global_mask)
        
        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        id_mask = (input_token == 3) * (input_token > 4) * (input_token < 160)
        input_obj_id[id_mask] = pred_id[id_mask]
        
        # get relation prior
        rel_mask = (pred_vocab >= 178) * (pred_vocab < 184)
        rel_prior = output_box[rel_mask].detach()
        rel_vocab = pred_vocab[rel_mask].detach()
        rel_classes = self.idx2vocab(rel_vocab, 'text')
        rel_prior_xy = rel_prior[:, :2]
#         print(rel_prior_xy)
        self.save_relation_prior(rel_classes, rel_prior_xy, self.save_dir)
        
        # construct mask
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))
        mask_obj_avg = []
        for i in range(1, int(max(input_obj_id_list))+1):
#             idx = len(input_obj_id_list) - 1 - input_obj_id_list[::-1].index(i)
            mask_obj_avg.append((torch.LongTensor(input_obj_id_list) == i).reshape(1, -1)) 
            idx = input_obj_id_list.index(i)
            mask[0, idx] = 1
        mask = mask.bool()
        
        pred_vocab = pred_vocab[mask].detach()
        output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - output_box[0, 2::4, :2]
        
        output_boxes = output_box[mask].detach()
        
        output_class_ids = input_obj_id[mask].detach()
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)

        pred_classes = self.idx2vocab(pred_vocab, 'text')
        
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')
#         print(self.get_iou(output_boxes.squeeze(0), bb_gt))
        
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        if refine_box is not None:
            refine_boxes = refine_box[mask].detach()
            refine_boxes = self.xcycwh2xyxy(refine_boxes, image_wh)
            self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                          labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                          save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name, 
                        name=image_id, idx=dataset_idx)
        self.write_json(output_sentence, input_obj_id_list, json_file_name, 
                        name=image_id, idx=dataset_idx)
        if layout_save is not None: 
            self.save_layout(boxes=refine_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                             name=image_id, image_wh=image_wh)

    def check_from_sg(self, input_dict, model, layout_save=None):
        model.to(self.device)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("PARAMETERS:", pytorch_total_params)
        model.eval()
        print("self input")
        image_id = input_dict['image_id']
        dataset_idx = input_dict['dataset_idx']
        image_wh = [640, 640]
        
        if layout_save is not None: 
            ensure_dir(layout_save)
            image_wh = [640, 640]
        image_size = [image_wh[1], image_wh[0], 3]
        log_file_name = os.path.join(self.save_dir, str(image_id)+'.txt')
        json_save_dir = os.path.join(self.save_dir, 'sg2im_json')
        ensure_dir(json_save_dir)
        json_file_name = os.path.join(json_save_dir, str(image_id)+'.json')

        input_token = input_dict['tensor_list'][0].unsqueeze(0).to(self.device)
        input_obj_id = input_dict['tensor_list'][1].unsqueeze(0).to(self.device)
        segment_label = input_dict['tensor_list'][2].unsqueeze(0).to(self.device)
        token_type = input_dict['tensor_list'][3].unsqueeze(0).to(self.device)
        src_mask = (input_token != 0).unsqueeze(1).to(self.device)
        global_mask = input_token >= 2
        
        ####
        input_token = input_token.repeat(64, 1)
        input_obj_id = input_obj_id.repeat(64, 1)
        segment_label = segment_label.repeat(64, 1)
        token_type = token_type.repeat(64, 1)
        src_mask = src_mask.repeat(64, 1, 1)
        global_mask = global_mask.repeat(64, 1)

        ##
        with torch.no_grad():
            start = time.time()
            vocab_logits, obj_id_logits, token_type_logits, output_box, _, refine_box, _ = \
                model(input_token, input_obj_id, segment_label, token_type, src_mask, inference=True, epoch=0, global_mask=global_mask)
            end = time.time()

        print("Elapsed time,", end-start)
        print("Batch Size", vocab_logits.size(0))
        exit()
        pred_vocab = vocab_logits.argmax(2)
        pred_id = obj_id_logits.argmax(2)
        id_mask = (input_token == 3) * (pred_vocab > 4) * (pred_vocab < 177)
        input_obj_id[id_mask] = pred_id[id_mask]
        # construct mask
        
        input_obj_id_list = list(input_obj_id[0].cpu().numpy())
        mask = torch.zeros(1,len(input_obj_id_list))
        mask_obj_avg = []
        for i in range(1, int(max(input_obj_id_list))+1):
#             idx = len(input_obj_id_list) - 1 - input_obj_id_list[::-1].index(i)
            mask_obj_avg.append((torch.LongTensor(input_obj_id_list) == i).reshape(1, -1)) 
            idx = input_obj_id_list.index(i)
            mask[0, idx] = 1
            
#         mask[0, input_obj_id[0] > 0] = 1
        mask = mask.bool()
#         pred_mask = (pred_vocab >= 4) * (pred_vocab < 176)
        pred_vocab = pred_vocab[mask].detach()
#         pred_id = pred_id[mask].detach()
        # use relation
#         print(output_box[0, 2::4, :2])
        output_box[0, 3::4, :2] = output_box[0, 1::4, :2] - output_box[0, 2::4, :2]
    
        output_boxes = output_box[mask].detach()
        refine_boxes = refine_box[mask].detach()
        output_class_ids = input_obj_id[mask].detach()
        
        output_boxes = self.xcycwh2xyxy(output_boxes, image_wh)
        refine_boxes = self.xcycwh2xyxy(refine_boxes, image_wh)
        pred_classes = self.idx2vocab(pred_vocab, 'text')
#         print(pred_classes)
        output_sentence = self.idx2vocab(vocab_logits.argmax(2).squeeze(0), 'text')
#         print(output_sentence)
        input_sentence = self.idx2vocab(input_token.squeeze(0), 'text')
#         print(self.get_iou(output_boxes.squeeze(0), bb_gt))
#         print(input_sentence)
#         print(output_sentence)
        self.draw_img(image_size = image_size, boxes=output_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx)
        self.draw_img(image_size = image_size, boxes=refine_boxes.squeeze(0),
                      labels=pred_classes, label_ids=output_class_ids.squeeze(0),
                      save_dir = self.save_dir, name=image_id, idx=dataset_idx, mode='r')
        self.write_log(output_sentence, input_obj_id_list, log_file_name, 
                        name=image_id, idx=dataset_idx)
        self.write_json(output_sentence, input_obj_id_list, json_file_name, 
                        name=image_id, idx=dataset_idx)
        if layout_save is not None: 
            self.save_layout(boxes=refine_boxes.squeeze(0), objs=pred_classes, 
                         save_path=layout_save, label_ids=output_class_ids.squeeze(0),
                             name=image_id, image_wh=image_wh)
            
    def save_relation_prior(self, rel_classes, rel_prior_xy, save_dir):
        rel_prior_xy = rel_prior_xy.tolist()
        try:
            with open(os.path.join(save_dir, 'rel_prior.json'), 'r') as fp:
                rel_dict = json.load(fp)
        except:
            rel_dict = dict()
        for i, rel in enumerate(rel_classes):
            if rel in rel_dict.keys():
                rel_dict[rel].append(rel_prior_xy[i])
            else:
                rel_dict[rel] = [rel_prior_xy[i]]
        with open(os.path.join(save_dir, 'rel_prior.json'), 'w') as fp:
            json.dump(rel_dict, fp)
        
        
#     def predict_val(self, )
    def draw_img(self, image_size, boxes, labels, label_ids, save_dir, name, idx, mode='c'):
        # setting color
        color = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver',
                'navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 
                 'orange', 'red', 'maroon', 'fuchsia', 'purple', 'black', 'gray' ,'silver']
        image = np.full(image_size, 200.)
        
        boxes[boxes < 0] = 0
        boxes[boxes > image_size[0]] = image_size[0]
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(0)
        for i in range(len(boxes)):
            bb.add(image, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                   str(labels[i]+'[{}]'.format(label_ids[i])), 
                   color=color[ord(labels[i][0])-ord('a')])
        self.show_and_save(image,
                   os.path.join(save_dir,str(name) + '_{}_{}'.format(idx, mode) +'.png'))

    def write_log(self, sentence, class_ids, log_file_name, name=None, idx=None):
        f = open(log_file_name, 'w')
        for i in range(1, len(sentence), 4):
            if sentence[i+1] == '__in_image__':
                break
            single_pair = ''
            single_pair += sentence[i] + '[{}]'.format(class_ids[i]) + ' '
            single_pair += sentence[i+1] + ' '
            single_pair += sentence[i+2] + '[{}]'.format(class_ids[i+2]) + ' '
            single_pair += '\n'
            f.write(single_pair)
    
    def write_json(self, sentence, class_ids, log_file_name, name=None, idx=None):
        out_dict = dict()
        out_dict['image_id'] = name
        out_dict['dataset_idx'] = idx
        out_dict['objects'] = ["None" for i in range(max(class_ids))]
        out_dict['relationships'] = []
        for i in range(1, len(sentence), 4):
            if sentence[i+1] == '__in_image__':
                break
            out_dict['objects'][int(class_ids[i]-1)] = sentence[i]
            out_dict['objects'][int(class_ids[i+2]-1)] = sentence[i+2]
            single_rel = [int(class_ids[i]-1), sentence[i+1], int(class_ids[i+2]-1)]
            out_dict['relationships'].append(single_rel)
        with open(log_file_name, 'w') as outfile:
            json.dump(out_dict, outfile)
        
    def save_layout(self, boxes, objs, save_path, label_ids, name, image_wh):
        output_dict = dict()
        output_dict['image_id'] = name
        output_dict['boxes'] = (boxes/image_wh[0]).tolist()
        output_dict['classes'] = objs
        output_dict['class_ids'] = label_ids.tolist()
        output_file_name = os.path.join(save_path,str(name)+'.json')
        with open(output_file_name, 'w') as fp:
            json.dump(output_dict, fp)
        return 0
        
    def xywh2xyxy(self, boxes, image_wh):
        center = boxes[:,:2].copy()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
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
        print("Dataset: ",len(dataset))
        layout_path = cfg['TEST']['LAYOUT_MODE'] if cfg['TEST']['LAYOUT_MODE'] != '' else None

        if cfg['TEST']['MODE'] == 'gt':
            if cfg['TEST']['RANDOM']: logger.warning('Test gt mode do not support random.')
            self.check_GT(cfg['TEST']['SAMPLE_IDX'], dataset)
        elif cfg['TEST']['MODE'] == 'model':
            if cfg['TEST']['RANDOM']: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
                    for idx in tqdm(range(50)):
                        self.check_from_model(idx, dataset = dataset, model=model, 
                                              random=True, layout_save=layout_path)
                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset, model, 
                                       random=True, layout_save=layout_path)
            else: 
                if cfg['TEST']['SAMPLE_IDX'] == -1:
                    for idx in tqdm(range(50)):
                        self.check_from_model(idx, dataset = dataset, model=model,
                                             layout_save=layout_path)
                else:
                    self.check_from_model(cfg['TEST']['SAMPLE_IDX'], dataset=dataset,
                                         model=model, layout_save=layout_path)
        else:
            logger.error('We only support gt and model test mode.')
            
    def get_iou(self, bb, bb_gt):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """
        print(bb)
        print(bb_gt)

        # determine the coordinates of the intersection rectangle
        iou_list = []
        for i in range(len(bb)):
            x_left = max(bb[i][0], bb_gt[i][0])
            y_top = max(bb[i][1], bb_gt[i][1])
            x_right = min(bb[i][2], bb_gt[i][2])
            y_bottom = min(bb[i][3], bb_gt[i][3])

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # The intersection of two axis-aligned bounding boxes is always an
            # axis-aligned bounding box
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb[i][2] - bb[i][0]) * (bb[i][3] - bb[i][1])
            bb2_area = (bb_gt[i][2] - bb_gt[i][0]) * (bb_gt[i][3] - bb_gt[i][1])

            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
            assert iou >= 0.0
            assert iou <= 1.0
            iou_list.append(iou)
        return iou_list


# ============ Above codes are from original repo =============
# ============ Below codes are specifically for ours usage ====

nlp = spacy.load("en_core_web_sm")
engine = inflect.engine()
word_set = {}
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
curr_path = os.path.abspath(os.path.join(os.path.join(os.path.abspath(__file__), os.pardir), os.pardir))
with open(curr_path + '/data/coco/category_dict.pkl', 'rb') as f:
    all_categories = pkl.load(f)
for each in all_categories:
	word_set[each] = [each.lower()]
for each in word_set:
	synonyms = []
	for syn in wordnet.synsets(each, pos='n'):
		for l in syn.lemmas():
			synonyms.append(l.name().lower())
	synonyms.append(engine.plural(each))
	synonyms = set(synonyms)
	for each_syn in synonyms:
		word_set[each].append(each_syn)

def check_in_mscoco(noun_pharse):
    # If noun_phrase is in ms_coco, return True
    for each_cate in word_set:
        if each_cate in noun_pharse:
            return True
    return False

cfg_path = curr_path + '/configs/coco/coco_seq2seq_v9_ablation_4.yaml'
model_path = curr_path + "/saved/coco_F_seq2seq_v9_ablation_4/checkpoint_90_0.0.pth"
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

# build model
model = build_model(cfg)
checkpoint = torch.load(model_path)

model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()

from nltk.corpus import stopwords
nlp = spacy.load("en_core_web_sm")
stoplist = set(stopwords.words("english")).union(
    nlp.Defaults.stop_words
)


def inference_sentence(sentence, opt=None):
    with torch.no_grad():
        def check_relation(sentence, object_index):
            bpe_toks = roberta.encode(sentence)
            padding = torch.ones(128 - bpe_toks.shape[0]).int()
            bpe_toks = torch.cat((bpe_toks, padding), dim = 0)
            src_mask = (bpe_toks != 1).to(bpe_toks)
            bpe_toks = bpe_toks.unsqueeze(0).to("cuda")
            src_mask = src_mask.unsqueeze(0).to("cuda").unsqueeze(0)
            trg_tmp = bpe_toks[:,:-1].to("cuda")
            trg_mask = (trg_tmp != 1).unsqueeze(1)
            trg_mask[:,0] = 1
            doc = nlp(sentence)
            alignment = alignment_utils.align_bpe_to_words(roberta, roberta.encode(sentence), doc)
            object_tensor = torch.zeros(128).to(torch.bool)
            for each_object_index in object_index:
                object_tensor[alignment[each_object_index]] = True
            object_tensor = object_tensor.unsqueeze(0)
            output1, _, _, _ = model(bpe_toks, src_mask, None, trg_mask=trg_mask, object_pos_tensor=object_tensor)
            return output1, alignment

        if sentence is None:
            sentence = opt.sentence if (opt is not None) else 'The silver bed was situated to the right of the white couch.'
        # sentence = 'The silver bed was situated to the right of the white couch.'
        # sentence = 'The brown fork was placed on top of the table, with the red bottle resting securely below it.'
        # # sentence = 'The silver laptop was perched atop the green keyboard, its screen illuminated with a bright glow.'
        # # sentence = "The apple is placed above the banana."
        # # sentence = "The apple is placed right of the banana."
        # sentence = "The apple is placed right of the banana, and a sandwitch is placed to their left."
        sentence = sentence.replace("\n", "")
        sentence = sentence.rstrip()
        sentence = sentence.lstrip()
        doc = nlp(sentence)
        pos = []
        for chunk in doc.noun_chunks:
            full_noun = chunk.text
            if full_noun.lower() in stoplist:
                continue
            key_noun = chunk.root.text
            word_index = chunk.root.i
            key_noun = key_noun.lower()
            if check_in_mscoco(full_noun):
                pos.append(word_index)
        try:
            output, alignment = check_relation(sentence, pos)
        except:
            return None
        index = 0
        print("Sentence: %s"%(sentence))
        results = {}
        for chunk in doc.noun_chunks:
            if check_in_mscoco(chunk.text):
                result_index = alignment[pos[index]][0]
                x_cord = output[:,result_index][0][0]
                y_cord = output[:,result_index][0][1]
                print("%s position: (%.3f, %.3f)"%(chunk.text, x_cord, y_cord))
                results[chunk.text] = [x_cord.item(), y_cord.item()]
                index += 1
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence",
        type=str,
        nargs="?",
        default="The silver bed was situated to the right of the white couch."
    )
    parser.add_argument(
        "--mscoco",
        type=bool,
        nargs="?",
        default=True
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        default='saved/coco_F_seq2seq_v9_ablation_4/checkpoint_90_0.0.pth'
    )
    opt = parser.parse_args()
    inference_sentence(None, opt)
    # inference_sentence(opt)
