import argparse
import glob
import multiprocessing as mp
import sys
import os
from tqdm import tqdm

sys.path.insert(0, "../")  # noqa
from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

import torch

import pickle as pkl


mapping = {
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

def relation_pass(relation, object1_pos, object2_pos):
    assert relation in ['below', 'left of', 'right of', 'above']
    assert len(object1_pos) == 4
    assert len(object2_pos) == 4
    object1_x = (object1_pos[0] + object1_pos[2]) / 2
    object1_y = (object1_pos[1] + object1_pos[3]) / 2
    object2_x = (object2_pos[0] + object2_pos[2]) / 2
    object2_y = (object2_pos[1] + object2_pos[3]) / 2
    if relation == 'below':
        return (object1_y > object2_y)
    if relation == "left of":
        return (object1_x < object2_x)
    if relation == "right of":
        return (object1_x > object2_x)
    if relation == "above":
        return (object1_y < object2_y)



with torch.no_grad():
    args_config_file = "../projects/dino/configs/dino-swin/dino_swin_large_384_4scale_36ep.py"
    detector_cfg = LazyConfig.load(args_config_file)
    detector_cfg = LazyConfig.apply_overrides(detector_cfg, ['train.init_checkpoint=../dino_swin_large_384_4scale_36ep.pth'])

    detector_model = instantiate(detector_cfg.model)
    detector_model.to(detector_cfg.train.device)
    detector_checkpointer = DetectionCheckpointer(detector_model)
    detector_checkpointer.load(detector_cfg.train.init_checkpoint)

    detector_model.eval()

    detector_demo = VisualizationDemo(
        model=detector_model,
        min_size_test=800,
        max_size_test=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
    )

    detector_confidence_threshold = 0.5

    # detect all images
    test_folder = "../../result_outputs/"
    gt = "../../../../datasets/vsr.pkl"
    relation_keywords = ['below', 'left of', 'right of', 'above']
    gt_relations = []

    with open(gt, "rb") as f:
        temp = pkl.load(f)
        for i in range(500):
            # extract relation: "below" "left of" "right of" "above" "at the right side of" "at the left side of"
            curr_relation_pure = temp[i][3][0][2]
            curr_relation_objects = temp[i][4]
            curr_relations_no_attributes = []
            if curr_relation_pure == "at the right side of":
                curr_relation_pure = "right of"
            if curr_relation_pure == "at the left side of":
                curr_relation_pure = "left of"
            curr_relations_no_attributes.append(curr_relation_pure)
            # check what objects are involved, remove attributes
            assert len(curr_relation_objects) == 2
            curr_relation_objects[0] = curr_relation_objects[0][4:] # remove "the "
            curr_relation_objects[1] = curr_relation_objects[1][4:]
            curr_relations_no_attributes.append(curr_relation_objects[0])
            curr_relations_no_attributes.append(curr_relation_objects[1])
            gt_relations.append(curr_relations_no_attributes)


    files = os.listdir(test_folder)

    seed = 1
    cnt = 0
    corr = 0
    real_files = []
    for each_file in files:
        if "final2_s1_" in each_file:
            real_files.append(each_file)

    for i in tqdm(range(1)):
        file_name = test_folder + real_files[i]
        img = read_image(file_name)
        curr_file_idx = int(real_files[i].split('_')[-1][:-4])
        predictions, _ = detector_demo.run_on_image(img, detector_confidence_threshold)
        bbox = predictions['instances'].pred_boxes.tensor.cpu().numpy() # upper w, upper h, lower w, lower h 
        cate = predictions['instances'].pred_classes.cpu().numpy()
        cate_name_pos = {} # key-value: cate_name-[object_pos_1, (object_pos_2, ... )] (multiple positions when generating multiple objects)

        # information of the current image
        for idx_each, cate_each in enumerate(cate):
            if mapping.__contains__(cate_each + 1):
                curr_name = mapping[cate_each+1]
                if not cate_name_pos.__contains__(curr_name):
                    cate_name_pos[curr_name] = [bbox[idx_each]]
                else:
                    cate_name_pos[curr_name].append(bbox[idx_each])

        # check each relation correctness
        assert (len(gt_relations[curr_file_idx]) % 3 == 0)
        for j in range(len(gt_relations[curr_file_idx]) // 3):
            curr_test_relation_triplet = gt_relations[curr_file_idx][3*j : (3*j+3)]
            curr_test_relation = curr_test_relation_triplet[0]
            curr_test_relation_object1 = curr_test_relation_triplet[1]
            curr_test_relation_object2 = curr_test_relation_triplet[2]
            # If objects are not synthesized, the relations are considered WRONG, and WILL NOT BE CALCULATED
            if not cate_name_pos.__contains__(curr_test_relation_object1):
                continue
            if not cate_name_pos.__contains__(curr_test_relation_object2):
                continue

            # Now that both objects exist, loop every coordinate to see if there is a match
            cnt += 1
            IMMEDIATE_BREAK = False
            for p in range(len(cate_name_pos[curr_test_relation_object1])):
                if IMMEDIATE_BREAK:
                    break
                for q in range(len(cate_name_pos[curr_test_relation_object2])):
                    if IMMEDIATE_BREAK:
                        break
                    temp_pos_object1 = cate_name_pos[curr_test_relation_object1][p]
                    temp_pos_object2 = cate_name_pos[curr_test_relation_object2][q]
                    if relation_pass(curr_test_relation, temp_pos_object1, temp_pos_object2):
                        corr += 1
                        IMMEDIATE_BREAK = True

    print("seed %d"%(seed))
    print("All relation numbers: %d"%(cnt))
    print("Correct relation numbers: %d"%(corr))
    print(corr/cnt)
