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

    detector_confidence_threshold = 0.4

    # detect all images
    test_folder = "../../result_outputs/"
    gt = "../../../../datasets/gpt.txt"
    gt_objects = []

    with open(gt, "r") as f:
        contents = f.read()
        rows = contents.split('\n')
        for i in range(500):
            curr_objects = rows[4*i][9:].split(',')
            curr_objects_no_attributes = []
            for each in curr_objects:
                all_words = each.split()
                if len(all_words) > 1 and all_words[-2] + " " + all_words[-1] in ['hair drier', 'teddy bear', 'cell phone', 'dining table', 'potted plant', 'hot dog', 'wine glass', 'tennis racket', 'baseball glove', 'baseball bat', 'sports ball']:
                    curr_objects_no_attributes.append(all_words[-2] + " " + all_words[-1]) 
                else:
                    curr_objects_no_attributes.append(each.split()[-1])
            gt_objects.append(curr_objects_no_attributes)


    files = os.listdir(test_folder)


    cnt = 0
    corr = 0
    real_files = []
    for each_file in files:
        if "final2_s1_" in each_file:
            real_files.append(each_file)

    for i in tqdm(range(len(real_files))):
        file_name = test_folder + real_files[i]
        curr_txt_index = int(real_files[i].split('_')[-1][:-4])
        img = read_image(file_name)
        predictions, _ = detector_demo.run_on_image(img, detector_confidence_threshold)
        bbox = predictions['instances'].pred_boxes.tensor.cpu().numpy()
        cate = predictions['instances'].pred_classes.cpu().numpy()
        cate_name = []
        for cate_each in cate:
            if mapping.__contains__(cate_each + 1):
                cate_name.append(mapping[cate_each+1])

        for j in range(len(gt_objects[curr_txt_index])):
            cnt += 1
            if gt_objects[curr_txt_index][j] in cate_name:
                corr += 1

    print("All object numbers: %d"%(cnt))
    print("Generated object numbers: %d"%(corr))
    print(corr/cnt)
