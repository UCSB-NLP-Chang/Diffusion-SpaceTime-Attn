import numpy as np
import skimage as io
import matplotlib as mpl
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import pylab
import urllib
from io import BytesIO
import requests as req
from PIL import Image
import os
import json
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle

def pad(arr, pad_idx=0, pad_length=64):
    '''
    Args:
        pad_idx: 0
        bos_idx: 1
        eos_idx: 2
    '''
    if len(arr) < pad_length:
        arr = np.pad(arr, ((0, pad_length - len(arr))), 'constant', constant_values = pad_idx)
    else:
        if len(arr) != pad_length:
            arr = arr[:pad_length-1]
            arr = np.append(arr, [2])
    return arr




annFile = '../data/annotations/instances_val2017.json'
caption_annFile = '../data/annotations/captions_val2017.json'

coco = COCO(annFile)
captioncoco = COCO(caption_annFile)

image_folder = '../data/val2017/'
# save_folder = 'label_val2017/'
val_anns = []
shapeset = {}
length = set()
shape = []

for fn in os.listdir(image_folder):
    img_id = int(fn.split('.')[0])
    imgs = coco.loadImgs(img_id)
    annIds = coco.getAnnIds(imgIds=img_id)
    img_h = imgs[0]['height']
    img_w = imgs[0]['width']
    anns = coco.loadAnns(annIds)
    single_ann = {}
    cats_name = []
    cats_id = [0]
    center_pos = [0]
    for i in range(len(anns)):
        x, y, w, h = anns[i]['bbox']
        shape_x = round(w/img_w, 2)
        shape_y = round(h/img_h, 2)
        shape.append([shape_x, shape_y])
       
shape = np.array(shape)
pred = MiniBatchKMeans(n_clusters=64, random_state=0).fit(shape)
pred = pred = [int(i) + 1 for i in pred.labels_]
predset= set()
for p in pred:
    predset.add(p)
print(predset)
pred_idx = 0
for fn in os.listdir(image_folder):
    img_id = int(fn.split('.')[0])
    imgs = coco.loadImgs(img_id)
    annIds = coco.getAnnIds(imgIds=img_id)
    img_h = imgs[0]['height']
    img_w = imgs[0]['width']
    anns = coco.loadAnns(annIds)
    annIds_captions = captioncoco.getAnnIds(imgIds=img_id)
    anns_captions = captioncoco.loadAnns(annIds_captions)
    single_ann = {}
    cats_name = []
    cats_id = [1]
    center_pos = [1]
    shape_centroid = [1]
    for i in range(len(anns)):
        x, y, w, h = anns[i]['bbox']
        cat_id = anns[i]['category_id']
        cats = coco.loadCats(cat_id)
        cats_name.append(cats[0]['name'])
        cats_id.append(cats[0]['id'] + 2)
        center_x = x + 0.5*w
        center_y = y + 0.5*h
        center_x_idx = int(center_x / (img_w / 8)) + 1
        center_y_idx = int(center_y / (img_h / 8))
        center_idx = center_y_idx * 8 + center_x_idx
        center_pos.append(center_idx + 2)
    for j in range(1, len(cats_id)):
        shape_centroid.append(pred[pred_idx] + 2)
        pred_idx += 1

    assert len(cats_id) == len(center_pos)
    assert len(cats_id) == len(shape_centroid)

    center_pos = np.array(center_pos)
    sort_idx = np.argsort(center_pos)
    center_pos = center_pos[sort_idx]
    cats_id = np.array(cats_id)
    cats_id = cats_id[sort_idx]
    shape_centroid = np.array(shape_centroid)
    shape_centroid = shape_centroid[sort_idx]

    center_pos = np.append(center_pos, [2])
    cats_id = np.append(cats_id, [2])
    shape_centroid = np.append(shape_centroid, [2])

    center_pos = pad(center_pos)
    cats_id = pad(cats_id)
    shape_centroid = pad(shape_centroid)

    sentence = anns_captions[0]['caption']
    single_ann['img_id'] = img_id
    single_ann['categories'] = cats_id
    single_ann['position'] = center_pos
    single_ann['caption'] = sentence
    single_ann['shape'] = shape_centroid

    val_anns.append(single_ann)
    # print(single_ann)


with open('val_input.pkl', 'wb+') as file:
    pickle.dump(val_anns, file)


