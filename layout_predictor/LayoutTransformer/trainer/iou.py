import torch
import numpy as np
import json
import os

class IOU_calculator():
    def __init__(self, reduction = 'sum', cfg=None):
        self.reduction = reduction
        if cfg is not None:
            self.cfg = cfg
            self.sta_path = os.path.join(cfg['DATASETS']['DATA_DIR_PATH'], 'sta_dict.json')
            with open(self.sta_path, 'r') as f:
                self.sta_dict = json.load(f)
    
    def val_iou(self, pred_boxes, target_boxes, is_std=False):
        pred_boxes = pred_boxes.reshape(-1,4).detach()
        target_boxes = target_boxes.reshape(-1,4).detach()
        pred_boxes = pred_boxes[1::2]
        target_boxes = target_boxes[1::2]
        non_ignore_mask = target_boxes[:, 0] != 2.
        
        if is_std:
            pred_boxes = self.de_sta_normalized(pred_boxes[non_ignore_mask])
            target_boxes = self.de_sta_normalized(target_boxes[non_ignore_mask])
            pred_boxes_xyxy = self.xcycwh2xyxy(pred_boxes,image_wh=[800,600])
            target_boxes_xyxy = self.xcycwh2xyxy(target_boxes,image_wh=[800,600])
        else:
            pred_boxes_xyxy = self.xcycwh2xyxy(pred_boxes[non_ignore_mask],image_wh=[800,600])
            target_boxes_xyxy = self.xcycwh2xyxy(target_boxes[non_ignore_mask],image_wh=[800,600])
        
        total_iou = self.get_iou(pred_boxes_xyxy, target_boxes_xyxy)
        if self.reduction == 'sum':
            return total_iou
        else:
            return total_iou /  len(target_boxes_xyxy)
    
    def de_sta_normalized(self, ins):
        """
        x*std+mean
        """
        ins[:, 0] = ins[:, 0] * self.sta_dict['x_std'] + self.sta_dict['x_mean']
        ins[:, 1] = ins[:, 1] * self.sta_dict['y_std'] + self.sta_dict['y_mean']
        ins[:, 2] = ins[:, 2] * self.sta_dict['w_std'] + self.sta_dict['w_mean']
        ins[:, 3] = ins[:, 3] * self.sta_dict['h_std'] + self.sta_dict['h_mean']
        return ins
    
    def xcycwh2xyxy(self, boxes, image_wh):
        boxes[:,0] *= image_wh[0]
        boxes[:,1] *= image_wh[1]
        boxes[:,2] *= image_wh[0]
        boxes[:,3] *= image_wh[1]
        center = boxes[:,:2].clone()
        boxes[:,:2] = center - boxes[:,2:]/2.
        boxes[:,2:] = center + boxes[:,2:]/2.
#         boxes[:,0] *= image_wh[0]
#         boxes[:,1] *= image_wh[1]
#         boxes[:,2] *= image_wh[0]
#         boxes[:,3] *= image_wh[0]
        return boxes

    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : B * 4
            Keys: ['x1', 'x2', 'y1', 'y2']
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        # determine the coordinates of the intersection rectangle
        x_left = torch.max(bb1[:,0], bb2[:,0])
        y_top = torch.max(bb1[:,1], bb2[:,1])
        x_right = torch.min(bb1[:,2], bb2[:,2])
        y_bottom = torch.min(bb1[:,3], bb2[:,3])

        legal_index = (x_right >= x_left) & (y_bottom >= y_top)
        x_left = x_left[legal_index]
        y_top = y_top[legal_index]
        x_right = x_right[legal_index]
        y_bottom = y_bottom[legal_index]
        bb1 = bb1[legal_index, :]
        bb2 = bb2[legal_index, :]
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[:,2] - bb1[:,0]) * (bb1[:,3] - bb1[:,1])
        bb2_area = (bb2[:,2] - bb2[:,0]) * (bb2[:,3] - bb2[:,1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / (bb1_area + bb2_area - intersection_area)
        legal_num = iou.size(0)
        iou = torch.sum(iou, dim=0).item()
        if legal_num == 0 or not iou/legal_num >= 0.0 or not iou/legal_num <= 1.0:
            return 0

        return iou