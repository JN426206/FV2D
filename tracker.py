from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.deep_sort.sort import iou_matching

import detected_object
import torch

import numpy as np

from deep_sort_pytorch.deep_sort.sort.preprocessing import non_max_suppression

"""
TODO: Split tracking object by team and object class as independents.
"""
class Tracker:
    
    def __init__(self, model_path, model_config, nms_max_overlap=0.5, max_track_box_iou_threshold = 0.5):
        # self.cfg.merge_from_file(opt.config_deepsort)
        # self.cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.nms_max_overlap = nms_max_overlap
        self.max_track_box_iou_threshold = max_track_box_iou_threshold
        # self.deepsort = DeepSort(model_path="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
        #                     max_dist=0.2, min_confidence=0.3,
        #                     nms_max_overlap=self.nms_max_overlap, max_iou_distance=0.7,
        #                     max_age=70, n_init=3, nn_budget=100,
        #                     use_cuda=True)
        self.deepsort = DeepSort(model_path=model_path,
                                 model_config=model_config,
                    max_dist=0.5, min_confidence=0.3,
                    nms_max_overlap=self.nms_max_overlap, max_iou_distance=0.7,
                    max_age=100, n_init=3, nn_budget=100,
                    use_cuda=True)
    def update(self, detectedObject, frame):
        xywh = detectedObject.get_xywh()
        print(xywh)
        xywhT = torch.Tensor([detectedObject.get_xywh()])
        confsT = torch.Tensor([detectedObject.score])
        # Pass detections to deepsort
        outputs = self.deepsort.update(xywhT, confsT, frame)
        
    def update_all(self, detected_objects, frame):
        bbox_xywh = []
        scores = []
        
        for detectedObject in detected_objects:
            detectedObject.trackId = None
            if detectedObject.isout is not None and not detectedObject.isout:
                bbox_xywh.append(detectedObject.get_xcycwh())
                scores.append([detectedObject.score.item()])
            
        if len(bbox_xywh) == 0:
            return
            
        xywhT = torch.Tensor(bbox_xywh)
        scoresT = torch.Tensor(scores)
        outputs = self.deepsort.update(xywhT, scoresT, frame)
        
        for index, output in enumerate(outputs):
            for detectedObject in detected_objects:
                if self.iou(detectedObject.bbox, output[:4]) >= self.max_track_box_iou_threshold:
                    detectedObject.trackId = output[4]
                    break
            
    def iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
