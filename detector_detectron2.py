
# Setup detectron2 logger
# import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, _PanopticPrediction
from detectron2.data import MetadataCatalog, DatasetCatalog

from detected_object import DetectedObject
import torch
import cv2
import numpy as np
import detector

class DetectronDetector(detector.Detector):
        
    def __init__(self, model_threshold=0.5, model_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml", 
                 model_config_path = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"):
        """_summary_

        Args:
            model_threshold (float, optional): _description_. Defaults to 0.5.
            model_path (str, optional): Set path to the model or Detectron2 Model Zoo and Baselines models file convention name like as default. The model with the Detectron2 convention will be downloaded automatically.  Defaults to "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml".
            model_config_path (str, optional): Same as model by for now for config. If you use model from local sotrage you can still use config from Detectron2 Model Zoo and Baselines. Defaults to "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml".
        """
                
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config_path))
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        if model_path.split(".")[-1] == "yaml":
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
        else:
            self.cfg.MODEL.WEIGHTS = model_path
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)        
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )

    def detect_objects(self, image, classes = [0, 32]) -> [DetectedObject]:
        # Detect objects and if model have panoptic or instance segmenteation detects mask too
        predictions = self.predictor(image)                

        # Copy instances predictions form GPU ram to CPU ram. Instances mean every single detected object
        outputs_instances = predictions["instances"].to("cpu")
        detected_objects = []
        for index, box in enumerate(outputs_instances.pred_boxes):
            pred_class = outputs_instances.pred_classes[index].detach().numpy().astype(int)            
            try:
                pred_mask = outputs_instances.pred_masks[index].detach().numpy().astype(bool)
            except:
                pred_mask = None
            if(pred_class in classes):
                box = box.detach().numpy().astype(int)
                score = outputs_instances.scores[index].detach().numpy().astype(float)
                maski = None
                if pred_mask is not None: 
                    maski = np.zeros(image.shape[:2], dtype=np.uint8)
                    for ix, x in enumerate(pred_mask):
                        for iy, y in enumerate(x):
                            if y:
                                maski[ix][iy] = 255
                    # imagei = cv2.bitwise_and(image, image, mask = maski)
                    # cv2.imshow('mask', imagei)
                    # cv2.waitKey()
                detected_objects.append(DetectedObject(box, pred_class, score, mask=maski))
                
        return detected_objects
