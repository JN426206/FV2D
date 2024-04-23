import detected_object

import numpy as np
import cv2
# import some common narya utilities
import tensorflow as tf

from narya.narya.tracker.homography_estimator import HomographyEstimator

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
class Homography:
    
    def __init__(self, weights_keypoint_path  = ("https://storage.googleapis.com/narya-bucket-1/models/keypoint_detector.h5"),
                 weights_keypoint_totar = False,
                 weights_deephomo_path = ("https://storage.googleapis.com/narya-bucket-1/models/deep_homo_model.h5"),
                 weights_deephomo_totar = False,
                 pretreined = True,
                 BALL_OBJECT_CLASS = 32,
                 threshold = 0.9):        

        self.homo_estimator = HomographyEstimator(pretrained=pretreined, weights_homo=weights_deephomo_path, weights_keypoints=weights_keypoint_path)
        self.method = "cv"
        self.BALL_OBJECT_CLASS = BALL_OBJECT_CLASS
        self.threshold = threshold
        
    def predict_homography(self, image): 
                
        pred_homo, self.method = self.homo_estimator(image, self.threshold)
        
        return pred_homo
    
    def check_object_isout(self, image_shape, deteced_objects, predicted_homography):   
                
        self.homo_estimator.shape_in_x = image_shape[1]
        self.homo_estimator.shape_in_y = image_shape[0]
        
        if predicted_homography is None:
            return
        
        for index, objectDetected in enumerate(deteced_objects):
            bbox = objectDetected.bbox
            # x_1 = int(bbox[0])
            # y_1 = int(bbox[1])
            # x_2 = int(bbox[2])
            # y_2 = int(bbox[3])
            # x = (x_1 + x_2) / 2.0 * ratiox
            # y = max(y_1, y_2) * ratioy
            # pts = np.array([int(x), int(y)])
            
            dst = self.homo_estimator.get_field_coordinates(bbox, predicted_homography, self.method)
            if not np.isnan(dst[0]) or not np.isnan(dst[1]):
                if objectDetected.object_class != None:
                    if dst[0] > self.homo_estimator.shape_out+5 or dst[1] > self.homo_estimator.shape_out+5:
                        objectDetected.isout = True
        
    
    def generate_pitch(self, image_shape, deteced_objects, predicted_homography, pitch_2D_width = 320, pitch_2D_height = 180, template_path = 'TV2D/narya/world_cup_template.png'):
        
        self.homo_estimator.shape_in_x = image_shape[1]
        self.homo_estimator.shape_in_y = image_shape[0]
        
        template = cv2.imread(template_path)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        template = cv2.resize(template, (int(self.homo_estimator.shape_out), int(self.homo_estimator.shape_out)))
                
        if predicted_homography is None:            
            return cv2.resize(template, (pitch_2D_width, pitch_2D_height))
        
        for index, objectDetected in enumerate(deteced_objects):
            bbox = objectDetected.bbox
            
            dst = self.homo_estimator.get_field_coordinates(bbox, predicted_homography, self.method)
            if not np.isnan(dst[0]) or not np.isnan(dst[1]):
                color = (255,0,0)
                if objectDetected.object_class != None:
                        
                    if objectDetected.object_class == self.BALL_OBJECT_CLASS:
                        color = (0,0,255)
                        
                    if objectDetected.color is not None:
                        color = (int(objectDetected.color[0]), int(objectDetected.color[1]), int(objectDetected.color[2]))
                    
                    if dst[0] > self.homo_estimator.shape_out+5 or dst[1] > self.homo_estimator.shape_out+5:
                        objectDetected.isout = True                     
                    else:
                        cv2.circle(template, (int(dst[0]), int(dst[1])), 3, color, -1)
                    # Set detected object normalized to [0;1] x and y cordinates on pitch.
                    objectDetected.pitchxy = (dst[0]/self.homo_estimator.shape_out, dst[1]/self.homo_estimator.shape_out)
                   
        
        template=np.float32(template)
        template=cv2.resize(template, (pitch_2D_width, pitch_2D_height))
        return template
        