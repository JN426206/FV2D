import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from narya.narya.utils.vizualization import visualize
from narya.narya.tracker.homography_estimator import HomographyEstimator

# image = cv2.imread('narya/test_image.jpg')
image = cv2.imread('data/jsladjf111128.jpg')
# image = cv2.resize(image, (1024,1024))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image shape: {}".format(image.shape))
# visualize(image=image)

homo_estimator = HomographyEstimator(shape_in_x=1280, shape_in_y=720)

pred_homo, method = homo_estimator(image)

# bbox = [370, 592, 402, 694] #'narya/test_image.jpg'
bbox = [556, 360, 556+40, 360+56] #'data/jsladjf111128.jpg'
bbox = [568, 252, 568+30, 252+59] #'data/jsladjf111128.jpg'

dst = homo_estimator.get_field_coordinates(bbox, pred_homo, method)

print(dst)