import numpy as np
import cv2
# import narya.narya.tracker.homography_estimator
import sys
sys.path.insert(0, '.')
from narya.narya.tracker.homography_estimator import HomographyEstimator

image = cv2.imread("homography_dataset/test_img/0.jpg")
# image = cv2.resize(image, (int(512), int(512)))
homo0 = np.load("homography_dataset/test_homo/0_homo.npy")
print(homo0)
homo0pred = np.load("homography_dataset/test_pred_homo/0_homo.npy")
print(homo0pred)

estimator = HomographyEstimator()

estimator.shape_in_x = image.shape[1]
estimator.shape_in_y = image.shape[0]
bbox = [63, 29, 86, 55] # [275.6059272  -74.53022195]
dst = estimator.get_field_coordinates(bbox, homo0pred, "cv")
print(dst)