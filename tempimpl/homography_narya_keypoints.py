import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from narya.narya.utils.vizualization import visualize
# image = cv2.imread('narya/test_image.jpg')
image = cv2.imread('data/jsladjf111128.jpg')
image = cv2.resize(image, (1024,1024))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Image shape: {}".format(image.shape))
# visualize(image=image)

from narya.narya.models.keras_models import KeypointDetectorModel

kp_model = KeypointDetectorModel(
    backbone='efficientnetb3', num_classes=29, input_shape=(320, 320),
)

WEIGHTS_PATH = (
    "https://storage.googleapis.com/narya-bucket-1/models/keypoint_detector.h5"
)
WEIGHTS_NAME = "keypoint_detector.h5"
WEIGHTS_TOTAR = False

checkpoints = tf.keras.utils.get_file(
                WEIGHTS_NAME, WEIGHTS_PATH, WEIGHTS_TOTAR,
            )

kp_model.load_weights(checkpoints)

pr_mask = kp_model(image)
visualize(
        image=denormalize(image.squeeze()),
        pr_mask=pr_mask[..., -1].squeeze(),
    )

template = cv2.imread('narya/world_cup_template.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
template = cv2.resize(template, (1280,720))/255.
# visualize(template=template)

from narya.narya.utils.masks import _points_from_mask 
from narya.narya.utils.homography import get_perspective_transform

src,dst = _points_from_mask(pr_mask[0])
pred_homo = get_perspective_transform(dst,src)
pred_warp = warp_image(cv2.resize(template, (320,320)),pred_homo,out_shape=(320,320))
visualize(
        image=denormalize(image.squeeze()),
        warped_homography=pred_warp,
    )

test = merge_template(image/255.,cv2.resize(pred_warp, (1024,1024)))
visualize(image = test)

ratiox = 320 / image.shape[1]
ratioy = 320 / image.shape[0]

pts = [370, 592, 402, 694]

x_1 = int(pts[0])
y_1 = int(pts[1])
x_2 = int(pts[2])
y_2 = int(pts[3])
x = (x_1 + x_2) / 2.0 * ratiox
y = max(y_1, y_2) * ratioy
pts = np.array([int(x), int(y)])
print(pred_homo)
# pts = np.array([1, 2])
print(pts)
# dst = warp_point(pts, pred_homo, method='torch')
dst = warp_point(pts, np.linalg.inv(pred_homo), method="cv")
print(dst)
print(dst)
pred_warp = warp_image(np_img_to_torch_img(template),to_torch(pred_homo),method='torch')
pred_warp = torch_img_to_np_img(pred_warp[0])
# visualize(image=image,warped_template=cv2.resize(pred_warp, (1024,1024)))

from narya.narya.utils.vizualization import merge_template

test = merge_template(image/255.,cv2.resize(pred_warp, (1024,1024)))
visualize(image = test)