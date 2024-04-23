import numpy as np
import cv2 as cv
import json
# import matplotlib.pyplot as plt
# from caffe.proto import caffe_pb2

# lmdb_env = lmdb.open('ncexclude/bbox_lmdb')
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()

# for key, value in lmdb_cursor:
#     print(key, value)
#     break



# image = cv.imread("D:\\segithubminarium_temp\\1chHQ0000.png", cv.IMREAD_COLOR)
# i=0
vidcap = cv.VideoCapture('data/1_HQ.mkv')
success,clear_image = vidcap.read()
count = 0
while success and count != 7936:    
  success,clear_image = vidcap.read()
  count += 1
# cv.imshow("Image",image)
# cv.waitKey(0)
prediction=0
with open('data/1_HQ_25_player_bbox.json') as json_file:
    data = json.load(json_file)
    while True:
        image = clear_image.copy()
        p = data['predictions'][prediction]
        for box in p['bboxes']:
            cv.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
        cv.imshow(f"Image frame",image)
        k=cv.waitKey(1)
        if k==27:
            break
        # if k==113: # Q
        #     continue
        if k==119: # W
            success,clear_image = vidcap.read()
            if not success:
                break
            else:
                count += 1
        if k==97: # A
            if prediction>0:
                prediction -= 1
        if k==115: # S
            prediction += 1
# for box in boxes[0]:
#     if i==length:
#         break
#     if score[i]>0.5:
#         cv.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
#     i += 1

