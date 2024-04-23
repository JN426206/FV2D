# import caffe
import lmdb
import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt
# from caffe.proto import caffe_pb2

# lmdb_env = lmdb.open('ncexclude/bbox_lmdb')
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()

# for key, value in lmdb_cursor:
#     print(key, value)
#     break

lmdb_env = lmdb.open('ncexclude/bbox_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
# datum = caffe_pb2.Datum()
#seg_id,video_name,start_time,end_time,event_start_time,event_end_time,cls_id,highlight_cls
#001-b256a299-d2c5-11e8-967f-1c36bbecf341,c925a918cc7511e885786c96cfde8f.mp4,00:14:55,00:15:10,00:15:00,00:15:05,0,0
#Offset = Event start at 00:15:00 fragment start at 00:14:55 so 00:15:00-00:14:55=00:00:05 Video have 25f/s 5*25=125 offset[0]=125. All event take 00:15:05-00:15:00=00:00:05 5*25=125 offset[64]=248 (248-125)/25=123/25=4,92seconds. So frames with bboxes are on 00:14:55+125frames, 00:14:55+127frames,..., 00:14:55+248frames.
#By VirtualDub offset 125 is on 22496 (00:14:59.840) frame in c925a918cc7511e885786c96cfde8f.mp4
#011-26dbe219-d2c6-11e8-84a5-1c36bbecf341,c925a918cc7511e885786c96cfde8f.mp4,00:20:50,00:21:08,00:20:56,00:21:02,3 6,0
#c925a918cc7511e885786c96cfde8f_100_origin_000001_010621,c925a918cc7511e885786c96cfde8f.mp4,00:00:23,00:00:35,00:00:23,00:00:35,0,0
seg_id='001-b256a299-d2c5-11e8-967f-1c36bbecf341'
index=0
offset_b = lmdb_txn.get((seg_id+'_offset').encode())
offset=np.frombuffer(offset_b, dtype=np.int16)
print(offset)
print(np.sum(offset))
offset_shape = np.shape(offset)
print(offset_shape[0])

bboxes_b = lmdb_txn.get((seg_id+'_bboxes').encode())
boxes=np.ndarray((offset_shape[0],32,4),np.int16,bboxes_b)
print(boxes[0])

length_b = lmdb_txn.get((seg_id+'_length').encode())
length = int(np.frombuffer(length_b, dtype=np.int8)[0])
print(length)

score_b = lmdb_txn.get((seg_id+'_score').encode())
score = np.frombuffer(score_b, dtype=np.float32)
print(score)
img = cv.imread("screenshots\\225000000.png", cv.IMREAD_COLOR)
i=0
for box in boxes[0]:
    if i==length:
        break
    if score[i]>0.5:
        cv.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,0,255), 2)
    i += 1
cv.imshow("Image",img)
cv.waitKey(0)
# print(offset)
# for key, value in lmdb_cursor:
#     print(key)
#     # print(value) 
#     if "offset" in str(key):
#         print(value)
#     if index == 5:
#         break
#     index += 1
# for key, value in lmdb_cursor:
#     print(key)
#     # print(value)
#     # deserialized_bytes = np.frombuffer(bytes, dtype=np.int8)
#     # deserialized_x = np.reshape(deserialized_bytes, newshape=(2, 2))
#     if "offset" in str(key):
#         offset=np.frombuffer(value, dtype=np.int16)
#         print(np.shape(offset))
#     elif "length" in str(key):
#         print(np.shape(np.frombuffer(value, dtype=np.int8)))
#     elif "bboxes" in str(key):
#         print(np.shape(np.frombuffer(value, dtype=np.int16)))
#         offset_shape=np.shape(offset)
#         if offset != None and offset_shape != 0:
#             print(offset)
#             print(np.ndarray((offset_shape[0],32,4),np.int16,value))
#     elif "ids" in str(key):
#         print(np.frombuffer(value, dtype=np.int8))  
#     elif "score" in str(key):
#         print(np.shape(np.frombuffer(value, dtype=np.float32)))
#     # label = datum.label
#     # # data = caffe.io.datum_to_array(datum)
#     # # im = data.astype(np.uint8)
#     # # im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
#     # print ("label ", label)

#     # plt.imshow(im)
#     # plt.show()
#     # break
#     if index == 5:
#         break
#     index += 1