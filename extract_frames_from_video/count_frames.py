"""
    This program cut video into several scenarios by time marks
"""

#! /usr/bin/env python
#coding=utf-8
import csv
import cv2
import os

item = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
db = 'Face2Face'

ite = ['name', 'fps', 'frame_width', 'frame_height', 'fourcc', 'frame_count']
video_info = []
video_info.append(ite)

# src_path = 'E:/3 PhD_research/12 facefake/DB_manipulated/' + db + '/raw/videos/'
src_path = 'F:/00000_research/Face2Face/Face2Face/raw/videos/'
video_list = os.listdir(src_path)
sum = 0
for n in range(len(video_list)):
    videoname = video_list[n]
    if n%100 == 0:
        print str(n)
    videoCapture = cv2.VideoCapture(src_path + videoname)
    # if (videoCapture.isOpened()):
    #     print 'Open'
    # else:
    #     print 'Fail to open!'

    # get frame rate, size, codec
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    sum += frame_count

    video_info.append([videoname, str(fps), str(size[0]), str(size[1]), str(fourcc), str(frame_count)])
    videoCapture.release()

video_info.append(['total', '', '', '', '', str(sum)])

with open(db + '_frame_count.csv', 'wb') as f:
    ft = csv.writer(f)
    ft.writerows(video_info)