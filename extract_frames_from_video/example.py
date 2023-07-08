"""
    This program extract frames from videos
"""

#! /usr/bin/env python
#coding=utf-8
import cv2
import os
import shutil
import csv


dst_path = 'D:/2_zn_research/9_deepfake/images/'
if (os.path.exists(dst_path) is False):
    os.makedirs(dst_path)

file = '008_990.mp4'

videoCapture = cv2.VideoCapture(file)
if (videoCapture.isOpened()):
    print(' Open the video: ' + file)
else:
    print ('Fail to open ' + file)

# fps = videoCapture.get(cv2.CAP_PROP_FPS)

success, image = videoCapture.read()
num = 1
while(success):
    number = "{:06d}".format(num)
    cv2.imwrite(dst_path + 'frame_' + str(number) + '.jpg', image)
    num = num + 1
    print 'num: '+str(num)
    success, image = videoCapture.read()


videoCapture.release()




