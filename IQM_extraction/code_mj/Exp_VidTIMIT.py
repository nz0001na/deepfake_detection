import csv
import os.path
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot
import matplotlib._png as png
from sklearn.svm import SVC

## Feature Extracting
train_path = '/home/guo/Min/dataset/VidTIMIT_deepfakeTIMIT_cropped faces/Test/lower_quality'
# VidTIMIT:1, lower_quality:0, higher_quality:0
annotation = 0
feature_18 = []
label = []
feature_121 = []
path = []
level1_dict = os.listdir(train_path)
for i in range(len(level1_dict)):
    print(level1_dict[i])
    level2_dict = os.listdir(train_path + '/' + level1_dict[i])
    for j in range(len(level2_dict)):
        print(level2_dict[j])
        img_list = os.listdir(train_path + '/' + level1_dict[i] + '/' + level2_dict[j])
        #frame_interval = len(img_list) // 10
        for k in range(len(img_list)): # frame_interval
            if k % 100 == 0:
                print(k)
            img_path = train_path + '/' + level1_dict[i] + '/' + level2_dict[j] + '/' + img_list[k]
            im = cv2.imread(img_path, 1)
            r = im.transpose(2, 0, 1)
            f_18 = compute_quality_features(r)
            label.append(annotation)
            feature_18.append(f_18)
            f_121 = compute_msu_iqa_features(r)
            feature_121.append(f_121)
            path.append(img_path)

print("save feature....")
with open('test_VidTimit_Low_f18.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_18)

with open('test_VidTimit_Low_f121.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_121)

with open('test_VidTimit_Low_label.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows([label])

with open('test_VidTimit_Low_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(path, label))
f.close()


train_path = '/home/guo/Min/dataset/VidTIMIT_deepfakeTIMIT_cropped faces/Test/higher_quality'
# VidTIMIT:1, lower_quality:0, higher_quality:0
annotation = 0
feature_18 = []
label = []
feature_121 = []
path = []
level1_dict = os.listdir(train_path)
for i in range(len(level1_dict)):
    print(level1_dict[i])
    level2_dict = os.listdir(train_path + '/' + level1_dict[i])
    for j in range(len(level2_dict)):
        print(level2_dict[j])
        img_list = os.listdir(train_path + '/' + level1_dict[i] + '/' + level2_dict[j])
        #frame_interval = len(img_list) // 10
        for k in range(len(img_list)): # frame_interval
            if k % 100 == 0:
                print(k)
            img_path = train_path + '/' + level1_dict[i] + '/' + level2_dict[j] + '/' + img_list[k]
            im = cv2.imread(img_path, 1)
            r = im.transpose(2, 0, 1)
            f_18 = compute_quality_features(r)
            label.append(annotation)
            feature_18.append(f_18)
            f_121 = compute_msu_iqa_features(r)
            feature_121.append(f_121)
            path.append(img_path)

print("save feature....")
with open('test_VidTimit_High_f18.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_18)

with open('test_VidTimit_High_f121.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_121)

with open('test_VidTimit_High_label.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows([label])

with open('test_VidTimit_High_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(path, label))
f.close()