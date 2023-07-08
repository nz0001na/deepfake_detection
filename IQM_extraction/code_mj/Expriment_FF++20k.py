import csv
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import cv2
import numpy as np
import pickle
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.pyplot
import matplotlib._png as png
from sklearn.svm import SVC

print('read list....')
filelist = []
f = csv.reader(open('/home/guo/Min/DeepFake/DeepFakeDetection-master/Experiments_Faces-HQ/train_test_list/FF++_20K/Forensics_20k_Raw_train_list.csv', 'r'))
for row in f:
    filelist.append([row[0],row[1]])

filelist_test = []
f = csv.reader(open('/home/guo/Min/DeepFake/DeepFakeDetection-master/Experiments_Faces-HQ/train_test_list/FF++_20K/Forensics_20k_Raw_test_list.csv', 'r'))
for row in f:
    filelist_test.append([row[0],row[1]])

print('extract train feature....')
feature_18 = []
label = []
feature_121 = []
for i in range(len(filelist)):
    print(str(i))
    im = cv2.imread(filelist[i][0], 1)
    r = im.transpose(2, 0, 1)
    if r.shape[1] < 70 or r.shape[2] < 50:
        continue
    f_18 = compute_quality_features(r)
    label.append(float(filelist[i][1]))
    feature_18.append(f_18)
    f_121 = compute_msu_iqa_features(r)
    feature_121.append(f_121)

print("extract test feature....")
# test feature
test_feature_18 = []
test_label = []
test_feature_121 = []
for i in range(len(filelist_test)):
    print(str(i))
    im = cv2.imread(filelist_test[i][0], 1)
    r = im.transpose(2, 0, 1)
    if r.shape[1] < 70 or r.shape[2] < 50:
        continue
    f_18 = compute_quality_features(r)
    test_label.append(float(filelist_test[i][1]))
    test_feature_18.append(f_18)
    f_121 = compute_msu_iqa_features(r)
    test_feature_121.append(f_121)

# save as .pkl
data = {}
data["train_f18"] = feature_18
data["train_f121"] = feature_121
data["train_label"] = label
data["test_f18"] = test_feature_18
data["test_f121"] = test_feature_121
data["test_label"] = test_label
output = open('Forensics_20k_Raw.pkl', 'wb')
pickle.dump(data, output)
output.close()
print("DATA Saved")

# save as .mat
pkl_file = open('Forensics_20k_Raw.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
train_f18 = data["train_f18"]
train_f121 = data["train_f121"]
train_label = data["train_label"]
test_f18 = data["test_f18"]
test_f121 = data["test_f121"]
test_label = data["test_label"]
train_label = list(np.float_(train_label))
test_label = list(np.float_(test_label))
scipy.io.savemat('Forensics_20k_Raw.mat', {'train_f18': train_f18, "train_f121": train_f121, "train_label": train_label, "test_f18": test_f18, "test_f121": test_f121, "test_label": test_label})


