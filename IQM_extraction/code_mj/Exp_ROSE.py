import csv
import os.path
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import cv2
import pickle
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.pyplot
import matplotlib._png as png
from sklearn.svm import SVC

###################################################################
## Feature Extracting
train_path = '/home/guo/Min/dataset/ROSE-Youtu/Croped/Test'

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
        if level2_dict[j] == 'print' or level2_dict[j] == 'replay':
            annotation = 0
        if level2_dict[j] == 'real':
            annotation = 1
        sub_list = os.listdir(train_path + '/' + level1_dict[i] + '/' + level2_dict[j])
        for ll in range(len(sub_list)):
            print(sub_list[ll])
            img_list = os.listdir(train_path + '/' + level1_dict[i] + '/' + level2_dict[j] + '/' + sub_list[ll])
            frame_interval = len(img_list) // 30
            ss = 0
            for k in range(0, len(img_list), frame_interval):
                print('{} is {}'.format(ss, img_list[k]))
                ss += 1
                #if k % 100 == 0:
                #    print(k)
                img_path = train_path + '/' + level1_dict[i] + '/' + level2_dict[j] + '/' + sub_list[ll] + '/' + img_list[k]
                im = cv2.imread(img_path, 1)
                if im is None:
                    continue
                r = im.transpose(2, 0, 1)
                if r.shape[1] < 50 or r.shape[2] < 50:
                    continue
                f_18 = compute_quality_features(r)
                label.append(annotation)
                feature_18.append(f_18)
                f_121 = compute_msu_iqa_features(r)
                feature_121.append(f_121)
                path.append(img_path)

############################################
print('Save Features....')

with open('test_ROSE_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(path, label))
f.close()

# save as .pkl
data = {}
data["test_f18"] = feature_18
data["test_f121"] = feature_121
data["test_label"] = label
output = open('ROSE_Test_IMQ.pkl', 'wb')
pickle.dump(data, output)
output.close()
print("DATA Saved")

# save as .mat
pkl_file = open('ROSE_Test_IMQ.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
f18_train = data["test_f18"]
f121_train = data["test_f121"]
label_train = data["test_label"]
label_train = list(np.float_(label_train))
scipy.io.savemat('ROSE_Test_IMQ.mat', {'test_f18': f18_train, "test_f121": f121_train, "test_label": label_train})


