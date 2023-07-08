import csv
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import cv2
import scipy.io
import pickle
from matplotlib import pyplot as plt
import matplotlib.pyplot
import matplotlib._png as png
from PIL import Image
import numpy as np
from sklearn.svm import SVC


print('read list....')
filelist = []
f = csv.reader(open('./train_test_list/FFHQ/FFHQ30k_train_list.csv', 'r'))
for row in f:
    filelist.append([row[0],row[1]])

filelist_test = []
f = csv.reader(open('./train_test_list/FFHQ/FFHQ30k_test_list.csv', 'r'))
for row in f:
    filelist_test.append([row[0],row[1]])


print('extract feature....')
feature_18 = []
label = []
feature_121 = []
for i in range(len(filelist)):#(len(filelist)):  4000
    print(str(i))
    sub = filelist[i][0]
    im = png.read_png_int(filelist[i][0])
    # im = cv2.resize(im, (299, 299), interpolation = cv2.INTER_CUBIC)
    r = im.transpose(2, 0, 1)
    f_18 = compute_quality_features(r)
    label.append(float(filelist[i][1]))
    feature_18.append(f_18)
    f_121 = compute_msu_iqa_features(r)
    feature_121.append(f_121)
    # print('f')

# print("save feature....")
# with open('train_f18_partial_FFHQ_resize299.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows(feature_18)
#
# with open('train_f121_partial_FFHQ_resize299.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows(feature_121)
#
# with open('train_label_partial_forensics.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows([label])


print("extract test feature....")
# test feature
test_feature_18 = []
test_label = []
test_feature_121 = []
for i in range(len(filelist_test)):   #len(filelist_test)  len(filelist_test)
    print(str(i))
    sub = filelist_test[i][0]
    im = png.read_png_int(filelist_test[i][0])
    # im = cv2.resize(im, (299, 299), interpolation = cv2.INTER_CUBIC)
    r = im.transpose(2, 0, 1)
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
output = open('FFHQ30k_IMQ_1024.pkl', 'wb')
pickle.dump(data, output)
output.close()
print("DATA Saved")

# save as .mat
pkl_file = open('FFHQ30k_IMQ_1024.pkl', 'rb')
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
scipy.io.savemat('FFHQ30k_IMQ_1024.mat', {'train_f18': train_f18, "train_f121": train_f121, "train_label": train_label, "test_f18": test_f18, "test_f121": test_f121, "test_label": test_label})


# print("save test feature....")
# with open('test_feature_18_partial_forensics.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows(test_feature_18)
#
# with open('test_feature_121_partial_forensics.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows(test_feature_121)
#
# with open('test_label_partial_forensics.csv', 'w', newline='') as f:
#     ft = csv.writer(f)
#     ft.writerows([test_label])

# f = open('linear_svm_partial_forensics.txt', 'w')
# f1 = open('rbf_svm_partial_forensics.txt', 'w')
#
# # train SVM -1000
# # f18
# print('begin training: svm liner f18 for #1000:')
# svc_linear_f18_1000 = SVC(kernel='linear')
# svc_linear_f18_1000.fit(feature_18[0:1000], label[0:1000])
# res_svc_linear_f18_1000 = svc_linear_f18_1000.score(test_feature_18, test_label)
# print('svm liner f18 for #1000: {}'.format(res_svc_linear_f18_1000))
# f.write('svc_linear_f18_1000 accuracy = {:.4f}\n'.format(res_svc_linear_f18_1000))
#
# print('begin training: svm rbf f18 for #1000:')
# svc_rbf_f18_1000 = SVC(kernel='rbf')
# svc_rbf_f18_1000.fit(feature_18[0:1000], label[0:1000])
# res_svc_rbf_f18_1000 = svc_rbf_f18_1000.score(test_feature_18, test_label)
# print('svm rbf f18 for #1000: {}'.format(res_svc_rbf_f18_1000))
# f1.write('svc_rbf_f18_1000 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_1000))
#
# # f121
# print('begin training: svm liner f121 for #1000:')
# svc_linear_f121_1000 = SVC(kernel='linear')
# svc_linear_f121_1000.fit(feature_121[0:1000], label[0:1000])
# res_svc_linear_f121_1000 = svc_linear_f121_1000.score(test_feature_121, test_label)
# print('svm liner f121 for #1000: {}'.format(res_svc_linear_f121_1000))
# f.write('svc_linear_f121_1000 accuracy = {:.4f}\n'.format(res_svc_linear_f121_1000))
#
# print('begin training: svm rbf f121 for #1000:')
# svc_rbf_f121_1000 = SVC(kernel='rbf')
# svc_rbf_f121_1000.fit(feature_121[0:1000], label[0:1000])
# res_svc_rbf_f121_1000 = svc_rbf_f121_1000.score(test_feature_121, test_label)
# print('svm rbf f121 for #1000: {}'.format(res_svc_rbf_f121_1000))
# f1.write('svc_rbf_f121_1000 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_1000))
#
#
# # train SVM -2000
# # f18
# print('begin training: svm liner f18 for #2000:')
# svc_linear_f18_2000 = SVC(kernel='linear')
# svc_linear_f18_2000.fit(feature_18[0:2000], label[0:2000])
# res_svc_linear_f18_2000 = svc_linear_f18_2000.score(test_feature_18, test_label)
# print('svm liner f18 for #2000: {}'.format(res_svc_linear_f18_2000))
# f.write('svc_linear_f18_2000 accuracy = {:.4f}\n'.format(res_svc_linear_f18_2000))
#
# print('begin training: svm rbf f18 for #2000:')
# svc_rbf_f18_2000 = SVC(kernel='rbf')
# svc_rbf_f18_2000.fit(feature_18[0:2000], label[0:2000])
# res_svc_rbf_f18_2000 = svc_rbf_f18_2000.score(test_feature_18, test_label)
# print('svm rbf f18 for #2000: {}'.format(res_svc_rbf_f18_2000))
# f1.write('svc_rbf_f18_2000 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_2000))
#
# # f121
# print('begin training: svm liner f121 for #2000:')
# svc_linear_f121_2000 = SVC(kernel='linear')
# svc_linear_f121_2000.fit(feature_121[0:2000], label[0:2000])
# res_svc_linear_f121_2000 = svc_linear_f121_2000.score(test_feature_121, test_label)
# print('svm liner f121 for #2000: {}'.format(res_svc_linear_f121_2000))
# f.write('svc_linear_f121_2000 accuracy = {:.4f}\n'.format(res_svc_linear_f121_2000))
#
# print('begin training: svm rbf f121 for #2000:')
# svc_rbf_f121_2000 = SVC(kernel='rbf')
# svc_rbf_f121_2000.fit(feature_121[0:2000], label[0:2000])
# res_svc_rbf_f121_2000 = svc_rbf_f121_2000.score(test_feature_121, test_label)
# print('svm rbf f121 for #2000: {}'.format(res_svc_rbf_f121_2000))
# f1.write('svc_rbf_f121_2000 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_2000))
#
#
# # train SVM -3000
# # f18
# print('begin training: svm liner f18 for #3000:')
# svc_linear_f18_3000 = SVC(kernel='linear')
# svc_linear_f18_3000.fit(feature_18[0:3000], label[0:3000])
# res_svc_linear_f18_3000 = svc_linear_f18_3000.score(test_feature_18, test_label)
# print('svm liner f18 for #3000: {}'.format(res_svc_linear_f18_3000))
# f.write('svc_linear_f18_3000 accuracy = {:.4f}\n'.format(res_svc_linear_f18_3000))
#
# print('begin training: svm rbf f18 for #3000:')
# svc_rbf_f18_3000 = SVC(kernel='rbf')
# svc_rbf_f18_3000.fit(feature_18[0:3000], label[0:3000])
# res_svc_rbf_f18_3000 = svc_rbf_f18_3000.score(test_feature_18, test_label)
# print('svm rbf f18 for #3000: {}'.format(res_svc_rbf_f18_3000))
# f1.write('svc_rbf_f18_3000 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_3000))
#
# # f121
# print('begin training: svm liner f121 for #3000:')
# svc_linear_f121_3000 = SVC(kernel='linear')
# svc_linear_f121_3000.fit(feature_121[0:3000], label[0:3000])
# res_svc_linear_f121_3000 = svc_linear_f121_3000.score(test_feature_121, test_label)
# print('svm liner f121 for #3000: {}'.format(res_svc_linear_f121_3000))
# f.write('svc_linear_f121_3000 accuracy = {:.4f}\n'.format(res_svc_linear_f121_3000))
#
# print('begin training: svm rbf f121 for #3000:')
# svc_rbf_f121_3000 = SVC(kernel='rbf')
# svc_rbf_f121_3000.fit(feature_121[0:3000], label[0:3000])
# res_svc_rbf_f121_3000 = svc_rbf_f121_3000.score(test_feature_121, test_label)
# print('svm rbf f121 for #3000: {}'.format(res_svc_rbf_f121_3000))
# f1.write('svc_rbf_f121_3000 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_3000))
#
#
# # train SVM -4000
# # f18
# print('begin training: svm liner f18 for #4000:')
# svc_linear_f18_4000 = SVC(kernel='linear')
# svc_linear_f18_4000.fit(feature_18[0:4000], label[0:4000])
# res_svc_linear_f18_4000 = svc_linear_f18_4000.score(test_feature_18, test_label)
# print('svm liner f18 for #4000: {}'.format(res_svc_linear_f18_4000))
# f.write('svc_linear_f18_4000 accuracy = {:.4f}\n'.format(res_svc_linear_f18_4000))
#
# print('begin training: svm rbf f18 for #4000:')
# svc_rbf_f18_4000 = SVC(kernel='rbf')
# svc_rbf_f18_4000.fit(feature_18[0:4000], label[0:4000])
# res_svc_rbf_f18_4000 = svc_rbf_f18_4000.score(test_feature_18, test_label)
# print('svm rbf f18 for #4000: {}'.format(res_svc_rbf_f18_4000))
# f1.write('svc_rbf_f18_4000 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_4000))
#
# # f121
# print('begin training: svm liner f121 for #4000:')
# svc_linear_f121_4000 = SVC(kernel='linear')
# svc_linear_f121_4000.fit(feature_121[0:4000], label[0:4000])
# res_svc_linear_f121_4000 = svc_linear_f121_4000.score(test_feature_121, test_label)
# print('svm liner f121 for #4000: {}'.format(res_svc_linear_f121_4000))
# f.write('svc_linear_f121_4000 accuracy = {:.4f}\n'.format(res_svc_linear_f121_4000))
#
# print('begin training: svm rbf f121 for #4000:')
# svc_rbf_f121_4000 = SVC(kernel='rbf')
# svc_rbf_f121_4000.fit(feature_121[0:4000], label[0:4000])
# res_svc_rbf_f121_4000 = svc_rbf_f121_4000.score(test_feature_121, test_label)
# print('svm rbf f121 for #4000: {}'.format(res_svc_rbf_f121_4000))
# f1.write('svc_rbf_f121_4000 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_4000))
#
#
# f.close()
# f1.close()