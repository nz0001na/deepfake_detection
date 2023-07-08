import csv
from bob.ip.qualitymeasure import compute_quality_features, compute_msu_iqa_features
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot
import matplotlib._png as png
from sklearn.svm import SVC


print('read list....')
filelist = []
f = csv.reader(open('Forensics_partial_train_list.csv', 'r'))
for row in f:
    filelist.append([row[0],row[1]])

filelist_test = []
f = csv.reader(open('Forensics_partial_test_list.csv', 'r'))
for row in f:
    filelist_test.append([row[0],row[1]])


print('extract feature....')
feature_18 = []
label = []
feature_121 = []
for i in range(len(filelist)):#(len(filelist)):  4000
    print(str(i))
    # print(str(i))
    im = cv2.imread(filelist[i][0], 1)
    r = im.transpose(2, 0, 1)
    if r.shape[1] < 70 or r.shape[2] < 50:
        continue
    f_18 = compute_quality_features(r)
    label.append(float(filelist[i][1]))
    feature_18.append(f_18)
    f_121 = compute_msu_iqa_features(r)
    feature_121.append(f_121)
    # print('f')

print("save feature....")
with open('train_feature_18_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_18)

with open('train_feature_121_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(feature_121)

with open('train_label_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows([label])


print("extract test feature....")
# test feature
test_feature_18 = []
test_label = []
test_feature_121 = []
for i in range(len(filelist_test)):   #len(filelist_test)  len(filelist_test)
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
    # print('f')

print("save test feature....")
with open('test_feature_18_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(test_feature_18)

with open('test_feature_121_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows(test_feature_121)

with open('test_label_partial_forensics.csv', 'w', newline='') as f:
    ft = csv.writer(f)
    ft.writerows([test_label])

# read features
train_f18 = []
f = csv.reader(open('train_feature_18_partial_forensics.csv', 'r'))
for row in f:
    train_f18.append(row)

train_f121 = []
f = csv.reader(open('train_feature_121_partial_forensics.csv', 'r'))
for row in f:
    train_f121.append(row)

train_label = []
f = csv.reader(open('train_label_partial_forensics.csv', 'r'))
for row in f:
    train_label.append(row)
train_label = train_label[0]

test_f18 = []
f = csv.reader(open('test_feature_18_partial_forensics.csv', 'r'))
for row in f:
    test_f18.append(row)

test_f121 = []
f = csv.reader(open('test_feature_121_partial_forensics.csv', 'r'))
for row in f:
    test_f121.append(row)

test_label = []
f = csv.reader(open('test_label_partial_forensics.csv', 'r'))
for row in f:
    test_label.append(row)
test_label = test_label[0]

# train SVM -1546
#f = open('linear_svm_partial_forensics.txt', 'w')
#f1 = open('rbf_svm_partial_forensics.txt', 'w')
# f18
print('begin training: svm linear f18 for #1546:')
svc_linear_f18_1546 = SVC(kernel='linear')
svc_linear_f18_1546.fit(train_f18, train_label)
res_svc_linear_f18_1546 = svc_linear_f18_1546.score(test_f18, test_label)
print('svm liner f18 for #1546: {}'.format(res_svc_linear_f18_1546))
#f.write('svc_linear_f18_1546 accuracy = {:.4f}\n'.format(res_svc_linear_f18_1546))

print('begin training: svm rbf f18 for #1546:')
svc_rbf_f18_1546 = SVC(kernel='rbf')
svc_rbf_f18_1546.fit(train_f18, train_label)
res_svc_rbf_f18_1546 = svc_rbf_f18_1546.score(test_f18, test_label)
print('svm rbf f18 for #1546: {}'.format(res_svc_rbf_f18_1546))
#f1.write('svc_rbf_f18_1546 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_1546))

# f121
print('begin training: svm liner f121 for #1546:')
svc_linear_f121_1546 = SVC(kernel='linear')
svc_linear_f121_1546.fit(train_f121, train_label)
test_f121 = np.nan_to_num(test_f121)
res_svc_linear_f121_1546 = svc_linear_f121_1546.score(test_f121[0:2050], test_label[0:2050])
print('svm liner f121 for #1546: {}'.format(res_svc_linear_f121_1546))
#f.write('svc_linear_f121_1546 accuracy = {:.4f}\n'.format(res_svc_linear_f121_1546))

print('begin training: svm rbf f121 for #1546:')
svc_rbf_f121_1546 = SVC(kernel='rbf')
svc_rbf_f121_1546.fit(train_f121, train_label)
res_svc_rbf_f121_1546 = svc_rbf_f121_1546.score(test_f121[0:2050], test_label[0:2050])
print('svm rbf f121 for #1546: {}'.format(res_svc_rbf_f121_1546))
#f1.write('svc_rbf_f121_1546 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_1546))

# f18+f121
print('begin training: svm liner f139 for #1546:')
svc_linear_f139_1546 = SVC(kernel='linear')
train_f139 = np.concatenate((train_f18, train_f121), axis=1)
svc_linear_f139_1546.fit(train_f139, train_label)
test_f139 = np.concatenate((test_f18, test_f121), axis=1)
res_svc_linear_f139_1546 = svc_linear_f139_1546.score(test_f139[0:2050], test_label[0:2050])
print('svm liner f139 for #1546: {}'.format(res_svc_linear_f139_1546))
#f.write('svc_linear_f139_1546 accuracy = {:.4f}\n'.format(res_svc_linear_f139_1546))

print('begin training: svm rbf f139 for #1546:')
svc_rbf_f139_1546 = SVC(kernel='rbf')
svc_rbf_f139_1546.fit(train_f139, train_label)
res_svc_rbf_f139_1546 = svc_rbf_f139_1546.score(test_f139[0:2050], test_label[0:2050])
print('svm rbf f139 for #1546: {}'.format(res_svc_rbf_f139_1546))
#f1.write('svc_rbf_f139_1546 accuracy = {:.4f}\n'.format(res_svc_rbf_f139_1546))

# train SVM -4000
# f18
print('begin training: svm liner f18 for #4000:')
svc_linear_f18_4000 = SVC(kernel='linear')
train_f18_4000 = np.concatenate((train_f18, test_f18[0:2458]), axis=0)
train_label_4000 = np.concatenate((train_label, test_label[0:2458]), axis=0)
svc_linear_f18_4000.fit(train_f18_4000, train_label_4000)
res_svc_linear_f18_4000 = svc_linear_f18_4000.score(test_f18[2459:], test_label[2459:])
print('svm liner f18 for #4000: {}'.format(res_svc_linear_f18_4000))
#f.write('svc_linear_f18_4000 accuracy = {:.4f}\n'.format(res_svc_linear_f18_4000))

print('begin training: svm rbf f18 for #4000:')
svc_rbf_f18_4000 = SVC(kernel='rbf')
svc_rbf_f18_4000.fit(train_f18_4000, train_label_4000)
res_svc_rbf_f18_4000 = svc_rbf_f18_4000.score(test_f18[2459:], test_label[2459:])
print('svm rbf f18 for #4000: {}'.format(res_svc_rbf_f18_4000))
#f1.write('svc_rbf_f18_4000 accuracy = {:.4f}\n'.format(res_svc_rbf_f18_4000))

# f121
print('begin training: svm liner f121 for #4000:')
svc_linear_f121_4000 = SVC(kernel='linear')
train_f121_4000 = np.concatenate((train_f121, test_f121[0:2458]), axis=0)
svc_linear_f121_4000.fit(train_f121_4000, train_label_4000)
res_svc_linear_f121_4000 = svc_linear_f121_4000.score(test_f121[2459:], test_label[2459:])
print('svm liner f121 for #4000: {}'.format(res_svc_linear_f121_4000))
#f.write('svc_linear_f121_4000 accuracy = {:.4f}\n'.format(res_svc_linear_f121_4000))

print('begin training: svm rbf f121 for #4000:')
svc_rbf_f121_4000 = SVC(kernel='rbf')
svc_rbf_f121_4000.fit(train_f121_4000, train_label_4000)
res_svc_rbf_f121_4000 = svc_rbf_f121_4000.score(test_f121[2459:], test_label[2459:])
print('svm rbf f121 for #2000: {}'.format(res_svc_rbf_f121_4000))
#f1.write('svc_rbf_f121_2000 accuracy = {:.4f}\n'.format(res_svc_rbf_f121_4000))

# f18+f121
print('begin training: svm liner f139 for #1546:')
svc_linear_f139_4000 = SVC(kernel='linear')
train_f139_4000 = np.concatenate((train_f18_4000, train_f121_4000), axis=1)
svc_linear_f139_1546.fit(train_f139_4000, train_label_4000)
test_f139_4000 = np.concatenate((test_f18[2459:], test_f121[2459:]), axis=1)
res_svc_linear_f139_1546 = svc_linear_f139_1546.score(test_f139_4000, test_label[2459:])
print('svm liner f139 for #1546: {}'.format(res_svc_linear_f139_1546))
#f.write('svc_linear_f139_1546 accuracy = {:.4f}\n'.format(res_svc_linear_f139_1546))

print('begin training: svm rbf f139 for #1546:')
svc_rbf_f139_1546 = SVC(kernel='rbf')
svc_rbf_f139_1546.fit(train_f139_4000, train_label_4000)
res_svc_rbf_f139_1546 = svc_rbf_f139_1546.score(test_f139_4000, test_label[2459:])
print('svm rbf f139 for #1546: {}'.format(res_svc_rbf_f139_1546))
#f1.write('svc_rbf_f139_1546 accuracy = {:.4f}\n'.format(res_svc_rbf_f139_1546))


f.close()
f1.close()