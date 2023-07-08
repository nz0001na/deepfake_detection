import cv2
import os.path
import radialProfile
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
import csv
from scipy.interpolate import griddata
from matplotlib import pyplot as plt

# read Train images
print('read Train list....')
train_img_path = '/home/guo/Min/DeepFake/DeepFakeDetection-master/Experiments_Faces-HQ/train_test_list/FFHQ/FFHQ_train_list.csv'
train_img_list = []
f = csv.reader(open(train_img_path, 'r'))
for row in f:
    train_img_list.append([row[0], row[1]])

Train_list = []
Train_label = []
for i in range(0, 4000): # len(train_img_list)
    Train_list.append(train_img_list[i][0])
    Train_label.append(float(train_img_list[i][1]))

resize_num = 800
epsilon = 1e-8
data = {}
psd1D_train = []
for i in range(0, 4000): # len(Train_list)
    print('%d %d\n' % (i, len(Train_list)))
    img = cv2.imread(Train_list[i], 0)
    #img = cv2.resize(img, (resize_num, resize_num), interpolation = cv2.INTER_CUBIC) # INTER_AREA, INTER_CUBIC, INTER_LINEAR
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    psd1D_train.append(psd1D)

# read Test images
print('read Test list....')
test_img_path = '/home/guo/Min/DeepFake/DeepFakeDetection-master/Experiments_Faces-HQ/train_test_list/FFHQ/FFHQ_test_list.csv'
test_img_list = []
f = csv.reader(open(test_img_path, 'r'))
for row in f:
    test_img_list.append([row[0], row[1]])

Test_list = []
Test_label = []
for i in range(0, 2000): # len(test_img_list)
    Test_list.append(test_img_list[i][0])
    Test_label.append(float(test_img_list[i][1]))

epsilon = 1e-8
psd1D_test = []
for i in range(0, 2000): # len(Test_list)
    print('%d %d\n' % (i, len(Test_list)))
    img = cv2.imread(Test_list[i], 0)
    # img = cv2.resize(img, (resize_num, resize_num), interpolation = cv2.INTER_CUBIC)  # INTER_AREA, INTER_CUBIC, INTER_LINEAR
    # Calculate FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = radialProfile.azimuthalAverage(magnitude_spectrum)
    psd1D_test.append(psd1D)

data["train_data"] = psd1D_train
data["train_label"] = Train_label
data["test_data"] = psd1D_test
data["test_label"] = Test_label

# save as .pkl
output = open('FFHQ_all_resize800.pkl', 'wb')
pickle.dump(data, output)
output.close()
print("DATA Saved")

# save as .mat
import scipy.io
pkl_file = open('FFHQ_all_resize800.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
train_data = data["train_data"]
train_label = data["train_label"]
test_data = data["test_data"]
test_label = data["test_label"]
scipy.io.savemat('FFHQ_all_resize800_data.mat', {'train_data':train_data, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label})

