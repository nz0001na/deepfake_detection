import os
import csv
import cv2


img_path = '/media/guo/500B/1_code/steg_code/crop_face/'
f_list = os.listdir(img_path)

for i in range(len(f_list)):
    f = f_list[i]

    count = 0
    f_path = img_path + f + '/'
    img_list = os.listdir(f_path)
    print f + ' (1): ' + str(len(img_list))
    for j in range(len(img_list)):
        img_name = img_list[j]

        img = cv2.imread(f_path + img_name)
        if img == None:
            os.remove(f_path + img_name)
            count += 1
            continue
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        depth = img_shape[2]
        if height <70 or width <70:
            os.remove(f_path + img_name)
            count += 1
        # print 'd'
    print f + ' (2): ' + str(count)



