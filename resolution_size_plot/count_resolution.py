
import os
import csv
import cv2

# signle folder
# pa = 'E:/Face2Face/'
# size_list = []
# size_list.append(['name', 'height', 'width'])
#
# img_list = os.listdir(pa)
# for i in range(len(img_list)):
#     name = img_list[i]
#
#     img = cv2.imread(pa + name)
#     height = img.shape[0]
#     width = img.shape[1]
#     size_list.append([name, height, width])
#     print('d')


# with subfolder

# pa = 'E:/ff++/ff_data_4000_crop_dlib/Face2Face/'
# pa = 'E:/ff++/ff_data_4000_crop_face/Face2Face/'
# size_list = []
# size_list.append(['name', 'height', 'width'])
#
# img_list = os.listdir(pa)
# for i in range(len(img_list)):
#     fold = img_list[i]
#     fold_path = pa + fold + '/'
#
#     s_list = os.listdir(fold_path)
#     for j in range(len(s_list)):
#         name = s_list[j]
#         img = cv2.imread(fold_path + name)
#         try:
#             img.shape
#             print("checked for shape".format(img.shape))
#         except AttributeError:
#             print("shape not found")
#             continue
#
#         height = img.shape[0]
#         width = img.shape[1]
#         size_list.append([name, height, width])
#         print('d')

# vidtimit
pa = '/home/zn/Downloads/vid_timit_crop_dlib/'
size_list = []
size_list.append(['name', 'height', 'width'])

img_list = os.listdir(pa)
for i in range(len(img_list)):
    fold = img_list[i]
    fold_path = pa + fold + '/'

    s_list = os.listdir(fold_path)
    for j in range(len(s_list)):
        f = s_list[j]
        f_path = fold_path + f + '/'

        p_list = os.listdir(f_path)
        for k in range(len(p_list)):
            t = p_list[k]

            q_path = f_path + t + '/'

            q_list = os.listdir(q_path)
            for l in range(len(q_list)):
                name = q_list[l]
                img = cv2.imread(q_path + name)
                try:
                    img.shape
                    print("checked for shape".format(img.shape))
                except AttributeError:
                    print("shape not found")
                    continue

                height = img.shape[0]
                width = img.shape[1]
                size_list.append([name, height, width])
                print('d')

with open('vidtimit_dlib_crop.csv', 'wb') as f:
    ft = csv.writer(f)
    ft.writerows(size_list)