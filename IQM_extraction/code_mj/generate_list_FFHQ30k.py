import csv

print('read list....')
filelist = []
f = csv.reader(open('./train_test_list/FFHQ/FFHQ_train_list.csv', 'r'))
for row in f:
    filelist.append([row[0], row[1]])

filelist_test = []
f = csv.reader(open('./train_test_list/FFHQ/FFHQ_test_list.csv', 'r'))
for row in f:
    filelist_test.append([row[0], row[1]])

FFHQ30k_train_img = []
FFHQ30k_train_label = []
# train list
count_real = 5000
count_fake = 5000
for i in range(30000): # len(filelist)
    if float(filelist[i][1]) == 0 and count_fake > 0:
        count_fake -= 1
        FFHQ30k_train_img.append(filelist[i][0])
        FFHQ30k_train_label.append(filelist[i][1])
    elif float(filelist[i][1]) == 1 and count_real > 0:
        count_real -= 1
        FFHQ30k_train_img.append(filelist[i][0])
        FFHQ30k_train_label.append(filelist[i][1])
    else:
        continue

with open('FFHQ30k_train_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(FFHQ30k_train_img, FFHQ30k_train_label))
    f.close()

FFHQ30k_test_img = []
FFHQ30k_test_label = []
# test list
count_real = 10000
count_fake = 10000
for i in range(50000): # len(filelist)
    if float(filelist_test[i][1]) == 0 and count_fake > 0:
        count_fake -= 1
        FFHQ30k_test_img.append(filelist_test[i][0])
        FFHQ30k_test_label.append(filelist_test[i][1])
    elif float(filelist_test[i][1]) == 1 and count_real > 0:
        count_real -= 1
        FFHQ30k_test_img.append(filelist_test[i][0])
        FFHQ30k_test_label.append(filelist_test[i][1])
    else:
        continue

with open('FFHQ30k_test_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(FFHQ30k_test_img, FFHQ30k_test_label))
    f.close()