import os
import cv2

src_path = 'F:/6 Autism/1_scenario_8_frame/'
# des_path = 'F:/6 Autism/1_scenario_8_frame_342_256/'
des_path = 'J:/1_scenario_8_frame_224_224/new/'

subject_list = os.listdir(src_path)
for i in range(len(subject_list)):
    subject_name = subject_list[i].split(' ')[0]

    if subject_name != 'Damian_140416':
        continue

    print str(i) + ' / ' + str(len(subject_list)) + ' : ' + subject_name

    # create destnation
    dst_subject_path = des_path + subject_name + '/'
    if os.path.exists(dst_subject_path) is False:
        os.makedirs(dst_subject_path)

    frame_path = src_path + subject_name + '/'
    img_list = sorted(os.listdir(frame_path))
    for j in range(len(img_list)):
        img_name = img_list[j]
        img_data = frame_path + img_name
        img = cv2.imread(img_data)
        if (img is None):
            continue
        crop_img = img[190:414, 185:409]
        # crop_img = img[int(y1):int(y2), int(x1):int(x2)]

        dst_img = dst_subject_path + img_name
        cv2.imwrite(dst_img, crop_img)


print 'done'