"""
    This program extract frames from videos
"""

#! /usr/bin/env python
#coding=utf-8
import cv2
import os
import shutil
import csv

item = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


for m in range(len(item)):
    name = item[m]

    src_path = '/media/guo/500A/2_manipulate/' + name + '/raw/videos/'
    dst_path = '/media/guo/500B/zn/' + name + '/'
    if (os.path.exists(dst_path) is False):
        os.makedirs(dst_path)

    videolist = os.listdir(src_path)
    for n in range(len(videolist)):
        videoname = videolist[n]

        file = src_path + videoname

        videoCapture = cv2.VideoCapture(file)
        if (videoCapture.isOpened()):
            print(' Open the video: ' + videoname)
        else:
            print ('Fail to open ' + videoname)

        # fps = videoCapture.get(cv2.CAP_PROP_FPS)

        success, image = videoCapture.read()
        num = 1
        while(success):
            number = "{:06d}".format(num)
            if number == 60 or number == 100:
                cv2.imwrite(dst_path + videoname + '_frame_' + str(number) + '.jpg', image)
            num = num + 1
            if number > 120:
                break
            success, image = videoCapture.read()


        videoCapture.release()





            # start_frame = int(round(time_mark[0] * fps))
    # end_frame = int(round(time_mark[1] * fps))
    #
    # # videoWriter = cv2.VideoWriter(frame_dst, fourcc, fps, size)
    # frame_count = end_frame - start_frame
    # frame_number.append([folder, str(frame_count)])
    # n = 1
    # for i in range(0, end_frame+1):
    #     if i < start_frame:
    #         success, image = videoCapture.read()
    #         continue
    #     print (str(i) + '/' + str(frame_count))
    #     # videoCapture.set(cv2.CAP_PROP_POS_MSEC, i*1000)
    #     num = "{:06d}".format(n)
    #     success, image = videoCapture.read()
    #     if success:
    #         cv2.imwrite(frame_dst + '/frame_' + str(num) + '.jpg', image)
    #         n = n + 1
    #         # cv2.waitKey()




# frame_number = []
# frame_number.append(['subject','count'])
# t = csv.reader(open('scen8.txt'))
# for row in t:
#     tmp = row[0]
#     info = tmp.split('\t')
#     if info[0] == 'video_folder':
#         continue
#     folder = info[0].split('\t\n')[0]
#     name = info[1].split('\t\n')[0]
#     count = len(info)
#     time_mark = []
#     for m in range(2, count):
#         if info[m] == '':
#             break
#         time_m = info[m]
#         time_ms = time_m.split(':')
#         time_mm = int(time_ms[0]) * 60 * 60 + int(time_ms[1]) * 60 + int(time_ms[2])
#         time_mark.append(time_mm)
#
#     v_file = src_path + folder + '/' + name
#     print('%s/%s' % (folder, name))
#
#     if not os.path.exists(v_file):
#         print(name + ' non exist!')
#         continue
#
#
#
#
# # vidcap = cv2.VideoCapture('d:/video/keep/Le Sang Des Betes.mp4')
# # vidcap.set(cv2.CAP_PROP_POS_MSEC,20000)      # just cue to 20 sec. position
# # success,image = vidcap.read()
# # if success:
# #     cv2.imwrite("frame20sec.jpg", image)     # save frame as JPEG file
# #     cv2.imshow("20sec",image)
# #     cv2.waitKey()
#
#
#
#
# with open('scenario8_count.csv','w') as f:
#     file_t = csv.writer(f)
#     file_t.writerows(frame_number)




    # # get a period of video
    # videoWriter = cv2.VideoWriter(dst_path + "Brian_160125/Brian_160125_1_10.avi", fourcc, fps, size)
    # start_frame = int(round(time_mark[0] * fps))
    # end_frame = int(round(time_mark[1] * fps))
    # i = 1
    # frame_count = end_frame - start_frame
    # for i in range(start_frame, end_frame):
    #     print('%d  %d/%d' % (n, i, frame_count))
    #     success, frame = videoCapture.read()
    #     if success:
    #         videoWriter.write(frame)



    # frame_list = os.listdir(src_path)
    # number_sce = len(time_mark)
    # n = 0
    # start_sec = 1   # which scenario does it start?
    # for n in range(number_sce):
    #     scenario_id = start_sec + n
    #     print '%s/%s' % (scenario_id, number_sce)
    #     dst_name = dst_path + str(scenario_id)
    #     # directory = os.path.dirname(dst_name)
    #     if not os.path.exists(dst_name):
    #         os.makedirs(dst_name)
    #
    #     if n == 0:
    #         start_frame = 0
    #         end_frame = int(round(time_mark[n] * fps))
    #     else:
    #         start_frame = int(round(time_mark[n-1] * fps))
    #         end_frame = int(round(time_mark[n] * fps))

    #     i = 1
    #     frame_count = end_frame - start_frame
    #     for i in range(start_frame, end_frame):
    #         src = src_path + frame_list[i]
    #         dst = dst_name
    #         shutil.copy(src, dst)
    #
    #
    #     print 'done'
    #
    # videoCapture.release()