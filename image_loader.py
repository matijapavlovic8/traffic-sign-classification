import os
import cv2
import numpy as np


def load(path):
    images = []
    class_no = []

    cnt = 0

    class_list = os.listdir(path)
    print("Total Classes Detected:", len(class_list))
    classes_len = len(class_list)
    print("Importing Classes.....")
    for x in range(0, classes_len):
        myPicList = os.listdir(path + "/" + str(cnt))
        for y in myPicList:
            curr = cv2.imread(path + "/" + str(cnt) + "/" + y)
            images.append(curr)
            class_no.append(cnt)
        print(cnt, end=" ")
        cnt += 1
    print(" ")
    images = np.array(images)
    class_no = np.array(class_no)
    return {'images': images, 'class_no': class_no, 'classes_len': classes_len}
