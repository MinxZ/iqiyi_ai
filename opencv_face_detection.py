import glob
import os
from collections import defaultdict

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

%matplotlib　inline

png_path = '../data/IQIYI_VID_DATA_Part_1/IQIYI_VID_DATA_Part1'
png_list = glob.glob(f'{png_path}/*.png')
face_cascade = cv2.CascadeClassifier(
    '../opencv/data/haarcascades/haarcascade_frontalface_default.xml')
data = defaultdict(list)
for index, img_path in tqdm(enumerate(png_list)):
    #     print(index, img_path)
    bgr_image = cv2.imread(img_path)
#     bgr_image = resizeAndPad(bgr_image, (512,512))
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_image)
    for i, (x, y, w, h) in enumerate(faces):
        #         data[img_path[-20:-13]].append(rgb_image[y:y+h, x:x+w])
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(f'{img_path[-20:]}', bgr_image)
