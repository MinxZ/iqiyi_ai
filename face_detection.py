# %%
import glob
import os
from collections import defaultdict

import cv2 as cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

%matplotlib　inline
# %%
mp4_path = '../data/IQIYI_VID_DATA_Part_1/IQIYI_VID_DATA_Part1/IQIYI_VID_TRAIN'
mp4_list = glob.glob(f'{mp4_path}/*.mp4')
len(mp4_list)
cap = cv2.VideoCapture(mp4_list[0])
while(cap.isOpened()):
    ret, frame = cap.read()
    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# %%
png_path = '../data/IQIYI_VID_DATA_Part_1/IQIYI_VID_DATA_Part1'
png_list = glob.glob(f'{png_path}/*.png')
len(png_list)
# %%
for index1, img_path in tqdm(enumerate(png_list)):
    bgr_image = cv2.imread(img_path)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        cv2.imwrite(f'{img_path[-20:]}', bgr_image)
    else:
        for index, crop in enumerate(face_locations):
            y1, x1, y2, x2 = crop
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite(f'{img_path[-20:]}', bgr_image)

# %%
