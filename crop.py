# %%
import glob
import multiprocessing
import os
import pickle
from collections import defaultdict

import cv2 as cv2
import face_recognition
import numpy as np
from tqdm import tqdm

from model import resizeAndPad

# %%
data_path = '../data/IQIYI_VID_DATA_Part1'
for train_val in ['train']:
    png_list = glob.glob(f'{data_path}/png_{train_val}/*.png')

    if train_val == 'train':
        with open(f'{data_path}/{train_val}.txt') as f:
            contents = f.readlines()
        id2file = defaultdict(list)
        file2id = {}
        for index, line in enumerate(contents):
            file_path, id = line.rstrip('\n').split(' ')
            id2file[int(id)].append(file_path)
            file2id[file_path[-10:]] = int(id)
    else:
        with open(f'{data_path}/{train_val}.txt') as f:
            contents = f.readlines()
        id2file = defaultdict(list)
        file2id = {}
        for index, line in enumerate(contents):
            id_file_path = line.rstrip('\n').split(' ')
            id = int(id_file_path[0])
            id2file[id].extend(id_file_path[1:])
            for index1, file_path in enumerate(id_file_path[1:]):
                file2id[file_path[-10:]] = id

    faces = []
    ids = []
    for index1, img_path in tqdm(enumerate(png_list)):
        bgr_image = cv2.imread(img_path)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if not len(face_locations) == 0:
            id = file2id[img_path[-19:-9]]
            ids.append(id)

            crop = face_locations[0]
            y1, x1, y2, x2 = crop
            bgr_image = bgr_image[y1:y2, x2:x1, :]
            bgr_image = resizeAndPad(bgr_image, (224, 224))
            cv2.imwrite(
                f'{data_path}/face_{train_val}/{id}_{img_path[-20:]}', bgr_image)
