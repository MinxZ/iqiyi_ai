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

# %matplotlib　inline
# %%
data_path = '../data/IQIYI_VID_DATA_Part1'
png_list = glob.glob(f'{data_path}/png_train/*.png')

with open(data_path + '/train.txt') as f:
    contents = f.readlines()
id2file = defaultdict(list)
file2id = {}
for index, line in enumerate(contents):
    file_path, id = line.rstrip('\n').split(' ')
    id2file[int(id)].append(file_path)
    file2id[file_path[-10:]] = int(id)

# %%
faces = []
ids = []
for index1, img_path in tqdm(enumerate(png_list)):
    bgr_image = cv2.imread(img_path)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    if not len(face_locations) == 0:
        crop = face_locations[0]
        y1, x1, y2, x2 = crop
        bgr_image = bgr_image[y1:y2, x2:x1, :]
        bgr_image = resizeAndPad(bgr_image, (224, 224))
        cv2.imwrite(f'{data_path}/face_train/{img_path[-20:]}', bgr_image)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces.append(rgb_image)
        id = file2id[img_path[-19:-9]]
        ids.append(id)

faces = np.array(faces)
ids = np.array(ids)
np.save(f'../x_train', faces)

num_class = np.max(ids)
num_sample = ids.shape[0]
y = np.zeros((num_sample, num_class), dtype=np.int8)
for i in range(num_sample):
    id = ids[i]
    y[i, id - 1] = 1
np.save(f'../y_train', y)

# %%
val_png_list = glob.glob(f'{data_path}/png_val/*.png')
with open(data_path + '/val.txt') as f:
    contents = f.readlines()
val_id2file = defaultdict(list)
val_file2id = {}
for index, line in enumerate(contents):
    id_file_path = line.rstrip('\n').split(' ')
    id = int(id_file_path[0])
    val_id2file[id].extend(id_file_path[1:])
    for index1, file_path in enumerate(id_file_path[1:]):
        val_file2id[file_path[-10:]] = id
pickle.dump(val_file2id, open("../data/val_file2id.p", "wb"))

faces = []
val2face = defaultdict(list)
i = -1
for index, img_path in tqdm(enumerate(val_png_list)):
    bgr_image = cv2.imread(img_path)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    if not len(face_locations) == 0:
        crop = face_locations[0]
        y1, x1, y2, x2 = crop
        bgr_image = bgr_image[y1:y2, x2:x1, :]
        bgr_image = resizeAndPad(bgr_image, (224, 224))
        faces.append(bgr_image)
        i += 1
        val2face[img_path[-19:-9]].append(i)

pickle.dump(val2face, open("../data/val2face.p", "wb"))
faces = np.array(faces)
np.save(f'../data/x_val', faces)
num_sample = x_val.shape[0]
y_val = np.zeros((num_sample, 574), dtype=np.int8)

for file, faces in file2face.items():
    id_true = file2id[file]
    for i in faces:
        y_val[i][id_true - 1] = 1
np.save('../data/y_val', y_val)
