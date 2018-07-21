# %%
import glob
import os
from collections import defaultdict

import face_recognition
import numpy as np
from tqdm import tqdm

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
# features
try:
    id_feature = np.load('id_feature.npy')
except:
    id_feature = defaultdict(list)
    for index, img_path in tqdm(enumerate(png_list)):
        image = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings != []:
            id = file2id[img_path[-19:-9]]
            id_feature[id].append(face_encodings[0])
    np.save('../data/id_feature', id_feature)

# %% test
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


def predict(face_encodings):
    y_pred = np.zeros(574)
    for id, features in id_feature.items():
        dist = []
        for i in range(len(features)):
            dist.append(np.linalg.norm(face_encodings - features[i]))
        dist_ave = np.average(dist)
        y_pred[id - 1] = 1 - dist_ave
    return y_pred


try:
    val_feature = np.load('../data/val_feature.npy')
except:
    val_feature = defaultdict(list)
    for index, img_path in tqdm(enumerate(val_png_list)):
        image = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings != []:
            id = img_path[-19:-9]
            val_feature[id].append(face_encodings[0])
    np.save('../data/val_feature', id_feature)

count = 0
win = 0
for mp4_path, id in tqdm(val_file2id.items()):
    count += 1
    pred_ids = []
    for val_feature in enumerate(val_feature[mp4_path[-10:]]):
        y_pred = predict(val_feature)
        pred_ids.append(y_pred)
    if pred_ids != []:
        pred_ids = np.average(np.array(pred_ids), axis=0)
        pred_id = np.argmax(pred_ids) + 1
        if pred_id == id:
            win += 1
            print(win / count)

for index, img_path in tqdm(enumerate(val_png_list)):
    image = face_recognition.load_image_file(img_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings != []:
        id = val_file2id[img_path[-19:-9]]
        val_feature[id].append(face_encodings[0])
