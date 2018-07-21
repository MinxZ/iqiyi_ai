# %%
import glob
import os
from collections import defaultdict

import face_recognition
import numpy as np
from tqdm import tqdm

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
# features
id_feature = defaultdict(list)
for id in tqdm(range(1, 10)):
    png_list = []
    mp4_list = id2file[id]
    for i, mp4_path in enumerate(mp4_list):
        png_list.extend(glob.glob(f'{data_path}/png_train/{mp4_path}*.png'))
    for index, img_path in enumerate(png_list):
        image = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(image)
        if face_encodings != []:
            id_feature[id].append(face_encodings[0])

for id in range(1, 10):
    print(len(id_feature[id]))


# %%
# test
dist4 = []
dist9 = []
id = 4
for i in range(len(id_feature[id])):
    dist4.append(np.linalg.norm(face_encodings - id_feature[id][i]))
id = 9
for i in range(len(id_feature[id])):
    dist9.append(np.linalg.norm(face_encodings - id_feature[id][i]))
dist4
dist9


# %%
for index1, img_path in tqdm(enumerate(png_list[:40])):
    bgr_image = cv2.imread(img_path)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    # cnn_face_locations = face_recognition.face_locations(image, model="cnn")

    if len(face_locations) == 0:
        print('None')
        # cv2.imwrite(f'{img_path[-20:]}', bgr_image)
    else:
        # for index, crop in enumerate(cnn_face_locations):
        #     y1, x1, y2, x2 = crop
        #     cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for index, crop in enumerate(face_locations):
            y1, x1, y2, x2 = crop
            bgr_image = bgr_image[y1:y2, x2:x1, :]
            # cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(f'{img_path[-20:]}', bgr_image)

# %%
bgr_image.shape
face_locations
face_encoding[0].shape
