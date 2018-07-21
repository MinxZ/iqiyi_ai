from __future__ import absolute_import, division, print_function

import cv2 as cv2
import keras
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tqdm import tqdm


def load_model_config():
    model_config = {
        "ResNet50": [32, ResNet50],
        "DenseNet121": [36, DenseNet121],
        "DenseNet169": [32, DenseNet169],
        "DenseNet201": [24, DenseNet201],
        "Xception": [32, Xception],
        "InceptionV3": [32, InceptionV3],
        "InceptionResNetV2": [24, InceptionResNetV2],
    }
    input_shape = (224, 224, 3)
    fc, pred, layer_names = [512], [574], 'face'
    return model_config, fc, pred, layer_names, input_shape


def load_data():
    x_train = np.load('../data/x_train.npy')
    y_train = np.load('../data/y_train.npy')
    x_val = np.load('../data/x_val.npy')
    y_val = np.load('../data/y_val.npy')
    return x_train, x_val, y_train, y_val


def tri_fc(inputs, x, fc, pred, layer_names, activation_1='elu', activation_2='softmax'):
    num = len(fc)
    processed = {}
    for i in range(num):
        processed[i] = Dropout(0.5)(x)
        processed[i] = Dense(fc[i], activation=activation_1,
                             name=f'processed{i}_fc{i}')(processed[i])
        processed[i] = Dropout(0.5)(processed[i])
        processed[i] = Dense(pred[i], activation=activation_2,
                             name=layer_names[i])(processed[i])
    if num == 1:
        outputs = processed[0]
    else:
        outputs = [processed[i] for i in range(num)]
    model = Model(inputs, outputs)

    return model


def get_model(cnn_model, data, preprocess_input):
    input_shape = data.shape[1:]
    inputs = Input(shape=input_shape)
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    cnn_model_preprocess = Model(inputs, x)
    return cnn_model_preprocess


def fc_model_train(x_train_fc, y_train_fc, x_val_fc, y_val_fc, batch_size, cnn_model, fc, pred, layer_names, model_name, preprocess_input, loss='categorical_crossentropy', metrics=['categorical_accuracy']):
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    multi_fc_model_name = f'{len(fc)}_fc_{model_name}'

    cnn_model_preprocess = get_model(cnn_model, x_train_fc, preprocess_input)
    f_train = cnn_model_preprocess.predict(
        x_train_fc, batch_size=batch_size, verbose=1)
    np.save(f'../data/f_train_{model_name}', f_train)
    f_val = cnn_model_preprocess.predict(
        x_val_fc, batch_size=batch_size, verbose=1)
    np.save(f'../data/f_val_{model_name}', f_val)

    f_input_shape = f_train.shape[1:]
    f_inputs = Input(shape=f_input_shape)
    f_x = f_inputs

    print(f'Train for {len(fc)}_fc model')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/{multi_fc_model_name}.h5', verbose=0, save_best_only=True)
    fc_model = tri_fc(f_inputs, f_x, fc, pred, layer_names)
    fc_model.compile(loss='categorical_crossentropy', optimizer='adam',
                     metrics=['categorical_accuracy'])
    fc_model.fit(
        f_train,
        y_train_fc,
        validation_data=(f_val, y_val_fc),
        batch_size=128,
        epochs=10000,
        callbacks=[checkpointer, early_stopping])


def build_model(input_shape, x_train, y_train, x_val, y_val, batch_size,
                fc, pred, layer_names, model_name, preprocess_input):
    print('\n  Build model')
    name_model = f'../models/{model_name}_{len(fc)}_fc.h5'
    checkpointer = ModelCheckpoint(
        filepath=name_model, verbose=0, save_best_only=True)
    cnn_model = MODEL(
        include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')
    inputs = Input(shape=input_shape)
    x = cnn_model(inputs)
    model = tri_fc(inputs, x, fc, pred, layer_names)

    try:
        model.load_weights(
            f'../models/{len(fc)}_fc_{model_name}.h5', by_name=True)
        print('\n  Succeed on loading fc wight ')
    except:
        print('\n  Train fc')
        fc_model_train(x_train, y_train, x_val, y_val, batch_size,
                       cnn_model, fc, pred, layer_names, model_name, preprocess_input)
        model.load_weights(
            f'../models/{len(fc)}_fc_{model_name}.h5', by_name=True)
    model.save(name_model)
    return model, checkpointer


def fc_256(inputs, x, fc, start_num):
    num = len(fc)
    processed = {}
    for i in range(num):
        processed[i] = Dropout(0.5)(x)
        processed[i] = Dense(fc[i], activation='elu',
                             name=f'processed{i}_fc{i}')(processed[i])
    if num == 1:
        outputs = processed[0]
    else:
        outputs = [processed[i] for i in range[num]]
        outputs = concatenate([processed[i], processed_b, processed_c])
    model = Model(inputs, outputs)

    return model


def resizeAndPad(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA
    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    # if on Python 2, you might need to cast as a float: float(w)/h
    aspect = w / h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(
            int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(
            int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # color image but only one color provided
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img
