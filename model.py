from __future__ import absolute_import, division, print_function

import keras
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tqdm import tqdm


def model_config():
    model_config = {
        "Xception": [32, Xception],
        "InceptionResNetV2": [424, InceptionResNetV2],
        "ResNet50": [32, ResNet50],
        "InceptionV3": [32, InceptionV3],
        "DenseNet201": [24, DenseNet201],
        "DenseNet169": [32, DenseNet169],
        "DenseNet121": [32, DenseNet121],
    }
    input_shape = (224, 224, 3)
    fc, pred, layer_names = [512], [574], 'face'
    return model_config, fc, pred, layer_names, input_shape


def load_data():
    p = np.load('../data/p.npy')
    num = p.shape[0]
    train = p[:int(num * 0.90)]
    val = p[int(num * 0.90):]
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    return X[train], X[val], y[train], y[val]


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
                     metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    fc_model.fit(
        f_train,
        y_train_fc,
        validation_data=(f_val, y_val_fc),
        batch_size=128,
        epochs=10000,
        callbacks=[checkpointer, early_stopping])


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
