from __future__ import absolute_import, division, print_function

import argparse
import multiprocessing as mp
import random

import numpy as np
import tensorflow as tf
from keras import backend
from keras.applications import *
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from model import *

model_name, optimizer, lr = "ResNet50", "Adam", 1e-5


def run(model_name, optimizer, lr):
    if model_name == "ResNet50":
        print('\n  For Resnet')
        from keras.applications.imagenet_utils import preprocess_input
    elif model_name[:-3] == "DenseNet":
        print('\n  For DenseNet')
        from keras.applications.densenet import preprocess_input
    else:
        print('\n  For model = tf')
        from keras.applications.inception_v3 import preprocess_input

    # Load datasets
    x_train, x_val, y_train, y_val = load_data()
    epochs = 10000

    # Loading model
    print('\n  Loading model')
    model_config, fc, pred, layer_names, input_shape = model_config()
    MODEL = model_config[model_name][1]
    batch_size = model_config[model_name][0]

    try:
        model = load_model(
            f'../models/{model_name}_{len(fc)}_fc_2.h5')
        checkpointer = ModelCheckpoint(
            filepath=f'../models/{model_name}_{len(fc)}_fc_4.h5', verbose=0, save_best_only=True)
        print('\n  Ready to fine tune.')
    except:
        model, checkpointer = build_model(input_shape, x_train, y_train, x_val, y_val, batch_size,
                                          fc, pred, layer_names, model_name, preprocess_input)
    # callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=2, mode='auto')
    reduce_lr = ReduceLROnPlateau(
        factor=np.sqrt(0.1), patience=2, verbose=2)

    if optimizer == 'SGD':
        opt = SGD(lr=lr, momentum=0.9, nesterov=True)
    elif optimizer == 'Adam':
        opt = Adam(lr=lr)

    # Compile model
    print(f"\n  {model_name}: Optimizer=" +
          optimizer + " lr=" + str(lr) + " \n")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['categorical_accuracy'])

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    checkpointer = ModelCheckpoint(
        filepath=f'../models/{model_name}_{len(fc)}_fc.h5', verbose=0, save_best_only=True)
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size,
        validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
        validation_steps=len(x_val) / batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpointer, reduce_lr],
        max_queue_size=10,
        workers=4,
        use_multiprocessing=False)

    quit()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="DenseNet169", type=str)
    parser.add_argument(
        "--optimizer", help="which optimizer to use", default="Adam", type=str)
    parser.add_argument(
        "--lr", help="learning rate", default=2e-5, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.optimizer, args.lr)
