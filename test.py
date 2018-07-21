from __future__ import absolute_import, division, print_function

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
from collections import defaultdict

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

model_name, optimizer, lr = "ResNet50", "Adam", 2e-5


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
    file2id = pickle.load(open("../data/val_file2id.p", "rb"))
    file2face = pickle.load(open("../data/val2face.p", "rb"))
    x_val = np.load('../data/x_val.npy')

    # Loading model
    print('\n  Loading model')
    model_config, fc, pred, layer_names, input_shape = model_config()
    batch_size = model_config[model_name][0]
    model = load_model(f'../models/{model_name}_{len(fc)}_fc.h5')
    y_pred = model.predict(x_val, batch_size, verbose=1)
    count = 0
    win = 0
    for file, faces in file2face.items():
        count += 1
        id_true = file2id[file]
        id_pred = np.argmax(np.average(y_pred[faces], axis=0)) + 1
        if id_pred == id_true:
            win += 1
    print(win / count)


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
