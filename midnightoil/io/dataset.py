import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math

from .recordshandler import construct_feature_description, parse
from ..augmentation import flip, rotate, oclusion

import glob

RNG = tf.random.Generator.from_seed(1331)

def load_dataset(path, columns=['y'], training=False, shuffle=True, batch_size=1, buffer_size=18000, augmentations=[flip, rotate, oclusion]):


    files = glob.glob(path)
    dataset = tf.data.TFRecordDataset(files)

    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True)
    dataset = dataset.map(map_function)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    if shuffle:
        dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)
    
    if training:
        dataset = dataset.repeat(400)

    for f in augmentations:
            dataset = dataset.map(lambda x, y: tf.cond(RNG.uniform((1,), 0, 1) > 0.8, lambda: f(x, y), lambda: (x, y)))


    dataset.prefetch(buffer_size=64).cache(filename='cached.cc')
    
    return dataset

def unravel_dataset(dataset, model, batch_size, ncolumns=3):

    batches = sum(1 for x in dataset)
    total = batches * batch_size

    X = np.zeros((total, 128, 128, 1), dtype=np.float32)
    y = np.zeros((total, ncolumns),  dtype=np.float32)     
    rootnames = []

    for batch, (xVal, yVal) in enumerate(dataset):
        """%%
            The first batch is logged into tensorboard for visual inspection
        """
        X[batch * batch_size : (batch+1) * batch_size] = xVal.numpy()
        y[batch * batch_size : (batch+1) * batch_size] = yVal.numpy()

        #rootnames.append(rootname.numpy()[0].decode())
        
    return X, y#, rootnames


