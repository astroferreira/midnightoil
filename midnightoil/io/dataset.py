import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math

from .recordshandler import construct_feature_description, parse
from ..augmentation import flip, rotate, rotate90, oclusion, color, central_crop, shear_x, shear_y

import glob

RNG = tf.random.Generator.from_seed(1331)

def load_dataset(path, columns=['y'], epochs=400, 
                 training=False, batch_size=128, 
                 buffer_size=18000, augmentations=[rotate90, flip, rotate, central_crop, shear_x, shear_y, oclusion], 
                 probs=[0.5, 0.5, 0.5, 0.4, 0.0,  0.0, 0.8]):
    

    files = sorted(glob.glob(path))[0]
    print(files)
    dataset = tf.data.TFRecordDataset(files)
    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True)
    print(image_feature_description)
    files = sorted(glob.glob(path))
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(map_function)
    
    if training:
        dataset = dataset.shuffle(batch_size, reshuffle_each_iteration=True)

        for f, ps in zip(augmentations, probs):
            if ps == 0.0:
                continue
            
            dataset = dataset.map(lambda x, y: tf.cond(RNG.uniform((1,), 0, 1) <= ps, lambda: f(x, y), lambda: (x, y)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    
    
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat(epochs)
    else:
        dataset = dataset.batch(128, drop_remainder=True)

    #dataset = dataset.prefetch(buffer_size=64)#.cache(filename='cached.cc')

    return dataset

def squeeze(x, y):
    return tf.squeeze(x), y

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


