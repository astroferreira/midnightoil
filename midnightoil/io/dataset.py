import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math

from .recordshandler import construct_feature_description, parse
from ..augmentation import flip, rotate, rotate90, oclusion, color, central_crop, shear_x, shear_y,small_rot

import glob

RNG = tf.random.Generator.from_seed(1331)


augmentations_fns = {
        'flip': flip,
        'rotate90' : rotate90,
        'rotate' : rotate,
        'shear_x' : shear_x,
        'shear_y' : shear_y,
        'oclusion' : oclusion
}

def load_dataset(path, epochs, columns=['y'],
                 training=False, batch_size=128, 
                 buffer_size=18000, augmentations=None):
    
    files = sorted(glob.glob(path))[0]
    dataset = tf.data.TFRecordDataset(files)
    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True)

    files = glob.glob(path)
    files.sort(key=natural_keys)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(map_function)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    if training:
        

        for key in augmentations:
            
            fn = augmentations_fns[key]
            ps = augmentations[key]

            dataset = dataset.map(lambda x, y: tf.cond(tf.less(RNG.uniform((1,), 0, 1), ps), lambda: fn(x, y), lambda: (x, y)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    
        dataset.cache()
        
    
    #dataset = dataset.map(single_band,
    #                      num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.repeat(epochs)

    dataset = dataset.prefetch(buffer_size=8)
    return dataset


def single_band(x, y):
    return x[:,:,:, 0:1], y

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

import re
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    c = re.split('_', text)[-1].split('-')[0]
    return atoi(c)

def atoi(text):
    return int(text) if text.isdigit() else text
