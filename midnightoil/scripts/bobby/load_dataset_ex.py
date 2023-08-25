import os
import math
import glob
import numpy as np

import tensorflow as tf 
import tensorflow_addons as tfa

from .recordshandler import construct_feature_description, parse
from .augmentation import augment, zoom, shear


RNG = tf.random.Generator.from_seed(1331)

def build_TFRecordDataset(path, columns, with_rootnames=False, model_cfg=None):
    """
        Loads the tfrecord files in parallel and build the header
    """
    
    files = sorted(glob.glob(path))
    ds_files = tf.data.Dataset.from_tensor_slices(files)
    dataset = ds_files.shuffle(len(files)).interleave(lambda x: tf.data.TFRecordDataset(x,  num_parallel_reads=tf.data.AUTOTUNE))

    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True, with_rootnames=with_rootnames, model_cfg=model_cfg)
    
    dataset = dataset.map(map_function)

    return dataset

def load_dataset(path, epochs, columns=['y'],
                 training=False, batch_size=128, 
                 buffer_size=18000, with_rootnames=False,
                 model_cfg=None):
    """
        Loads the tfrecords into a tf.TFRecordDataset.
        
        If the training set is loaded this way it shuffles it, 
        applies the augmentations for each image individually and then
        generates batches of it. These steps are running in parallel with
        the help of the map function.
    """

    dataset = build_TFRecordDataset(path, columns, with_rootnames=with_rootnames, model_cfg=model_cfg)
    
    if training:  
        dataset = dataset.shuffle(20000)
        dataset = dataset.repeat(epochs)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(shear, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(zoom, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(8)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
    return dataset