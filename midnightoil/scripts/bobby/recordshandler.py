import math
import glob

import numpy as np
import pandas as pd
import tensorflow as tf 

from scipy.ndimage import zoom

def parse(image_feature_description, columns, with_labels=True, with_rootnames=False, model_cfg=None):

    def _parser(ep):

        """
            Parses an example from a TFRecord to the format we expect
        """

        example = tf.io.parse_single_example(ep, image_feature_description)
        image = tf.io.decode_raw(example['X'], out_type=np.float32)

        # input dimensions
        if model_cfg is None:
            input_shape = (128, 128, 1) 
        else:
            input_shape = (model_cfg['input_size'][0], model_cfg['input_size'][1], model_cfg['channels'])    
        
        image = tf.reshape(image, input_shape)
        labels = tf.one_hot(tf.cast(example['y'], dtype=tf.int64), depth=2)
                                
        if with_labels:
            if with_rootnames:
                return image, labels, example['DB_ID']
            else:
                return image, labels
        
        return image

    return _parser


def construct_feature_description(dataset):

    description = {
        'bytes' : tf.io.FixedLenFeature([], tf.string),
        'int64' : tf.io.FixedLenFeature([], tf.int64),
        'float' : tf.io.FixedLenFeature([], tf.float32)
    }

    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        break

    image_feature_description = {}

    features = list(example.features.feature)
    print(features)
    for feature in features:
        unserial_info = str(example.features.feature[feature]).split('_')[0]
        image_feature_description[feature] = description[unserial_info]

    return image_feature_description


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    value = float(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    value = int(value)
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _str_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    value = str.encode(str(value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))