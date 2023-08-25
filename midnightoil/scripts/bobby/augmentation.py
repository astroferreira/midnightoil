import math
import keras_cv

import numpy as np
import tensorflow as tf 
import tensorflow_addons as tfa

from typing import Tuple

RNG = tf.random.Generator.from_seed(1331)

def rotate(x, y):
    rot = np.random.uniform(-10, 10)
    x = tfa.image.transform_ops.rotate(x, np.radians(rot), fill_mode='reflect', interpolation='BILINEAR')

    del rot
    return tf.cast(x, dtype=tf.float32), y

random_shear = keras_cv.layers.RandomShear(0.05, 0.05, fill_mode='reflect')
def shear(x, y):
    return random_shear(x), y

def shift(x, y):
    dx = RNG.uniform([], -5, 5)
    dy = RNG.uniform([], -5, 5)
    x = tfa.image.translate(x, [dx, dy], interpolation='BILINEAR', fill_mode='reflect')
    del dx
    del dy
    return x, y

def rotate90(x: tf.Tensor, y: tf.Tensor)-> Tuple[tf.Tensor, tf.Tensor]:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), y

def flip(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x, y

random_zoom_layer = tf.keras.layers.RandomZoom([-0.3, 0.0])
def zoom(x, y):
    x = random_zoom_layer(x)
    return tf.cast(x, dtype=tf.float32), y


def augment(x, y):
    
    x, y = flip(x, y)
    x, y = rotate90(x, y)
    x, y = rotate(x, y)
    x, y = shift(x, y)

            
    return x, y