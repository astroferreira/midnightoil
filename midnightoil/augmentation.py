import tensorflow as tf 
import keras_cv
import tensorflow_addons as tfa
import numpy as np
import math
from typing import Tuple

RNG = tf.random.Generator.from_seed(1331)

random_cutout = keras_cv.layers.preprocessing.RandomCutout(0.25, 0.25)
def oclusion(x: tf.Tensor, y: tf.Tensor, size=16) -> Tuple[tf.Tensor, tf.Tensor]:
    x = random_cutout(x)
    return x, y


def rotate(x, y):
    rot = np.random.uniform(-10, 10)
    x = tfa.image.transform_ops.rotate(x, np.radians(rot), fill_mode='reflect', interpolation='BILINEAR')
    #lrr_width, lrr_height = _largest_rotated_rect(64, 64, math.radians(10))
    #resized_image = tf.image.central_crop(image, float(lrr_height)/64)    
    #image = tf.image.resize(resized_image, [64, 64], method=tf.image.ResizeMethod.BILINEAR)
    del rot
    return tf.cast(x, dtype=tf.float32), y

random_shear = keras_cv.layers.RandomShear(0.05, 0.05, fill_mode='reflect')
def shear(x, y):
    return random_shear(x), y

mix_up = keras_cv.layers.MixUp()

def MixUp(x, y):
    output = mix_up({'images': x, 'labels': y})
    return output['images'], output['labels']

def shear_x(x, y):
    level = RNG.uniform([], -0.1, 0.1)
    x = tfa.image.transform_ops.transform(x, [1.0, level, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], interpolation='BILINEAR', fill_mode='reflect')
    del level
    return x, y

def shear_y(x, y):
    level = RNG.uniform([], -0.1, 0.1)
    x = tfa.image.transform_ops.transform(x, [1.0, 0.0, 0.0, level, 1.0, 0.0, 0.0, 0.0], interpolation='BILINEAR', fill_mode='reflect')
    del level
    return x, y

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