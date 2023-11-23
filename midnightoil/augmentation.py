import tensorflow as tf 
import keras_cv
import tensorflow_addons as tfa
import numpy as np
import math
from typing import Tuple

RNG = tf.random.Generator.from_seed(1331)
random_shear = keras_cv.layers.RandomShear(0.05, 0.05, fill_mode='reflect')
random_cutout = keras_cv.layers.preprocessing.RandomCutout(0.25, 0.25)
random_rotation = tf.keras.layers.RandomRotation(factor=(-0.1, 0.1), fill_mode='reflect', interpolation='bilinear')

def oclusion(x: tf.Tensor, y: tf.Tensor, ps: tf.dtypes.float32, size=16) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.cond(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
            random_cutout(x),
            x
        )
    #x = random_cutout(x)
    return x, y

@tf.function
def rotate(x: tf.Tensor, y: tf.Tensor, ps:tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.cond(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
        random_rotation(x),
        x
    )
    return x, y

def shear(x: tf.Tensor, y: tf.Tensor, ps:tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    x = tf.cond(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
        random_shear(x),
        x
    )
    return x, y

@tf.function
def shift(x: tf.Tensor, y: tf.Tensor,  ps: tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    dx = RNG.uniform([], -5, 5)
    dy = RNG.uniform([], -5, 5)
    x = tf.cond(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
        tfa.image.translate(x, [dx, dy], interpolation='BILINEAR', fill_mode='reflect'),
        x
    )
    
    del dx
    del dy
    return x, y

@tf.function
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

@tf.function
def zoom(x, y, ps):
    x = tf.cond(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
        random_zoom_layer(x),
        x
    )
    return tf.cast(x, dtype=tf.float32), y

@tf.function
def rotate90(x: tf.Tensor, y: tf.Tensor, ps: tf.dtypes.float32)-> Tuple[tf.Tensor, tf.Tensor]:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.cond(
        tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2) > ps, 
        tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)),
        x
    )
    return x, y

def augment(x, y):
    
    x, y = flip(x, y)
    #x, y = rotate90(x, y)
    x, y = rotate(x, y)
    x, y = shift(x, y)

            
    return x, y