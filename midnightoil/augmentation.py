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
    def roclusion(): return random_cutout(x)
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        roclusion,
        rx)
    #x = random_cutout(x)
    return x, y

@tf.function
def rotate(x: tf.Tensor, y: tf.Tensor, ps:tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    def rrot(): return random_rotation(x)
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        rrot,
        rx)
    return x, y

def shear(x: tf.Tensor, y: tf.Tensor, ps:tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    def rshear(): return random_shear(x)
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        rshear,
        rx)
    return x, y

@tf.function
def shift(x: tf.Tensor, y: tf.Tensor,  ps: tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    dx = RNG.uniform([], -5, 5)
    dy = RNG.uniform([], -5, 5)
    def rshift(): return tfa.image.translate(x, [dx, dy], interpolation='BILINEAR', fill_mode='reflect')
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        rshift,
        rx)
    
    del dx
    del dy
    return x, y

@tf.function
def flip(x: tf.Tensor, y: tf.Tensor,  ps: tf.dtypes.float32) -> Tuple[tf.Tensor, tf.Tensor]:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    def rflip(): return tf.image.random_flip_up_down(tf.image.random_flip_left_right(x))
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        rflip,
        rx)
    
    return x, y

random_zoom_layer = tf.keras.layers.RandomZoom([-0.3, 0.0])

@tf.function
def zoom(x, y, ps):
    def rzoom(): return random_zoom_layer(x)
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        rzoom,
        rx)
    return tf.cast(x, dtype=tf.float32), y

@tf.function
def rotate90(x: tf.Tensor, y: tf.Tensor, ps: tf.dtypes.float32)-> Tuple[tf.Tensor, tf.Tensor]:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    def r90(): return  tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    def rx(): return x
    x = tf.cond(
        tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32, seed=2), ps), 
        r90,
        rx)
    return x, y

def augment(x, y):
    
    x, y = flip(x, y)
    #x, y = rotate90(x, y)
    x, y = rotate(x, y)
    x, y = shift(x, y)

            
    return x, y