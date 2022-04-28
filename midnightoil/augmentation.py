import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math
from typing import Tuple

RNG = tf.random.Generator.from_seed(1331)

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

def color(x, y):
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_brightness(x, 0.1)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

def oclusion(x, y, size=32):
    x = tf.expand_dims(x, axis=0)
    return tf.squeeze(tfa.image.random_cutout(x, (size, size), constant_values=0)), y

def central_crop(x, y):
    x = tf.image.central_crop(x, central_fraction=RNG.uniform((1,), 0.5, 0.95)[0])
    x = tf.cast(tf.image.resize(x, (128, 128)), dtype=tf.float64)
    return x, y


def shear_x(x, y):
    x = tfa.image.shear_x(x, level=RNG.uniform([], -0.2, 0.2), replace=0)
    return x, y

def shear_y(x, y):
    x = tfa.image.shear_y(x, level=RNG.uniform([], -0.2, 0.2), replace=0)
    return x, y

def small_rot(x, y):
    rot = np.random.uniform(0, 30)
    image = tfa.image.transform_ops.rotate(x, np.radians(rot), fill_mode='wrap', interpolation='BILINEAR')
    lrr_width, lrr_height = _largest_rotated_rect(128, 128, math.radians(10))
    resized_image = tf.image.central_crop(image, float(lrr_height)/128)    
    image = tf.image.resize(resized_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR)
    return tf.cast(image, dtype=tf.float64), y

def rotate90(x, y):
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)), y


def rotate(x, y):
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.5, 0.5))(x)
    x = tf.cast(x, dtype=tf.float64)
    return x, y
def zoom(x, y):
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.6, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(128, 128))
        # Return a random crop
        return tf.cast(crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)], dtype=tf.float64)


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x)), y

def _augment_image_fn():
    """RandAugment image for training."""

    def augment(image):
        import tensorflow_addons as tfa
        
        image = tfa.image.transform_ops.rotate(image, np.radians(10), interpolation='BILINEAR')
        lrr_width, lrr_height = _largest_rotated_rect(128, 128, math.radians(10))
        resized_image = tf.image.central_crop(image, float(lrr_height)/128)    
        image = tf.image.resize(resized_image, [128, 128], method=tf.image.ResizeMethod.BILINEAR)
        return image

    datasets = {
        "pmsf": augment,
    }

    return augment


def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )
