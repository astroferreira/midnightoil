from tensorflow import keras
from midnightoil.io.images import plot_to_image
from midnightoil.utils.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt

from datetime import datetime

import tensorflow as tf
import numpy as np

logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(logdir)

class TFDataDebugger(keras.callbacks.Callback):
    
    def __init__(self, planner):
        self.model = planner.model
        self.dataset = planner.training_dataset
        
    def on_epoch_end(self, epoch, logs={}):

        #if epoch % 10 == 0:

        X = None
        for i, ex in enumerate(self.dataset.take(1)):
            X = ex[0]
            
        
        mosaic = batch_to_mosaic(X)
        fig = plot_mosaic(mosaic)
        cm_image = plot_to_image(fig)

        with file_writer_cm.as_default():
            tf.summary.image("train batch", cm_image, step=epoch)
                

def batch_to_mosaic(X):

    w=4
    h=4
    size = 128
    mat = np.zeros((128*w, 128*h))

    if len(X.shape) < 4:
        return mat
    
    for i in range(w):
        for j in range(h):
            idx = (i * w) + j; 
            min_h, max_h = int(j*size), int((j+1)*size)
            min_w, max_w = int(i*size), int((i+1)*size)

            mat[min_h:max_h, min_w:max_w] = X[idx][:,:,0]
    
    
    return mat

def plot_mosaic(mosaic):

    f, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set(xticks=[], yticks=[])
    ax.grid('off')
    ax.imshow(np.log10(mosaic), cmap='gist_gray_r')

    return f

def image_grid(imgs):

   
  """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
  # Create a figure to contain the plot.
  figure = plt.figure(figsize=(10,10))
  if len(imgs) > 25:
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.log10(imgs[i]), cmap='gist_gray_r')


  return figure