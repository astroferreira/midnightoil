from tensorflow import keras
from midnightoil.io.images import plot_to_image
from midnightoil.utils.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt

from datetime import datetime

import tensorflow as tf
import numpy as np

logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(logdir)

class ConfusionMatrixCallback(keras.callbacks.Callback):
    
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        

    def on_epoch_end(self, epoch, logs={}):

        if epoch % 10 == 0:

            preds = []
            trues = []
            imgs = []
            for i, ex in enumerate(self.test_dataset.take(40)):
                X = ex[0]
                y = ex[1]
                
                imgs.append(X.numpy())
                preds.append(self.model.predict(X))
                trues.append(y)
        
            self.preds = np.array(np.concatenate(preds))
            self.trues = np.array(np.concatenate(trues))
            self.imgs = np.array(np.concatenate(imgs))

            self.bin_labels = np.argmax(self.trues, axis=1)
            self.bin_preds = np.argmax(self.preds, axis=1)
            
            self.cm = tf.math.confusion_matrix(np.argmax(self.trues, axis=1), np.argmax(self.preds, axis=1)).numpy()
            self.cmn = self.cm / self.cm.sum(axis=1, keepdims=True)

            labels = ['Ctrl', 'PMs']
            
            fig = plot_confusion_matrix(self.cm, class_names=labels)
            cm_image = plot_to_image(fig)

            true_positives = self.imgs[(self.bin_preds == 1) & (self.bin_labels == 1)]
            #true_negatives = self.imgs[np.where((self.bin_preds == 0) & (self.bin_labels == 0))[0]]
            #false_positives = np.where((self.bin_preds == 1) & (self.bin_labels == 0))[0]
            #false_negatives = np.where((self.bin_preds == 0) & (self.bin_labels == 1))[0]
            #print(self.imgs.shape)
            #print(true_positives.shape)
            #print(false_positives.shape)

            tp_fig = image_grid(true_positives)
            #tn_fig = image_grid(true_negatives)
            #fp_fig = image_grid(self.imgs[false_positives])
            #fn_fig = image_grid(self.imgs[false_negatives])

            
            
            
            

            with file_writer_cm.as_default():
                tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)
                tp_image = plot_to_image(tp_fig)
                tf.summary.image("true_positive_mosaic", tp_image, step=epoch)
                #tn_image = plot_to_image(tn_fig)
                #tf.summary.image("true_negative_mosaic", tn_image, step=epoch)
                #fp_image = plot_to_image(fp_fig)
                #tf.summary.image("false_positive_mosaic", fp_image, step=epoch)
                #fn_image = plot_to_image(fn_fig)
                #tf.summary.image("false_negative_mosaic", fn_image, step=epoch)

    
    
    #def log_cm(self):



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