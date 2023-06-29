from tensorflow import keras
from midnightoil.io.images import plot_to_image
from midnightoil.utils.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt

from datetime import datetime

import tensorflow as tf
import numpy as np

logdir = "logs/plots/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer_cm = tf.summary.create_file_writer(logdir)

class RegressionCallback(keras.callbacks.Callback):
    
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        

    def on_epoch_end(self, epoch, logs={}):

        #if epoch % 10 == 0:

        preds = []
        trues = []
        
        for i, ex in enumerate(self.test_dataset):
            X = ex[0]
            y = ex[1]
            
            preds.append(self.model.predict(X))
            trues.append(y)
    
        self.preds = np.array(np.concatenate(preds))
        self.trues = np.array(np.concatenate(trues))
        
        fig = plt.figure(dpi=100)
        plt.scatter(trues, preds)
        plt.xlabel('True Mass Ratio')
        plt.ylabel('Predicted Mass Ratio')
        plt.savefig('reg.png')
        image = plot_to_image(fig)

        
        with file_writer_cm.as_default():
            tf.summary.image("epoch_massratio", image, step=epoch)
            
    
    
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