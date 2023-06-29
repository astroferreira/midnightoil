import io
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def mosaic(ds):

  for ex in ds.take(1):
    imgs = ex[0].numpy()

  f, axs = plt.subplots(10, 10)

  for ax, img in zip(axs.flat, imgs):
    ax.set(xticks=[], yticks=[])
    ax.imshow(np.log10(img), cmap='gist_gray')

  plt.subplots_adjust(hspace=0, wspace=0)
  plt.savefig('test.png')