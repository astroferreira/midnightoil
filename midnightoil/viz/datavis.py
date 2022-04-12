import matplotlib.pyplot as plt
import numpy as np


def batch_mosaic(X):

    f, axs = plt.subplots(5, 5, figsize=(10.5, 10.7))


    for im, ax in zip(X, axs.flat):

        norm = np.log10(10**im.numpy().copy())


        for i in range(3):
            norm[:,:,i] = (norm[:,:, i] - norm.min()) / (norm.max() - norm.min())

        ax.imshow(norm)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(wspace=0, hspace=0)

    

