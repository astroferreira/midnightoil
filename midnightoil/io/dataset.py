import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math

from .recordshandler import construct_feature_description, parse
from ..augmentation import flip, rotate, rotate90, oclusion, color, central_crop, shear_x, shear_y,small_rot

import glob

RNG = tf.random.Generator.from_seed(1331)

augmentations_fns = {
        'flip': flip,
        'rotate90' : rotate90,
        'rotate' : rotate,
        'shear_x' : shear_x,
        'shear_y' : shear_y,
        'oclusion' : oclusion
}

def load_dataset(path, epochs, columns=['y'],
                 training=False, batch_size=128, 
                 buffer_size=18000, augmentations=None,
                 with_rootnames=False):
    
    files = sorted(glob.glob(path))[0]
    dataset = tf.data.TFRecordDataset(files)
    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True, with_rootnames=with_rootnames)

    files = glob.glob(path)
    files.sort(key=natural_keys)
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(map_function)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    if training:
        
        dataset = dataset.shuffle(batch_size//2, reshuffle_each_iteration=True)
        for key in augmentations:
            
            fn = augmentations_fns[key]
            ps = augmentations[key]

            dataset = dataset.map(lambda x, y: tf.cond(tf.less(RNG.uniform((1,), 0, 1), ps), lambda: fn(x, y), lambda: (x, y)),
                                  num_parallel_calls=tf.data.AUTOTUNE)
    
        dataset.cache()
        
    
    #dataset = dataset.map(single_band,
    #                      num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.repeat(epochs)

    dataset = dataset.prefetch(buffer_size=8)
    return dataset


def load_latest_weights(tPlanner, config, args, current_run, runPath):
    
    checkpoint_dir = f'{runPath}/checkpoints/'

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        last_loss, epoch = latest.split('/')[-1].split('_')
        epoch = int(epoch.split('.')[0])

    print(f'Loading weights from {latest}')
    tPlanner.model.load_weights(latest).expect_partial()
    return tPlanner

def single_band(x, y):
    return x[:,:,:, 0:1], y

def squeeze(x, y):
    return tf.squeeze(x), y

def unravel_dataset(tPlanner, batch_size=256):

    #imgs = []
    preds = []
    trues = []
    #rootnames = []
    tPlanner.loadData(training=False, batchSize=batch_size, with_rootnames=False)
    for i, ex in enumerate(tPlanner.test_dataset):
        print(i*batch_size)
        X = ex[0]
        y = ex[1]
        #r = ex[2]
        preds.append(tPlanner.model.predict(X))
        trues.append(y)
        #rootnames.append(['1'])
        #imgs.append(X.numpy())
    
    px = np.array(np.concatenate(preds))
    y = np.array(np.concatenate(trues))
    #rs = np.array(np.concatenate(rootnames)).astype(str)
    #X = np.array(np.concatenate(imgs))
        
    return y, px#,rs

import re
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    c = re.split('_', text)[-1].split('-')[0]
    return atoi(c)

def atoi(text):
    return int(text) if text.isdigit() else text
