import tensorflow as tf 
import tensorflow_addons as tfa
import numpy as np
import math
import os

SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')

from .recordshandler import construct_feature_description, parse
from ..augmentation import augment, zoom, shear, flip, rotate, rotate90, shift, oclusion

import glob

RNG = tf.random.Generator.from_seed(1331)


augmentation_map = {
    'zoom' : zoom,
    'shear': shear,
    'flip': flip,
    'rotate': rotate,
    'rotate90': rotate90,
    'shift' : shift
}

def build_TFRecordDataset(path, columns='y', training=False, with_rootnames=False, model_cfg=None, mock_survey=False):
    """
        Loads the tfrecord files in parallel and build the header
    """
    
    files = sorted(glob.glob(path))
    if training:
        ds_files = tf.data.Dataset.from_tensor_slices(files)
        dataset = ds_files.interleave(lambda x: tf.data.TFRecordDataset(x,  num_parallel_reads=tf.data.AUTOTUNE))
    else:
        dataset = tf.data.TFRecordDataset(files)

    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=True, with_rootnames=with_rootnames, model_cfg=model_cfg, mock_survey=mock_survey)
    
    dataset = dataset.map(map_function)

    return dataset



def load_dataset(path, epochs, columns='y',
                 training=False, batch_size=128, 
                 buffer_size=18000, with_rootnames=False,
                 model_cfg=None, mock_survey=False, augmentations=None):
    """
        Loads the tfrecords into a tf.TFRecordDataset.
        
        If the training set is loaded this way it shuffles it, 
        applies the augmentations for each image individually and then
        generates batches of it. These steps are running in parallel with
        the help of the map function.
    """
    print(path)  
    dataset = build_TFRecordDataset(path, columns, training=training,
                                    with_rootnames=with_rootnames,
                                    model_cfg=model_cfg, mock_survey=mock_survey)
    
    if training:  
        dataset = dataset.shuffle(20000)
        dataset = dataset.repeat(epochs)

        if augmentations is not None:
            for layer in range(augmentations['num_layers']):    
                dataset = dataset.map(lambda x, y: flip(x, y, augmentations['flip']), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(lambda x, y: rotate90(x, y, augmentations['rotate90']), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(lambda x, y: rotate(x, y, augmentations['rotate']), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(lambda x, y: shift(x, y, augmentations['shift']), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(lambda x, y: shear(x, y, augmentations['shear']), num_parallel_calls=tf.data.AUTOTUNE)
                dataset = dataset.map(lambda x, y: zoom(x, y, augmentations['zoom']), num_parallel_calls=tf.data.AUTOTUNE)
            
            dataset = dataset.map(lambda x, y: oclusion(x, y, augmentations['oclusion']), num_parallel_calls=tf.data.AUTOTUNE)
            #dataset = dataset.map(shear, num_parallel_calls=tf.data.AUTOTUNE)
        #dataset = dataset.map(zoom, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(batch_size)
        
    return dataset


def load_latest_weights(tPlanner, config, args, current_run, runPath):
    
    checkpoint_dir = f'{runPath}/checkpoints/'

    if args.eval_epoch is not None:
        
        if int(args.eval_epoch) < 100:
            checkpoint_dir = checkpoint_dir + f'0{args.eval_epoch}.ckpt'
        else:
            checkpoint_dir = checkpoint_dir + f'{args.eval_epoch}.ckpt'

        tPlanner.model.load_weights(checkpoint_dir).expect_partial()
    else:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is not None:
            epoch = latest.split('/')[-1]#.split('_')
            epoch = int(epoch.split('.')[0])

        print(f'Loading weights from {latest}')
        tPlanner.model.load_weights(latest).expect_partial()
    
    return tPlanner

def list_all_checkpoints(runPath):
    checkpoint_dir = f'{runPath}/checkpoints/*.index'
    ckpts = [f.split('.index')[0] for f in sorted(glob.glob(checkpoint_dir))]
    print(ckpts)
    return ckpts
    


def single_band(x, y):
    return x[:,:,:, 0:1], y

def squeeze(x, y):
    return tf.squeeze(x), y

def unravel_dataset(tPlanner, batch_size=512):

    #imgs = []
    trues = []
    rootnames = []
    preds = []
    tPlanner.loadData(training=False, batchSize=batch_size, with_rootnames=True)
    for i, ex in enumerate(tPlanner.test_dataset):
        print(100*((i/(744800//batch_size))))
        X = ex[0]
        y = ex[1]
        r = ex[2]
        preds.append(tPlanner.model.predict(X))
        trues.append(y)
        rootnames.append(r)
        #print(y)
        #print(preds[-1])
        cm = tf.math.confusion_matrix(np.argmax(np.array(np.concatenate(trues)), axis=1), np.argmax(np.array(np.concatenate(preds)), axis=1)).numpy()
        print(cm / cm.sum(axis=1, keepdims=True))
        #imgs.append(X.numpy())



    y = np.array(np.concatenate(trues))
    rs = np.array(np.concatenate(rootnames)).astype(str)
    ps = np.array(np.concatenate(preds))
    #X = np.array(np.concatenate(imgs))
    return y, ps, rs

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


import os
class DataAugmentationDino:
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_image_size=[128, 128],
        local_image_size=[64, 64],
        mean=[0.485, 0.456, 0.406],
        std_dev=[0.229, 0.224, 0.225],
    ):
        self.mean = mean
        self.std_dev = std_dev
        self.local_image_size = local_image_size
        self.global_image_size = global_image_size
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale

        self.flip_aug = tf.keras.Sequential(
            [tf.keras.layers.RandomFlip(mode="horizontal")]
        )

    def _standardize_normalize(self, image):
        image = image / 255.0
        image -= self.mean
        image /= self.std_dev
        image = tf.cast(image, tf.float32)
        return image

    def _color_jitter(image):
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.0, upper=0.4)
        image = tf.image.random_saturation(image, lower=0.0, upper=0.2)
        image = tf.image.random_hue(image, max_delta=0.1)
        return image

    def _crop_resize(self, image, mode="global"):
        scalee = self.global_crops_scale if mode == "global" else self.local_crops_scale
        final_size = (
            self.global_image_size if mode == "global" else self.local_image_size
        )
        height, width, channels = tf.shape(image)
        scaling_hw = tf.cast(tf.stack([height, width], axis=0), tf.float32)
        scale = tf.multiply(scalee, scaling_hw)
        scale = (
            tf.cast(scale[0].numpy(), tf.int32),
            tf.cast(scale[1].numpy(), tf.int32),
            channels,
        )
        image = tf.image.random_crop(value=image, size=scale)
        image = tf.image.resize(image, final_size, method="bicubic")
        return image

    def _apply_aug(self, image, mode="global"):
        image = self.flip_aug(image)
        image = self._crop_resize(image, mode)
        #image = self._standardize_normalize(image)

        return image

    def __call__(self, image, mode="global"):
        if mode == "global":
            return tf.stack([self._apply_aug(image, mode=mode), self._apply_aug(image)])# = []
        else:
            return tf.stack([self._apply_aug(image, mode=mode) for _ in range(self.local_crops_number)])#crops.append(self._apply_aug(image))

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        mode,
        batch_size,
        dataset_path,
        local_image_size,
        global_image_size,
        shuffle=True,
    ):
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.dataset = os.listdir(dataset_path)
        self.local_image_size = local_image_size
        self.global_image_size = global_image_size
        self.on_epoch_end()
    
    import numpy as np
    def _load_image(self, path, data_augmentation, mode):
        image = np.load(path)   
        image = data_augmentation(image, mode=mode) if self.mode == "train" else image
        return image

    def on_epoch_end(self):
        self.index = tf.range(len(self.dataset))
        if self.shuffle == True:
            tf.random.shuffle(self.index)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.index[idx * self.batch_size : (idx + 1) * self.batch_size]
        datset_keys = [self.dataset[k] for k in indexes]
        (global_images, local_images) = self.__data_generation(datset_keys)
        return global_images, local_images

    def __data_generation(self, index):
        batch_global, batch_local = [], []
        dino = DataAugmentationDino((0.4, 1.0), (0.05, 0.4), 8)
        batch_global = tf.concat([self._load_image(os.path.join(self.dataset_path, i), dino, mode='global') for idx, i in enumerate(index)], 0)
        batch_local = tf.concat([self._load_image(os.path.join(self.dataset_path, i), dino, mode='local') for idx, i in enumerate(index)], 0)
            #global_images, local_images = self._load_image(os.path.join(self.dataset_path, i), dino)
            #global_images = images[:2]
            # unable to stack varied size input in the dataset
            #local_images = images[2:]
            #batch_local.append(local_images)
            #batch_global.append(global_images)
        return batch_global, batch_local


