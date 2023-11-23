
from midnightoil.models.loader import get_model
import yaml
import numpy as np
import tensorflow as tf
import glob 
from keras.models import Model
from keras.layers import Input
from keras import layers, models
from tfswin import SwinTransformer, SwinTransformerTiny224, preprocess_input

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

models_path = ['/home/ferreira/scratch/runs/1051_B0_STAGE1_0', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_0_202309041700',
               '/home/ferreira/scratch/runs/1051_B0_STAGE1_1',
               '/home/ferreira/scratch/runs/1051_B0_STAGE1_2', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_1_202309041721',
               '/home/ferreira/scratch/runs/1051_B0_STAGE1_3', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_2_202309041737',
               '/home/ferreira/scratch/runs/1052_B0_STAGE1_4', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_3_202309042315',
               '/home/ferreira/scratch/runs/1053_B0_STAGE1_5', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_4_202309050005',
               '/home/ferreira/scratch/runs/1054_B0_STAGE1_6', #'/home/ferreira/scratch/runs/1000_B0_STAGE1_5_202309050011',
               '/home/ferreira/scratch/runs/1055_B0_STAGE1_7',#'/home/ferreira/scratch/runs/1000_B0_STAGE1_6_202309050011',
               '/home/ferreira/scratch/runs/1056_B0_STAGE1_8',#'/home/ferreira/scratch/runs/1000_B0_STAGE1_7_202309050011',
               '/home/ferreira/scratch/runs/1057_B0_STAGE1_9',#'/home/ferreira/scratch/runs/1000_B0_STAGE1_8_202309050011', 
               '/home/ferreira/scratch/runs/1018_SwinV3_v2STAGE1_Ensemble_0',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_0_202309051301',
               '/home/ferreira/scratch/runs/1029_SwinV3_v2STAGE1_Ensemble_1',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_1_202309041155',
               '/home/ferreira/scratch/runs/1021_SwinV3_v2STAGE1_Ensemble_2',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_2_202309041156',
               '/home/ferreira/scratch/runs/1021_SwinV3_v2STAGE1_Ensemble_3',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_3_202309041157',
               '/home/ferreira/scratch/runs/1021_SwinV3_v2STAGE1_Ensemble_4',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_4_202309041203',
               '/home/ferreira/scratch/runs/1021_SwinV3_v2STAGE1_Ensemble_5',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_5_202309041253',
               '/home/ferreira/scratch/runs/1022_SwinV3_v2STAGE1_Ensemble_6',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_6_202309041317',
               '/home/ferreira/scratch/runs/1022_SwinV3_v2STAGE1_Ensemble_7',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_7_202309041317',
               '/home/ferreira/scratch/runs/1023_SwinV3_v2STAGE1_Ensemble_8',#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_8_202309041316',
               '/home/ferreira/scratch/runs/1024_SwinV3_v2STAGE1_Ensemble_9']#,#'/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_9_202309051745']
"""
models_path = ['/home/ferreira/scratch/runs/1000_B0_STAGE1_0_202309041700',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_9_202309050017',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_1_202309041721',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_2_202309041737',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_3_202309042315',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_4_202309050005',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_5_202309050011',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_6_202309050011',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_7_202309050011',
               '/home/ferreira/scratch/runs/1000_B0_STAGE1_8_202309050011', 
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_0_202309051301',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_1_202309041155',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_2_202309041156',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_3_202309041157',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_4_202309041203',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_5_202309041253',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_6_202309041317',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_7_202309041317',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_8_202309041316',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_9_202309051745']

models_path = [
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_0_202309051301',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_1_202309041155',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_2_202309041156',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_3_202309041157',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_4_202309041203',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_5_202309041253',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_6_202309041317',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_7_202309041317',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_8_202309041316',
               '/home/ferreira/scratch/runs/1000_SwinV3_STAGE1_Ensemble_9_202309051745']

"""
#models_path = ['/home/ferreira/scratch/runs/837_SwinV3_STAGE2_202308252303']

def parse(image_feature_description, columns='y', with_labels=True, with_rootnames=False, model_cfg=None, mock_survey=False):

    def _parser(ep):

        example = tf.io.parse_single_example(ep, image_feature_description)
        
        image = tf.io.decode_raw(example['X'], out_type=np.float32)

        if model_cfg is None:
            input_shape = (128, 128, 1)
        else:
            input_shape = (model_cfg['input_size'][0], model_cfg['input_size'][1], model_cfg['channels'])    
        
        image = tf.reshape(image, input_shape)
        
        image = tf.image.resize(image, [256, 256], method=tf.image.ResizeMethod.BICUBIC)
        image = tf.math.divide(
                            tf.subtract(
                                image, 
                                tf.reduce_min(image)
                            ), 
                            tf.subtract(
                                tf.reduce_max(image), 
                                tf.reduce_min(image)
                            )
                        )
      
        if with_labels:
            return image#, example['DB_ID']
        else:
            return image#, example['mock_camera'], example['mock_redshift']

    return _parser

import glob
import re 
import math
from pathlib import Path 
import yaml
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def build_TFRecordDataset(path, columns='y', training=False, with_rootnames=False, model_cfg=None, mock_survey=False):
    """
        Loads the tfrecord files in parallel and build the header
    """
    
    files = sorted(glob.glob(path))
    print(files)
    files = sorted(glob.glob(path), key=get_order)
    print(files)
    if training:
        ds_files = tf.data.Dataset.from_tensor_slices(files)
        dataset = ds_files.shuffle(len(files)).interleave(lambda x: tf.data.TFRecordDataset(x,  num_parallel_reads=tf.data.AUTOTUNE))
    else:
        dataset = tf.data.TFRecordDataset(files)

    image_feature_description = construct_feature_description(dataset)
    map_function = parse(image_feature_description, columns=columns, with_labels=False, with_rootnames=with_rootnames, model_cfg=model_cfg, mock_survey=mock_survey)
    
    dataset = dataset.map(map_function)

    return dataset

def construct_feature_description(dataset):

    description = {
        'bytes' : tf.io.FixedLenFeature([], tf.string),
        'int64' : tf.io.FixedLenFeature([], tf.int64),
        'float' : tf.io.FixedLenFeature([], tf.float32)
    }

    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        break

    image_feature_description = {}

    features = list(example.features.feature)
    for feature in features:
        unserial_info = str(example.features.feature[feature]).split('_')[0]
        image_feature_description[feature] = description[unserial_info]

    return image_feature_description

def load_dataset(path, epochs, columns='y',
                 training=False, batch_size=128, 
                 buffer_size=18000, with_rootnames=False,
                 model_cfg=None, mock_survey=False):
    """
        Loads the tfrecords into a tf.TFRecordDataset.
        
        If the training set is loaded this way it shuffles it, 
        applies the augmentations for each image individually and then
        generates batches of it. These steps are running in parallel with
        the help of the map function.
    """
    print(path)  
    dataset = build_TFRecordDataset(path, columns, with_rootnames=with_rootnames, model_cfg=model_cfg, mock_survey=mock_survey)
    
    if training:  
        dataset = dataset.shuffle(20000)
        dataset = dataset.repeat(epochs+10)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(shear, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(zoom, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size)
        
    return dataset


dataset =  load_dataset('/home/ferreira/scratch/hierarchy_stage1/TNG2023_32/test/*.tfrecords', 1,  with_rootnames=False, batch_size=64)
#dataset = np.load('/home/ferreira/scratch/merger_histories.npy')
#redshifts = tf.concat([ex[2] for ex in dataset], axis=0)
#camera =   tf.concat([ex[1] for ex in dataset], axis=0)

#np.save('/home/ferreira/scratch/redshifts.npy', redshifts)
#np.save('/home/ferreira/scratch/camera.npy', camera)

def SwinV3(config):

    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(include_top=False, weights=None, input_shape=(128, 128, 1), swin_v2=True)(inputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(2, activation='sigmoid')(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def find_best_epoch(model, column='val_loss'):
    name = model.split('/')[-1]
    print(f'/{model}/history*.csv')
    histories = sorted(glob.glob(f'/{model}/history*.csv'), key=get_order)
    history_names = [f.split('/')[-1].split('.csv')[0] for f in histories]

    snap_min = [int(f.split('_')[1]) for f in history_names]
    snap_max = [int(f.split('_')[2]) for f in history_names]
    
    dfs = [pd.read_csv(h) for h in histories]
    
    for df, min in zip(dfs, snap_min):
        df['snap'] = np.arange(min+1, min+6) 

    dfs = pd.concat(dfs)
    dfs['name'] = name
    
    return dfs.loc[dfs.val_accuracy == np.max(dfs.val_accuracy)]['snap'].values[0]

models = []
rootnames = []
import pandas as pd
for i, mp in enumerate(models_path):
    epoch = find_best_epoch(mp)
    print(f'Evaluating {mp} at epoch {epoch}')
    
    
    all_preds =[]
    config = yaml.safe_load(open(mp + '/config.yaml')) 

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(config['modelName'], config['model'])
    
    latest = tf.train.latest_checkpoint(mp + '/checkpoints')
    if epoch > 99:
        model.load_weights(f'{mp}/checkpoints/{epoch}.ckpt').expect_partial()
    else:
        model.load_weights(f'{mp}/checkpoints/0{epoch}.ckpt').expect_partial() 

    inputs = Input(shape=(256, 256, 1))
    #outputs = model.layers[0](inputs)
    #outputs = model.layers[1](outputs)
    
    #model_extractor = Model(inputs=inputs, outputs=outputs)
    #rootnames = [ex[1] for ex in dataset]
    #for item in dataset:
    #    pred = model_extractor.predict(dataset)
     #   all_preds.append(pred) 
    
    pred = model.predict(dataset)
    print(pred.shape)
    models.append(pred)
    #print(.shape)
    np.save(f'/home/ferreira/scratch/STAGE1_on_test1_features_{i}.npy', np.stack(all))
#np.save(f'/home/ferreira/scratch/STAGE2_on_test1.npy', pred)

    #    del model, model_extractor, all_preds


