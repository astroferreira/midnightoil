import tensorflow as tf 
import numpy as np
import math
import pandas as pd
import tarfile
import glob
import os
import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    value = float(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    value = int(value)
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _str_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.

    value = str.encode(str(value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bptotf(npfile, filename:str="CFIS", max_files:int=8192, out_dir:str="tfrecords/"):
    
    
    
    splits = (npfile.shape[0]//max_files) + 1
    if npfile.shape[0] % max_files == 0:
        splits -= 1
    
    print(f"\nUsing {splits} shard(s) for {npfile.shape[0]} files, with up to {max_files} samples per shard")


    file_count = 0
    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}-{}.tfrecords".format(out_dir, filename, i+1, splits)
        check_exists = "/home/ferreira/scratch/CFIS_cutouts/{}_{}-{}.tfrecords".format(filename, i+1, splits)
        
        if os.path.exists(check_exists):
            continue
            
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        
        while current_shard_count < max_files: 

            index = i*max_files+current_shard_count
            if index == npfile.shape[0]: #when we have consumed the whole data, preempt generation
                break

            image = npfile[index]
            fits_bytes = image.astype(np.float32).tobytes()
            tf_example = serialize_df(fits_bytes)
            
           
            writer.write(tf_example.SerializeToString())
            current_shard_count+=1
            file_count += 1
        
        writer.flush()
        writer.close()
        #os.system('mv tfrecords/*.tfrecords /home/ferreira/scratch/hierarchy_stage1NL/TNG2023_32/train/')

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count

def serialize_df(X):
    """
        This function serializes the input image, its labels and all
        other columns present in the dataframe df.
    """

    feature_types = {
        'object' : _str_feature,
        'int64' : _int64_feature,
        'float64' : _float_feature
    }

    features = {'X' : _bytes_feature(X)}

    return tf.train.Example(features=tf.train.Features(feature=features))

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



dataset = np.load('/home/ferreira/scratch/packaged.npy')

bptotf(dataset, out_dir='/home/ferreira/scratch/CFIS_cutouts/')