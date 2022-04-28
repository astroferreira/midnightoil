import tensorflow as tf 
import numpy as np
import math
import pandas as pd

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

def parse(image_feature_description, columns, with_labels=True, with_rootnames=False):

    def _parser(ep):

        example = tf.io.parse_single_example(ep, image_feature_description)
        image = tf.io.decode_raw(example['X'], out_type=np.float64)
        image = tf.reshape(image, (128, 128, 3))
        #image = tf.image.resize(image, (224, 224))


        labels = []
        for col in columns:
            if col == 'N_major_mergers_aug':
                if len(columns) > 1:
                    labels.append(tf.one_hot(tf.cast(example[col], tf.uint8), depth=3))
                else:
                    labels = tf.one_hot(tf.cast(example[col], tf.uint8), depth=3)
                #labels.append(one_hot)
            else:
                labels.append(example[col])
                          
        if with_labels:
            if with_rootnames:
                return image, labels, example['rootname']
            else:
                return image, labels
        
        return image

    return _parser

def generate_TFRecords(outputname, df_name, dataset_name):
    
    dataframe = pd.read_pickle(f'/home/ppxlf2/mergenet/data/clean/{df_name}.pk')
    data = np.load(f'/home/ppxlf2/mergenet/data/clean/{dataset_name}.npy')
   
    #if filter:
    #    index = np.where((dataframe.NULL_FLUX_PP_FLAG == 0) & (dataframe.NULL_COVERAGE_FLAG == 0))
    #    dataframe = dataframe.iloc[index]
    # data = data[index]
    #dataframe = dataframe.reset_index(drop=True)

    record_file = f'/home/ppxlf2/mergenet/data/clean/{outputname}.tfrecords'    

    dataframe = dataframe.reset_index(drop=True)
    with tf.io.TFRecordWriter(record_file) as writer:
        for idx, row in dataframe.iterrows():
            print(idx)
            if row.label == 'PM':
                label = 0
            else:
                label = 1
            
            fits_bytes = data[idx, :,:].tobytes()
            
            tf_example = serialize_df(fits_bytes, 
                                       label,
                                       dataframe,
                                       row)

            writer.write(tf_example.SerializeToString())

import tqdm
def write_images_to_tfr_long(df_name, dataset_name, filename:str="datashard", max_files:int=1000, out_dir:str="/home/ppxlf2/mergenet/data/SDSS/"):

    
    dataframe = pd.read_pickle(f'{out_dir}{df_name}.pk')
    data = np.load(f'{out_dir}{dataset_name}.npy')

    splits = (dataframe.shape[0]//max_files) + 1
    if dataframe.shape[0] % max_files == 0:
        splits -= 1
    
    print(f"\nUsing {splits} shard(s) for {dataframe.shape[0]} files, with up to {max_files} samples per shard")

    file_count = 0


    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}-{}.tfrecords".format(out_dir, filename, i+1, splits)
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files: 

            index = i*max_files+current_shard_count
            if index == dataframe.shape[0]: #when we have consumed the whole data, preempt generation
                break


            current_image = data[index]
            row = dataframe.iloc[index]

            if row.label == 'PM':
                label = 0
            else:
                label = 1

            fits_bytes = current_image.tobytes()
            
            tf_example = serialize_df(fits_bytes, 
                                    label,
                                    dataframe,
                                    row)
            #create the required Example representation
           
            writer.write(tf_example.SerializeToString())
            current_shard_count+=1
            file_count += 1

    writer.close()
    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count

def serialize_df(X, y, df, row):
    """
        This function serializes the input image, its labels and all
        other columns present in the dataframe df.
    """

    feature_types = {
        'object' : _str_feature,
        'int64' : _int64_feature,
        'float64' : _float_feature
    }

    features = {'X' : _bytes_feature(X),
                'y' : _int64_feature(y)}

    for colname, dtype in zip(df.columns, df.dtypes):
        features[colname] = feature_types[dtype.name](row[colname])
    
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
