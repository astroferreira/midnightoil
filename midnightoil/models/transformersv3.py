from keras import layers, models
from tfswin import SwinTransformer, SwinTransformerTiny224, preprocess_input
import tensorflow as tf


def SwinV3(config):

    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(include_top=False, weights=None, input_shape=(128, 128, 1), swin_v2=True)(inputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(2, activation='sigmoid')(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def SwinV3UltraTiny(config):
    
    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(include_top=False, depths=(2, 2, 4, 2), num_heads=(2, 2, 6, 6), embed_dim=12, weights=None,
                            swin_v2=True, input_shape=(128, 128, 1))(inputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(128, activation='relu')(outputs)
    outputs = layers.Dropout(0.25)(outputs)
    outputs = layers.Dense(2, activation='sigmoid')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model 

def SwinV3UltraTinyWS(config):

    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(window_size=4, include_top=False, depths=(2, 2, 4, 2), num_heads=(2, 2, 6, 6), embed_dim=12, weights=None,
                            swin_v2=True, input_shape=(128, 128, 1))(inputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(128, activation='relu')(outputs)
    outputs = layers.Dropout(0.25)(outputs)
    outputs = layers.Dense(2, activation='sigmoid')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def SwinV3Config(config):

    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(classes=2, window_size=4, include_top=False, depths=(2, 2, 4, 2), num_heads=(2, 2, 6, 6), embed_dim=config['embed_dim'], weights=None,
                            swin_v2=True, classifier_activation='sigmoid', input_shape=(128, 128, 1))(inputs)
    outputs = layers.Flatten()(outputs)                        
    outputs = layers.Dense(2, dtype='float32', activation='sigmoid')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def SwinV3Config2(config):

    inputs = layers.Input(shape=(224, 224, 1))
    outputs =  SwinTransformer(model_name='swin_tiny_224', pretrain_size=224, window_size=config['window_size'],
                           embed_dim=config['embed_dim'], depths=config['depths'], num_heads=config['num_heads'], path_drop=config['drop_rate'],
                           weights=None, classes=config['num_classes'], include_top=False, input_shape=(224, 224, 1))(inputs)
    outputs = layers.Flatten()(outputs)                        
    outputs = layers.Dense(2, dtype='float32', activation='sigmoid')(outputs)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
