from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import coral_ordinal as coral


from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

def ENB0(cfg):


    inputs = Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 1))

    model_eff = EfficientNetB0(include_top=True, pooling=None, input_tensor=inputs,
                               weights=None, classes=2)
    #model = Model(inputs, model_eff.output, name='EB0_Classification')

    return model_eff

def ENB0withtop2(cfg):

    inputs = Input(shape=(cfg['input_size'][0],  cfg['input_size'][1], 1))

    model_eff = EfficientNetB0(include_top=False, input_tensor=inputs,
                               weights=None)
    x = Flatten()(model_eff.output)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(inputs, x, name='EB0_Classification')

    return model

def ENB0withtop(cfg):

    inputs = Input(shape=(cfg['input_size'][0],  cfg['input_size'][1], 1))

    model_eff = EfficientNetB0(include_top=False, input_tensor=inputs,
                               weights=None)

    x = GlobalAveragePooling2D()(model_eff.output)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='sigmoid')(x)
    model = Model(inputs, x, name='EB0_Classification')

    return model



def CORN_B0(cfg):

    inputs = Input(shape=(cfg['input_size'][0], cfg['input_size'][1], cfg['channels']))

    model = EfficientNetB0(include_top=False, input_tensor=inputs,
                              weights=None)

    x = GlobalAveragePooling2D()(model.output)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = coral.CoralOrdinal(num_classes = cfg['num_classes'])(x)

    model = Model(inputs, outputs, name='EB0_CORN')

    return model


