from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization

def bickley2021(cfg):

    # convolution model
    inputs = Input(shape=(128, 128, 1), name='main_input')
    
    x = Conv2D(32, kernel_size=(7,7),activation='relu',
               padding='same',strides=(1, 1),name='Conv_1')(inputs)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C1')(x)
    x = Dropout(cfg['dropout'],name='Drop_C1')(x)
    x = Conv2D(64, kernel_size=(7,7),activation='relu',
               padding='same',strides=(1, 1),name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C2')(x)
    x = Dropout(cfg['dropout'],name='Drop_C2')(x)
    x = BatchNormalization(name='BatchNorm')(x)
    x = Conv2D(128, kernel_size=(7,7),activation='relu',
               padding='same',strides=(1, 1),name='Conv_3')(x)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C3')(x)
    x = Dropout(cfg['dropout'],name='Drop_C3')(x)
    x = Conv2D(128, kernel_size=(7,7),activation='relu',
               padding='same',strides=(1, 1),name='Conv_4')(x)
    x = MaxPooling2D(pool_size=(2,2),name='MP_C4')(x)
    x = Dropout(cfg['dropout'],name='Drop_C4')(x)

    x = Flatten(name='Flatten')(x)
    x = Dense(512,activation='relu',name='Dense_1')(x)
    x = Dropout(cfg['dropout'],name='DropFCL_1')(x)
    x = Dense(128,activation='relu',name='Dense_2')(x)
    x = Dropout(cfg['dropout'],name='DropFCL_2')(x)
    outputs = Dense(2,activation='sigmoid',name='Dense_3')(x)

    # connect and compile
    model = Model(inputs=inputs,outputs=outputs)

    return model
