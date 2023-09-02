from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input

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
