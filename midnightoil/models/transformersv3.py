from keras import layers, models
from tfswin import SwinTransformerTiny224, preprocess_input

def SwinV3(config):

    inputs = layers.Input(shape=(128, 128, 1))
    outputs = SwinTransformerTiny224(include_top=False, weights=None, input_shape=(128, 128, 1), swin_v2=True)(inputs)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Dense(2, activation='sigmoid')(outputs)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


