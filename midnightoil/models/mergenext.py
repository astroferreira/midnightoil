import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import TruncatedNormal


class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.Model):
    """ConvNeXt block.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None


        self.dw_conv_1 = layers.Conv2D(
            filters=dim, kernel_size=5, padding="same", groups=dim,
            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None)
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = layers.Dense(2 * dim, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None))
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(dim, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None))
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x)


def createMergenext(config=None, classification=True) -> keras.Model:

    numOutputs = config['num_classes']
    inputShape = config['input_size']
    stemSize = 1#config['stemUndersample']
    normEpsilon = 0#config['layerNormalizationEpsilon']
    depths = [1, 2, 4] #config['depths']
    dims = [128, 256, 512]#config['dims']
    dropPathRate = 0.0#config['dropPathRate']
    layerScaleInitValue = 1#config['layerScaleInitValue']

    numStages = len(dims)

    inputs = layers.Input(inputShape)
    
    stem = keras.Sequential(
        [   
            layers.Conv2D(dims[0], kernel_size=stemSize, strides=stemSize, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None)),
            layers.LayerNormalization(epsilon=normEpsilon),
        ],
        name="patchfy",
    )

    downsample_layers = []
    downsample_layers.append(stem)
    for i in range(numStages-1):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None)),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    stages = []
    dp_rates = [x for x in tf.linspace(0.0, dropPathRate, sum(depths))]
    cur = 0
    for i in range(numStages):
        stage = keras.Sequential(
            [
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layerScaleInitValue,
                        name=f"convnext_block_{i}_{j}",
                    )
                    for j in range(depths[i])
                ]
            ],
            name=f"convnext_stage_{i}",
        )
        stages.append(stage)
        cur += depths[i]

    x = inputs
    for i in range(len(stages)):
        x = downsample_layers[i](x)
        x = stages[i](x)

    x = layers.GlobalMaxPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    if classification:
        outputs = layers.Dense(numOutputs, activation='softmax', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.2, seed=None), name="classification_head")(x)
    else:
        outputs = layers.Dense(numOutputs, activation='linear', name="regression_head")(x)
        

    return keras.Model(inputs, outputs, name='MergerNeXt')