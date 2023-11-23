from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def ConvNeXtTiny_model(cfg):

    inputs = Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 1))

    model_conv = ConvNeXtTiny(
                            model_name="convnext_tiny",
                            include_preprocessing=False,
                            include_top=True,
                            weights=None,
                            input_tensor=inputs,
                            classes=2,
                            classifier_activation="sigmoid",
                        )

    #model = Model(inputs, model_eff.output, name='EB0_Classification')

    return model_conv