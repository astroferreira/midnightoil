from .mergenext import createMergenext
from .transformers import SwinTransformer, SwinTransformerOpt, CORN_Swin
from .ferreira2020 import FERREIRA2020Net
from .bickley2021 import bickley2021
from .efficientnet import ENB0, CORN_B0, ENB0withtop
from .transformersv2 import SwinTransformerv2
from .transformersv3 import SwinV3, SwinV3UltraTiny, SwinV3UltraTinyWS, SwinV3Config, SwinV3Config2, SwinV3Config3, SwinV3Config4
from .convnext import ConvNeXtTiny_model
#from .maxvit import MaxViT


models_list = {
    'MergeNeXt' : createMergenext,
    'SwinTransformer' : SwinTransformer,
    'SwinTransformerv2' : SwinTransformerv2,
    'SwinTransformerOpt' : SwinTransformerOpt,
    'FERREIRA' : FERREIRA2020Net,
    'BICKLEY'  : bickley2021,
    'EfficientNetB0': ENB0,
    'EfficientCornB0' :  CORN_B0,
    'CORN_SWIN' : CORN_Swin,
    'SwinV3' : SwinV3,
    'SwinV3UltraTiny' : SwinV3UltraTiny,
    'SwinV3UltraTinyWS' : SwinV3UltraTinyWS,
    'SwinV3Config' : SwinV3Config,
    'SwinV3Config2' : SwinV3Config2,
    #'MaxViT' :  MaxViT,
    'SwinV3Config3': SwinV3Config3,
    'SwinV3Config4' : SwinV3Config4,
    'ENB0withtop' : ENB0withtop,
    'ENB0' : ENB0,
    'ConvNextTiny' : ConvNeXtTiny_model
}

def get_model(name, config):
    model = models_list[name](config)
    return model
