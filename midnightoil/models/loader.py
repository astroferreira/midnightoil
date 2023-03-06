from .mergenext import createMergenext
from .transformers import SwinTransformer, SwinTransformerOpt
from .ferreira2020 import FERREIRA2020Net


models_list = {
    'MergeNeXt' : createMergenext,
    'SwinTransformer' : SwinTransformer,
    'SwinTransformerOpt' : SwinTransformerOpt,
    'FERREIRA' : FERREIRA2020Net
}

def get_model(name, config):
    model = models_list[name](config)
    return model
