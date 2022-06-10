from .mergenext import createMergenext
from .transformers import SwinTransformer


models_list = {
    'MergeNeXt' : createMergenext,
    'SwinTransformer' : SwinTransformer
}

def get_model(name, config):
    model = models_list[name](config)
    return model
