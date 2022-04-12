from .mergenext import createMergenext
from .transformers import SwinTransformer


models_list = {
    'MergeNeXt' : createMergenext,
    'SwinTransformer' : SwinTransformer
}

def get_model(name, config, classification=False):
    model = models_list[name](config, classification=classification)
    return model
