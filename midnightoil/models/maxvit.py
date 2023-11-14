from keras_cv_attention_models import maxvit


def MaxViT(config):
    mm = maxvit.MaxViT([2, 2, 5, 2], [8, 16, 32, 64], input_shape=(224, 224, 1), head_dimension=8, num_classes=2, pretrained=None)
    return mm
