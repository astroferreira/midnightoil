from keras_cv_attention_models import maxvit

mm = maxvit.MaxViT_Tiny(input_shape=(224, 224, 1), num_classes=2, pretrained=None)

print(mm.summary())