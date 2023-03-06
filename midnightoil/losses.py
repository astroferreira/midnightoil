from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, BinaryFocalCrossentropy

def load_loss(name):

    loss_list = {
        'CategoricalCrossentropyLS' : CategoricalCrossentropy(label_smoothing=0.1),
        'BinaryCrossentropyLS' : BinaryCrossentropy(label_smoothing=0.1),
        'BinaryFocalCrossentropyLS' : BinaryFocalCrossentropy(label_smoothing=0.1),
        'binary_crossentropy' : 'binary_crossentropy'
    }

    return loss_list[name]
