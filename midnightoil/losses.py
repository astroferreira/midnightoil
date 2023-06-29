from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, BinaryFocalCrossentropy, MeanSquaredError, MeanAbsoluteError, MeanSquaredLogarithmicError
from coral_ordinal import OrdinalCrossEntropy


def load_loss(name):

    loss_list = {
        'CategoricalCrossentropyLS' : CategoricalCrossentropy(label_smoothing=0.1),
        'BinaryCrossentropyLS' : BinaryCrossentropy(label_smoothing=0.15),
        'BinaryFocalCrossentropyLS' : BinaryFocalCrossentropy(label_smoothing=0.1),
        'binary_crossentropy' : 'binary_crossentropy',
        'categorical_crossentropy' : 'categorical_crossentropy',
        'mse': MeanSquaredError(),
        'mae': MeanAbsoluteError(),
        'msle': MeanSquaredLogarithmicError(),
        'CordinalCrossEntropy' : OrdinalCrossEntropy()
    }

    return loss_list[name]
