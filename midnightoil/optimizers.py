from tensorflow.keras.optimizers.experimental import SGD, AdamW, Adam, Adadelta#, Lion
from .lion_opt import Lion
#from tensorflow.keras.optimizers import Lion

def load_optimizer(name, lrs):

    optimizers_list = {
        'NesterovSGD' : SGD(learning_rate=0.0001, nesterov=True, momentum=0.9),
        'NesterovSGDCA' : SGD(learning_rate=lrs, nesterov=True, momentum=0.9),
        'SGD': SGD(learning_rate=lrs),
        'Adam': Adam(learning_rate=lrs),
        'AdamWCA': AdamW(learning_rate=lrs),
        'AdamW': AdamW(learning_rate=0.05),
        'Adadelta': Adadelta(learning_rate=0.05),
        'LionCA' : Lion(learning_rate=lrs),
        'Lion' : Lion(learning_rate=1.0e-4),
        'LionLow' : Lion(learning_rate=1.0e-5)
    }

    return optimizers_list[name]
