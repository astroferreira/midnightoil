from tensorflow.keras.optimizers.experimental import SGD, AdamW, Adam, Adadelta

def load_optimizer(name, lrs):

    optimizers_list = {
        'NesterovSGD' : SGD(learning_rate=0.00001, nesterov=True, momentum=0.9),
        'NesterovSGDCA' : SGD(learning_rate=lrs, nesterov=True, momentum=0.9),
        'SGD': SGD(learning_rate=lrs),
        'Adam': Adam(learning_rate=lrs),
        'AdamWCA': AdamW(learning_rate=lrs),
        'AdamW': AdamW(learning_rate=0.05),
        'Adadelta': Adadelta(learning_rate=0.05)
    }

    return optimizers_list[name]
