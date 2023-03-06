from tensorflow.keras.optimizers.experimental import SGD, AdamW, Adam

def load_optimizer(name, lrs):

    optimizers_list = {
        'NesterovSGD' : SGD(learning_rate=lrs, nesterov=True, momentum=0.9),
        'SGD': SGD(learning_rate=lrs),
        'Adam': Adam(learning_rate=lrs),
        'AdamW': AdamW(learning_rate=lrs)
    }

    return optimizers_list[name]
