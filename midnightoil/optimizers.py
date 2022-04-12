from tensorflow.keras.optimizers import SGD

def load_optimizer(name, lrs):

    optimizers_list = {
        'NesterovSGD' : SGD(learning_rate=lrs, nesterov=True, momentum=0.9)
    }

    return optimizers_list[name]