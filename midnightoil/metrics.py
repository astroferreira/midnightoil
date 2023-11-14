import tensorflow as tf

from coral_ordinal import MeanAbsoluteErrorLabels

metrics_list = {
    'categorical_accuracy' : 'categorical_accuracy',
    'accuracy' : 'accuracy',
    'precision' : tf.keras.metrics.Precision(),
    'recall' : tf.keras.metrics.Recall(),
    'MAELabels' : MeanAbsoluteErrorLabels(),
    'mse' : 'mean_squared_error'
}

def get_metric(metric):
    return metrics_list[metric]


def lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr
