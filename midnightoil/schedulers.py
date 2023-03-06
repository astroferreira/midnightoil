import tensorflow as tf
import numpy as np


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, params, train_size=1000, batch_size=128, restart=True):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = params['upperLR']
        self.total_steps = params['cycleEpochs'] * (train_size // batch_size)
        self.warmup_learning_rate = params['lowerLR']
        self.warmup_steps = params['warmupEpochs'] * (train_size // batch_size)
        self.pi = tf.constant(np.pi)
        self.restart = restart

    def __call__(self, step):

        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        
        n_restarts = tf.cast(step//self.total_steps, dtype=tf.float32)
        mul_steps = tf.cast(self.total_steps*n_restarts, dtype=tf.float32)

        learning_rate = (
            0.5
            * self.learning_rate_base
            * (1 + tf.cos(
                    self.pi
                    * (tf.cast(step, dtype=tf.float32)-tf.cast(mul_steps, dtype=tf.float32) - tf.cast(self.warmup_steps, dtype=tf.float32))
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )

            slope = ( self.learning_rate_base - self.warmup_learning_rate) / self.warmup_steps
            warmup_rate = slope * (tf.cast(step, dtype=tf.float32)-tf.cast(mul_steps, dtype=tf.float32)) + self.warmup_learning_rate
            learning_rate = tf.where(tf.cast(step, dtype=tf.float32)-tf.cast(mul_steps, dtype=tf.float32) < self.warmup_steps, warmup_rate, learning_rate) / tf.cast(1+n_restarts, dtype=tf.float32)
        
        return tf.where(step > self.total_steps, learning_rate, learning_rate, name="learning_rate")



schedulers_list = {
    'WarmUpCosine' : WarmUpCosine
}

def load_scheduler(name, params, train_size=1000, batch_size=128):
    return schedulers_list[name](params, train_size=train_size, batch_size=batch_size)