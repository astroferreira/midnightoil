import yaml

import tensorflow as tf

from tensorflow.keras import mixed_precision




from ..models import loader
from ..schedulers import load_scheduler
from ..optimizers import load_optimizer
from ..metrics import lr_metric
from ..io.dataset import load_dataset

class TrainingPlanner:


    def __init__(self, config=None, currentRun=None, callbacks=[]):

        self.currentRun = currentRun
        self.config = config
        if self.config is None:
            self.loadDefault()
        else:
            self.loadConfig()

        self.callbacks = callbacks
    
    def loadConfig(self):

        self.configTraining = self.config['trainingPlan']
        self.runName = self.configTraining['runName']
        self.epochs  = self.configTraining['epochs']
        self.initialEpoch = self.configTraining['initialEpoch']
        self.batchSize  = self.configTraining['batchSize']
        self.dataPath  = self.configTraining['dataPath']
        self.columns = self.configTraining['tfrecordsColumns']
        self.classification = self.configTraining['classification']
        self.modelName = self.config['modelName']
        self.loss = self.config['loss']
        self.metrics = self.config['metrics']
        
        

        self.loadModel()
        #self.loadData()
        


    def loadModel(self):
        self.strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(self.strategy.num_replicas_in_sync))
        self.configScheduler = self.config['learningRateScheduler']

        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        with self.strategy.scope():
            self.model = loader.get_model(self.modelName, self.config['model'], self.classification)
            self.scheduled_lrs = load_scheduler(self.configScheduler['scheduler'], self.configScheduler)
            self.optimizer = load_optimizer(self.config['optimizer'], self.scheduled_lrs)
            self.metrics.append(lr_metric(self.optimizer))
            self.model.compile(optimizer=self.optimizer, loss=self.loss,
                               metrics=self.metrics)
        


    def loadData(self, training=True):
        
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        
        if training:
            self.training_dataset = load_dataset(f'{self.dataPath}/train/train*.tfrecords',
                                                epochs=(self.epochs-self.initialEpoch),
                                                columns=self.columns,
                                                training=training,
                                                batch_size=self.batchSize)
        
            self.training_dataset = self.training_dataset.with_options(ignore_order) 

        self.test_dataset = load_dataset(f'{self.dataPath}/val/val*.tfrecords',
                                            columns=self.columns, training=False,
                                            batch_size=self.batchSize, augmentations=[])
        
        self.test_dataset = self.test_dataset.with_options(ignore_order)


    def train(self):

        self.history = self.model.fit(self.training_dataset, 
                                    validation_data=self.test_dataset,
                                    epochs=self.epochs,
                                    initial_epoch=self.initialEpoch,
                                    steps_per_epoch=99648//self.batchSize,
                                    validation_steps=21504//self.batchSize,
                                    callbacks=self.callbacks)

        
