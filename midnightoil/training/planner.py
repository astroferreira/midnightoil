import yaml

import tensorflow as tf

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')


import optuna

from ..models import loader
from ..losses import load_loss
from ..schedulers import load_scheduler
from ..optimizers import load_optimizer
from ..metrics import lr_metric
from ..io.dataset import load_dataset
from ..metrics import get_metric

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
        #self.augs = self.config['augmentation']
        self.runName = self.configTraining['runName']
        self.epochs  = self.configTraining['epochs']
        self.initialEpoch = self.configTraining['initialEpoch']
        self.batchSize  = self.configTraining['batchSize']
        self.dataPath  = self.configTraining['dataPath']
        self.evalPath  = self.configTraining['evalPath']
        self.columns = self.configTraining['tfrecordsColumns']
        self.classification = self.config['model']['classification']
        self.modelName = self.config['modelName']
        self.metrics = self.config['metrics']
        self.configScheduler = self.config['learningRateScheduler']

        self.train_size = self.configTraining['train_size']
        self.test_size = self.configTraining['test_size']

        self.augmentations = {} #self.parse_augmentations()

        if not self.configTraining['optmization']:
            if self.configTraining['distributed']:
                self.loadModelDistributed()
            else:
                self.loadModel()    

    def loadModel(self):

        
        

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            self.model = loader.get_model(self.modelName, self.config['model'])
            self.scheduled_lrs = load_scheduler(self.configScheduler['scheduler'], 
                                                self.configScheduler, 
                                                train_size=self.train_size, 
                                                batch_size=self.batchSize)

            self.optimizer = load_optimizer(self.config['optimizer'],
                                            self.scheduled_lrs)
            
            self.loss = load_loss(self.config['loss'])
            
            self.metrics = [get_metric(metric) for metric in self.metrics]


            self.model.compile(optimizer=self.optimizer, loss=self.loss,
                            metrics=self.metrics)

            print(self.model.summary(expand_nested=True))

    def loadModelOpt(self, trial):

        #mixed_precision.set_global_policy('mixed_float16')

        self.model = loader.get_model('SwinTransformerOpt', trial)
        self.scheduled_lrs = load_scheduler(self.configScheduler['scheduler'], self.configScheduler, train_size=self.train_size, batch_size=self.batchSize)

        optmizer = trial.suggest_categorical('optimizer', ['NesterovSGD', 'SGD', 'Adam', 'AdamW'])

        self.optimizer = load_optimizer(optmizer, self.scheduled_lrs)
        self.loss = load_loss(self.config['loss'])
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                        metrics=self.metrics)

    def loadData(self, training=True, batchSize=None, with_rootnames=False):
        
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        print(f'EPOCHS = {self.epochs}')

        if training:
            self.training_dataset = load_dataset(f'{self.dataPath}/*.tfrecords',
                                                epochs=(self.epochs-self.initialEpoch),
                                                columns=self.columns,
                                                training=self.train_size,
                                                batch_size=self.batchSize, 
                                                augmentations=self.augmentations, 
                                                model_cfg=self.config['model'])
        
            self.training_dataset = self.training_dataset.with_options(ignore_order) 

        
            
        if batchSize is None:
            batchSize = self.batchSize

        dataPath = f'{self.dataPath}/val/*1-83.tfrecords'

        if self.evalPath is not False:
            dataPath = f'{self.evalPath}/*.tfrecords'



        self.test_dataset = load_dataset(dataPath, epochs=self.epochs,
                                            columns=self.columns, training=False,
                                            batch_size=batchSize, augmentations=[], with_rootnames=with_rootnames)
        
        self.test_dataset = self.test_dataset.with_options(ignore_order)

    def train(self):

        self.history = self.model.fit(self.training_dataset, 
                                    validation_data=self.test_dataset,
                                    epochs=self.epochs,
                                    initial_epoch=self.initialEpoch,
                                    steps_per_epoch=self.train_size//self.batchSize,
                                    validation_steps=self.test_size//self.batchSize,
                                    callbacks=self.callbacks,
                                    use_multiprocessing = True)

    def parse_augmentations(self):

        self.augmentations = {
            'num_layers' : int(self.augs['num_layers']),
            'flip': self.augs['flip'],
            'rotate90': self.augs['rotate90'],
            'rotate': self.augs['rotate'],
            'shear_x': self.augs['shear_x'],
            'shear_y': self.augs['shear_y'],
            'oclusion': self.augs['oclusion'],
            'zoom': self.augs['zoom']
            }

        
        
