import yaml
import os
import tensorflow as tf

from tensorflow.keras import mixed_precision
from tensorflow import keras



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
        self.basePath  = os.getcwd()#self.configTraining['dataPath']
        self.dataPath = self.configTraining['dataPath']
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

        
        
        #mixed_precision.set_global_policy('mixed_float16')
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
            self.metrics.append(tf.keras.metrics.Precision())
            self.metrics.append(tf.keras.metrics.Recall())
            self.model.compile(optimizer=self.optimizer, loss=self.loss,
                            metrics=self.metrics)

            print(self.model.summary(expand_nested=True))

        #self.train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        #self.val_acc_metric = tf.keras.metrics.CategoricalAccuracy()



    def loadModelOpt(self, trial):

        #mixed_precision.set_global_policy('mixed_float16')

        self.model = loader.get_model('SwinTransformerOpt', trial)
        self.scheduled_lrs = load_scheduler(self.configScheduler['scheduler'], self.configScheduler, train_size=self.train_size, batch_size=self.batchSize)

        optmizer = trial.suggest_categorical('optimizer', ['NesterovSGD', 'SGD', 'Adam', 'AdamW'])

        self.optimizer = load_optimizer(optmizer, self.scheduled_lrs)
        self.loss = load_loss(self.config['loss'])
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                        metrics=self.metrics)

    def loadData(self, training=True, batchSize=None, with_rootnames=False, mock_survey=False):
        
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        print(f'EPOCHS = {self.epochs}')

        if training:
            print(f'{self.basePath}/{self.dataPath}*.tfrecords')
            self.training_dataset = load_dataset(f'{self.basePath}/{self.dataPath}*.tfrecords',
                                                epochs=(self.epochs-self.initialEpoch),
                                                columns=self.columns,
                                                training=self.train_size,
                                                batch_size=self.batchSize, 
                                                model_cfg=self.config['model'], mock_survey=mock_survey)
        
            self.training_dataset = self.training_dataset.with_options(ignore_order) 

        
            
        if batchSize is None:
            batchSize = self.batchSize

        dataPath = f'{self.dataPath}/val/*1-83.tfrecords'

        if self.evalPath is not False:
            dataPath = f'{self.basePath}/{self.evalPath}*.tfrecords'



        self.test_dataset = load_dataset(dataPath, epochs=self.epochs,
                                            columns=self.columns,
                                            training=False,
                                            batch_size=batchSize,
                                            model_cfg=self.config['model'],
                                            with_rootnames=with_rootnames,
                                            mock_survey=mock_survey)
        
        self.test_dataset = self.test_dataset.with_options(ignore_order)

    def train(self):
        class_weight = {0: 1.,
                        1: 8.,
                        2: 8.,
                        3: 8.,
                        4: 8.,
                        5: 8.,
                        6: 8.,
                        7: 8.,
                        8: 8}
        self.history = self.model.fit(self.training_dataset, 
                                    validation_data=self.test_dataset,
                                    epochs=self.epochs,
                                    initial_epoch=self.initialEpoch,
                                    steps_per_epoch=self.train_size//self.batchSize,
                                    validation_steps=self.test_size//self.batchSize,
                                    callbacks=self.callbacks,
                                    #class_weight=class_weight,
                                    use_multiprocessing=True)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss(y, logits)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        val_logits = self.model(x, training=False)
        val_acc_metric.update_state(y, val_logits)

    def train_custom(self):
        import time
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.training_dataset):
                loss_value = self.train_step(x_batch_train, y_batch_train)

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %d samples" % ((step + 1) * batch_size))

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in self.test_dataset:
                test_step(x_batch_val, y_batch_val)

            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))


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

        
        
