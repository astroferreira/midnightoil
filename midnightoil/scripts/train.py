from midnightoil.config.preamble import handle_args

import numpy as np
import logging
import glob

current_run, args, config = handle_args()


fileformat = logging.Formatter('%(levelname)s:%(message)s', datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file = logging.FileHandler(f'{config["basePath"]}/runs/{current_run}/debug.log')
file.setLevel(logging.DEBUG)
file.setFormatter(fileformat)

logger.addHandler(file)

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping 

from midnightoil.callbacks.confusionmatrix import ConfusionMatrixCallback
from midnightoil.callbacks.regression import RegressionCallback
from midnightoil.callbacks.learningrate import LRTensorBoard
from midnightoil.callbacks.tfdatadebugger import TFDataDebugger
from midnightoil.training.planner import TrainingPlanner

runPath = f"{config['basePath']}/runs/{current_run}/"
logDir = f"{config['basePath']}/logs/scalars/{current_run}"


"""
    Callback setups
"""
csv_logger = CSVLogger(f"{runPath}/training.csv", append=True, separator=';')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logDir)
checkpoint_callback = ModelCheckpoint(f'{runPath}/checkpoints/' + '{val_loss:.4}_{epoch:03d}.ckpt', 'val_loss', mode='min', save_best_only=True, save_weights_only=True)

callbacks = [checkpoint_callback]#[csv_logger, tensorboard_callback, LRTensorBoard(log_dir=logDir), checkpoint_callback]

if args.resume_run:

    checkpoint_dir = f'{config["basePath"]}/runs/{current_run}/checkpoints/'
    initial_epoch = config['trainingPlan']['from-epoch']

    #if config['from-epoch'] == 0:
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        last_loss, epoch = latest.split('/')[-1].split('_')
        epoch = int(epoch.split('.')[0])
    
        logger.info(f'Resuming training from epoch {epoch} to {epoch+200}, starting with val_loss {last_loss}')

        tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=callbacks)    
        logger.info(f'Loading weights from {latest}')
        tPlanner.model.load_weights(latest)
        tPlanner.initialEpoch = epoch
        tPlanner.epochs = epoch+config['trainingPlan']['epochs']
    else:
        exit()
    #else:
    #    latest = glob.glob(f'{checkpoint_dir}/{config["from-epoch"]}_*')
    #    if len(latest) > 0:
    #        last_loss, epoch = latest.split('/')[-1].split('_')
    #        epoch = int(epoch.split('.')[0])

    #config['trainingPlan']
    
else:    
    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=callbacks)
    tPlanner.model.summary()

tPlanner.loadData(training=True)
#tfdebugger = TFDataDebugger(tPlanner)
#cmcallback = RegressionCallback(tPlanner.model, tPlanner.test_dataset)
#tPlanner.callbacks.append(cmcallback)
tPlanner.train()

