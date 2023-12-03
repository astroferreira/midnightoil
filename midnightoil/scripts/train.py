from midnightoil.config.preamble import handle_args
import os
import numpy as np
import logging
import glob
import pandas as pd

current_run, args, config = handle_args()

fileformat = logging.Formatter('%(levelname)s:%(message)s', datefmt="%H:%M:%S")

basePath = os.getcwd()

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping 

from midnightoil.callbacks.confusionmatrix import ConfusionMatrixCallback
from midnightoil.callbacks.regression import RegressionCallback
from midnightoil.callbacks.learningrate import LRTensorBoard
from midnightoil.callbacks.tfdatadebugger import TFDataDebugger
from midnightoil.callbacks.clearmemory import ClearMemory
from midnightoil.training.planner import TrainingPlanner

if os.path.exists('current_run.txt'):
    with open('current_run.txt', 'r') as cr:
        for line in cr:
            current_run = line.strip()

        args.resume_run = True

runPath = f"/home/ferreira/scratch/runs/{current_run}/"
logDir = f"{basePath}/logs/scalars/{current_run}"
"""
    Callback setups
"""
csv_logger = CSVLogger(f"{runPath}/training.csv", append=True, separator=';')
checkpoint_callback = ModelCheckpoint(runPath + 'checkpoints/{epoch:03d}.ckpt', 'val_loss', mode='min', save_weights_only=True)#, save_best_only=True)
clearmemory_callback = ClearMemory()
callbacks = [checkpoint_callback, clearmemory_callback]#[csv_logger, tensorboard_callback, LRTensorBoard(log_dir=logDir), checkpoint_callback]

if args.resume_run:
    checkpoint_dir = f'/home/ferreira/scratch/runs/{current_run}/checkpoints/'
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        epoch = latest.split('/')[-1]
        epoch = int(epoch.split('.')[0])
        config['trainingPlan']['epochs'] = epoch + 10
    
#        print(f'Resuming training from epoch {epoch} to {epoch+5}, starting with val_loss {last_loss}')
        tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=callbacks)    
        print(f'Loading weights from {latest}')
        tPlanner.model.load_weights(latest)
        tPlanner.initialEpoch = epoch
    else:
        exit()

else:    
    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=callbacks)
    tPlanner.model.summary()

tPlanner.loadData(training=True)

try:
    tPlanner.train()
    hist_df = pd.DataFrame(tPlanner.history.history) 
    hist_csv_file = f'{runPath}/history_{tPlanner.initialEpoch}_{tPlanner.epochs}.csv'
    
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

except Exception as ex:
    print(ex)
    pass

with open('current_run.txt', 'w') as cr:
    cr.write(current_run)

