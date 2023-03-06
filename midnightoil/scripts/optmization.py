import tensorflow as tf
import optuna
import numpy as np
import logging
import glob

from midnightoil.config.preamble import handle_args


current_run, args, config = handle_args()
fileformat = logging.Formatter('%(levelname)s:%(message)s', datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file = logging.FileHandler(f'runs/{current_run}/debug.log')
file.setLevel(logging.DEBUG)
file.setFormatter(fileformat)

logger.addHandler(file)

from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping 

from midnightoil.callbacks.confusionmatrix import ConfusionMatrixCallback
from midnightoil.training.planner import TrainingPlanner

runPath = f"{config['basePath']}/runs/{current_run}/"
logDir = f"{config['basePath']}/logs/scalars/{current_run}"


"""
    Callback setups
"""
csv_logger = CSVLogger(f"{runPath}/training.csv", append=True, separator=';')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logDir)
checkpoint_callback = ModelCheckpoint(f'{runPath}/checkpoints/' + '{val_loss:.4}_{epoch:03d}.ckpt', 'val_loss', mode='min', save_best_only=True, save_weights_only=True)
earlystopping_callback = EarlyStopping(monitor='val_loss', patience=50)
callbacks = [csv_logger, tensorboard_callback, checkpoint_callback, earlystopping_callback]

def objective(trial):

    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=callbacks)
    tPlanner.loadModelOpt(trial)
    tPlanner.model.summary()


    tPlanner.loadData(training=True)

    if tPlanner.model.count_params() < 1e8:
        tPlanner.train()

    preds = []
    trues = []
    tPlanner.loadData(training=False)
    for i, ex in enumerate(tPlanner.test_dataset):
        
        X = ex[0]
        y = ex[1]
        
        preds.append(tPlanner.model.predict(X))
        trues.append(y)
    
    preds = np.array(np.concatenate(preds))
    trues = np.array(np.concatenate(trues))

    np.save(f'{runPath}/trues.npy', trues)
    np.save(f'{runPath}/preds.npy', preds)
    
    
    cm = tf.math.confusion_matrix(np.argmax(trues, axis=1), np.argmax(preds, axis=1)).numpy()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    logger.debug(cm)
    logger.debug(cm / cm.sum(axis=1, keepdims=True))
    
    return np.min([cmn[0, 0], cmn[1, 1]])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)




