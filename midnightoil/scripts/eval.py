

from midnightoil.config.preamble import handle_args
current_run, args, config = handle_args()

import numpy as np
import glob

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping 

from midnightoil.io.dataset import load_latest_weights, unravel_dataset
from midnightoil.io.images import mosaic
from midnightoil.training.planner import TrainingPlanner

import coral_ordinal as coral

runPath = f"{config['basePath']}/runs/{current_run}/"
logDir = f"{config['basePath']}/logs/scalars/{current_run}"


tPlanner = TrainingPlanner(config, currentRun=current_run)
tPlanner = load_latest_weights(tPlanner, args, config, current_run, runPath)
print('Loading dataset')
#predictions = tPlanner.model.predict(tPlanner.test_dataset)
#np.save(f'{runPath}/predictions.npy', predictions)

trues, preds, rootnames = unravel_dataset(tPlanner)

print(trues)
print(preds)

print(trues.shape, preds.shape)
#mosaic(tPlanner.test_dataset)
np.save(f'{runPath}/trues.npy', trues)
np.save(f'{runPath}/preds.npy', preds)
np.save(f'{runPath}/rootnames.npy', rootnames)

#preds_ordinal = coral.ordinal_softmax(preds)
#np.save(f'{runPath}/preds.npy', preds_ordinal)
#print(preds_ordinal)
#print(trues)


cm = tf.math.confusion_matrix(np.argmax(trues, axis=1), np.argmax(preds, axis=1)).numpy()
print(cm)
print(cm / cm.sum(axis=1, keepdims=True))

exit()
