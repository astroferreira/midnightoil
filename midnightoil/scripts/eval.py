

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

from midnightoil.io.dataset import load_latest_weights, unravel_dataset, list_all_checkpoints
from midnightoil.io.images import mosaic
from midnightoil.training.planner import TrainingPlanner

import coral_ordinal as coral

runPath = f"/home/ferreira/scratch/runs/{current_run}/"
logDir = f"/home/ferreira/scratch/logs/scalars/{current_run}"


tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=[])
tPlanner = load_latest_weights(tPlanner, config, args, current_run, runPath)
print('Loading dataset')

tPlanner.loadData(training=False, batchSize=1536, with_rootnames=True, mock_survey=True)
preds = tPlanner.model.predict(tPlanner.test_dataset)
rootnames = tf.concat([ex[2] for ex in tPlanner.test_dataset], axis=0)

np.save(f"{runPath}/preds_mock_surveyv3.npy", preds)
np.save(f'{runPath}/mock_rootnamesv3.npy', rootnames)


