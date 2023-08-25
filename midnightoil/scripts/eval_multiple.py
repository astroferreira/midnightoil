

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


ckpts = list_all_checkpoints(runPath)
#tPlanner = load_latest_weights(tPlanner, args, config, current_run, runPath)
print(ckpts)
print('Loading dataset')

#trues, preds, rootnames = unravel_dataset(tPlanner, batch_size=1024)
tPlanner = TrainingPlanner(config, currentRun=current_run)
tPlanner.loadData(training=False, batchSize=1536, with_rootnames=False)
#trues = tf.concat([ex[1] for ex in tPlanner.test_dataset], axis=0)
#rootnames = tf.concat([ex[2] for ex in tPlanner.test_dataset], axis=0)
##cameras = tf.concat([ex[3] for ex in tPlanner.test_dataset], axis=0)
#redshifts = tf.concat([ex[4] for ex in tPlanner.test_dataset], axis=0)

#np.save(f'{runPath}/trues.npy', trues)
#np.save(f'{runPath}/rootnames.npy', rootnames)
#np.save(f'{runPath}/cameras.npy', cameras)
#np.save(f'{runPath}/redshifts.npy', redshifts)
#exit()
for ckpt in ckpts:
    print(ckpt)
    print(f'Loading weights from {ckpt}')
    tPlanner.model.load_weights(ckpt).expect_partial()
    preds = tPlanner.model.predict(tPlanner.test_dataset)
    np.save(f"{runPath}/preds_{ckpt.split('/')[-1]}.npy", preds)

#tPlanner.loadData(training=False, batchSize=512, with_rootnames=True)
#trues = tf.concat([ex[1] for ex in tPlanner.test_dataset], axis=0)
#rootnames = tf.concat([ex[2] for ex in tPlanner.test_dataset], axis=0)
#cameras = tf.concat([ex[3] for ex in tPlanner.test_dataset], axis=0)
#redshifts = tf.concat([ex[4] for ex in tPlanner.test_dataset], axis=0)


#print(trues)
#print(preds)
#print(trues.shape, preds.shape)
#np.save(f'{runPath}/trues.npy', trues)

#np.save(f'{runPath}/rootnames.npy', rootnames)
#np.save(f'{runPath}/cameras.npy', cameras)
#np.save(f'{runPath}/redshifts.npy', redshifts)




#cm = tf.math.confusion_matrix(np.argmax(trues, axis=1), np.argmax(preds, axis=1)).numpy()
#print(cm)
#print(cm / cm.sum(axis=1, keepdims=True))

exit()
