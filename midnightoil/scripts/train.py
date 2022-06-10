from midnightoil.config.preamble import handle_args
current_run, args, config = handle_args()

import logging
fileformat = logging.Formatter('%(levelname)s:%(message)s', datefmt="%H:%M:%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file = logging.FileHandler(f'runs/{current_run}/debug.log')
file.setLevel(logging.DEBUG)
file.setFormatter(fileformat)

logger.addHandler(file)


import numpy as np
import glob


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping 


from midnightoil.training.planner import TrainingPlanner




runPath = f"{config['basePath']}/runs/{current_run}/"

csv_logger = CSVLogger(f"{runPath}/training.csv", append=True, separator=';')
logDir = f"{config['basePath']}/logs/scalars/{current_run}"
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logDir)
checkpoint_callback = ModelCheckpoint(f'{runPath}/checkpoints/' + '{val_loss:.4}_{epoch:03d}.ckpt', 'val_loss', mode='min', save_best_only=True, save_weights_only=True)


if args.eval:
    config['trainingPlan']['epochs'] = 1
    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=[checkpoint_callback, csv_logger])
    checkpoint_dir = f'{runPath}/checkpoints/'

    if args.from_epoch == 0:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print(latest)
        if latest is not None:
            last_loss, epoch = latest.split('/')[-1].split('_')
            epoch = int(epoch.split('.')[0])

            logger.info(f'Resuming training from epoch {epoch} to {epoch+200}, starting with val_loss {last_loss}')

    else:
        latest = glob.glob(f'{checkpoint_dir}/*_{args.from_epoch}.ckpt')
        if len(latest) > 0:
            last_loss, epoch = latest.split('/')[-1].split('_')
            epoch = int(epoch.split('.')[0])
 
    print(latest)
    logger.info(f'Loading weights from {latest}')
    tPlanner.model.load_weights(latest).expect_partial()

    preds = []
    trues = []
    tPlanner.loadData(training=False, batchSize=128)
    for i, ex in enumerate(tPlanner.test_dataset):
        
        X = ex[0]
        y = ex[1]
        
        preds.append(tPlanner.model.predict(X))
        trues.append(y)
    
    preds = np.array(np.concatenate(preds))
    trues = np.array(np.concatenate(trues))

    np.save(f'{runPath}/trues.npy', trues)
    np.save(f'{runPath}/preds.npy', preds)
    
    #logger.debug(np.round(trues).shape)
    #logger.debug(np.round(np.abs(preds)).shape)
    #logger.debug(np.unique(np.round(np.abs(preds)), return_counts=True))
    #logger.debug(np.unique(np.round(np.abs(trues)), return_counts=True))
    #
    # logger.debug(np.round(np.abs(trues)))
    cm = tf.math.confusion_matrix(np.argmax(trues, axis=1), np.argmax(preds, axis=1)).numpy()
    logger.debug(cm)
    print(cm)
    logger.debug(cm / cm.sum(axis=1, keepdims=True))
    print(cm / cm.sum(axis=1, keepdims=True))


if args.resume_run is not None:
    checkpoint_dir = f'runs/{current_run}/checkpoints/'
    
    
    epoch, initial_epoch = 200, 0

    
    if config['from-epoch'] == 0:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print(latest)
        if latest is not None:
            last_loss, epoch = latest.split('/')[-1].split('_')
            epoch = int(epoch.split('.')[0])

            logger.info(f'Resuming training from epoch {epoch} to {epoch+200}, starting with val_loss {last_loss}')

    else:
        latest = glob.glob(f'{checkpoint_dir}/{config["from-epoch"]}_*')
        if len(latest) > 0:
            last_loss, epoch = latest.split('/')[-1].split('_')
            epoch = int(epoch.split('.')[0])

    #config['trainingPlan']
    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=[checkpoint_callback, csv_logger])    
    logger.info(f'Loading weights from {latest}')
    tPlanner.model.load_weights(latest)
    tPlanner.initialEpoch = epoch
    tPlanner.epochs = epoch+200
    tPlanner.loadData(training=True)
    
    
    tPlanner.train()
else:    
    tPlanner = TrainingPlanner(config, currentRun=current_run, callbacks=[checkpoint_callback, csv_logger])
    tPlanner.model.summary()
    tPlanner.loadData(training=True)

    tPlanner.train()

