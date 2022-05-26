import glob
import os
import shutil

runs = sorted(glob.glob('runs/*'))


for run in runs:
    checkpoints = glob.glob(run + '/checkpoints/*')
    if len(checkpoints) == 0:
        shutil.rmtree(run)