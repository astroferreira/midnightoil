import os 
import sys
import glob
import yaml
import argparse
import tensorflow as tf

tf.keras.backend.set_floatx('float32')
print(tf.keras.backend.floatx())

from datetime import datetime 

from pathlib import Path 
import re 
import math
file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

class Config(object):

    def __init__(self):

        tf.random.set_seed(4466)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

def handle_args():
    parser = argparse.ArgumentParser(description="Train the merger CNN")
    parser.add_argument('--config', default='default.yml')
    parser.add_argument('--GPUS', default='0,1,2,3')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--tf_log_level', default='3')
    parser.add_argument('--resume_run', default=None, nargs='?', const=None)
    parser.add_argument('--from_epoch', default=0)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--eval_epoch', default=None)

    args = parser.parse_args(sys.argv[1:])

    config_name = args.config.split('/')[-1].split('.')[0]
    config = yaml.safe_load(open(args.config)) 
    #config['eval_epoch'] = args.eval_epoch
   
    basePath = os.getcwd()

    print(args.dataset)
    if args.dataset is not None:
        print('TEST')
        config['trainingPlan']['evalPath'] = args.dataset

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.tf_log_level

    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    
    clean_log_folders()
    
    if args.resume_run is None:
        runs = [str(f.split('/')[-1]) for f in sorted(glob.glob(f"/home/ferreira/scratch/runs/*"), key=get_order)]

        if len(runs) > 0:
            print(int(runs[-1].split('_')[0]))
            current_run = f"{int(runs[-1].split('_')[0])+1:04}_{config['model']['name']}"
        else:
            current_run = f"001_{config['model']['name']}"
 
        run_dirname = f"/home/ferreira/scratch/runs/{current_run}"
        if not os.path.exists(run_dirname):
            os.makedirs(run_dirname)
            os.makedirs(run_dirname + '/checkpoints')
            if os.path.exists(args.config):
                with open(f'{run_dirname}/config.yaml', 'w') as file:
                    documents = yaml.dump(config, file)

    else:
        current_run = [f.split('/')[-1] for f in sorted(glob.glob(f"/home/ferreira/scratch/runs/{args.resume_run}*"))][0]
        print(f'Resuming {current_run}')

    


    tfconfig = Config()

    return current_run, args, config


def clean_log_folders():

    runs = sorted(glob.glob(f'/home/ferreira/scratch/runs/*'))

    for run in runs:
        checkpoints = glob.glob(os.path.join(run, 'checkpoints', '*'))
        no_config = ~os.path.exists(os.path.join(run, '/config.yaml'))
        if (len(checkpoints) == 0) & (no_config):
            print(f'Deleting empty run {run}')
            os.system(f'rm -r {run}')
        
    
