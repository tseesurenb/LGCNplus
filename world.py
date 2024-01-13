'''
Created on Oct 12, 2023
Pytorch Implementation of TempLGCN in
Tseesuren et al. tempLGCN: Simple and Time-aware Graph Convolution Network for Collaborative Filtering Recommendation

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

#ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
#CODE_PATH = join(ROOT_PATH, 'code')
#DATA_PATH = join(ROOT_PATH, 'data')
#BOARD_PATH = join(CODE_PATH, 'runs')
#FILE_PATH = join(CODE_PATH, 'checkpoints')
#import sys
#sys.path.append(join(CODE_PATH, 'sources'))

#if not os.path.exists(FILE_PATH):
#    os.makedirs(FILE_PATH, exist_ok=True)

config = {}
config['batch_size'] = args.batch_size
config['lr'] = args.lr
config['dataset'] = args.dataset
config['num_layers'] = args.layer
config['emb_dim'] = args.emb_dim
config['model'] = args.model
config['decay'] = args.decay
config['epochs'] = args.epochs
config['top_k'] = args.top_k
config['verbose'] = args.verbose
config['epochs_per_eval'] = args.epochs_per_eval
config['epochs_per_lr_decay'] = args.epochs_per_lr_decay
config['seed'] = args.seed
config['win'] = args.win
config['r_beta'] = args.r_beta
config['a_beta'] = args.a_beta
config['a_method'] = args.a_method
config['r_method'] = args.r_method

#GPU = torch.cuda.is_available()
#device = torch.device('cuda' if GPU else "cpu")
#CORES = multiprocessing.cpu_count() // 2
#seed = args.seed

#dataset = args.dataset
#model_name = args.model
#if dataset not in all_dataset:
#    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
#if model_name not in all_models:
#    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

#TRAIN_epochs = args.epochs
#LOAD = args.load
#PATH = args.path
#topks = eval(args.topks)
#tensorboard = args.tensorboard
#comment = args.comment
# let pandas shut up
#from warnings import simplefilter
#simplefilter(action="ignore", category=FutureWarning)
