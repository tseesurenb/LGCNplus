'''
Created on Oct 12, 2023
Pytorch Implementation of TempLGCN in
Tseesuren et al. tempLGCN: Simple and Time-aware Graph Convolution Network for Collaborative Filtering Recommendation

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog="tempLGCN", description="Dynamic GCN-based CF recommender")
    parser.add_argument('--model', type=str, default='lgcn', help='rec-model, support [lgcn, lgcn_b, lgcn_b+, lgcn_b++]')
    parser.add_argument('--dataset', type=str, default='ml100k', help="available datasets: [ml100k, ml1m, ml10m]")
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--emb_dim', type=int, default=64, help="the embedding size for learning parameters")
    parser.add_argument('--layer', type=int, default=3, help="the layer num of GCN")
    parser.add_argument('--batch_size', type=int, default= 8192, help="the batch size for bpr loss training procedure")
    parser.add_argument('--epochs', type=int,default=501)
    parser.add_argument('--epochs_per_eval', type=int,default=30)
    parser.add_argument('--epochs_per_lr_decay', type=int,default=5000)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
    parser.add_argument('--decay', type=float, default=0, help="the weight decay for l2 normalizaton")
    parser.add_argument('--path', type=str, default="./checkpoints", help="path to save weights")
    parser.add_argument('--top_k', type=int, default=10, help="@k test list")
    parser.add_argument('--r_beta', type=float, default=0.01)
    parser.add_argument('--win', type=float, default=1)
    parser.add_argument('--a_beta', type=float, default=0.15)
    parser.add_argument('--a_method', type=str, default='log')
    parser.add_argument('--r_method', type=str, default='log')
    parser.add_argument('--loadedModel', type=bool, default=False)
    
    
    return parser.parse_args()