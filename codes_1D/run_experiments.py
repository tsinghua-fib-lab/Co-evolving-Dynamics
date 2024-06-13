import numpy as np
import pandas as pd
import torch
import argparse
import os
import time
import setproctitle
import pickle
from GPUSimulation import *
import tqdm


if __name__ == '__main__':
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Bipartite Network, 1D')
    parser.add_argument('-sd', '--seed', type=int, default=4, help='random seed')
    parser.add_argument('-dv', '--device', type=str, default='cpu', help='device')
    parser.add_argument('-f', '--num_feature', type=int, default=1, help='number of features')
    parser.add_argument('-md', '--mode', type=str, default='vector', help='experiment mode')
    parser.add_argument('-dt', '--dt', type=float, default=0.1, help='interval time')
    parser.add_argument('-T', '--T', type=float, default=5000, help='overall time')
    parser.add_argument('-nf', '--node_feature_path', type=str, default='./Data/node_random.csv', help='node feature path')
    parser.add_argument('-ns', '--network_structure_path', type=str, default='./Data/edge_random.csv', help='network structure path')
    parser.add_argument('-ut', '--update_type', type=str, default='Both+X', help='update type')

    args = parser.parse_args()

    # Settings
    torch.manual_seed(args.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True   


    alpha_list = np.arange(0.00001, 0.00071, 0.00001)
    gamma_plus_list = [0.03]
    gamma_minus_list = [0.01] 
    sigma_list = [0.0003]

    jpm_list = []
    for jp in [1.0]:
        for jm in [-0.9]:
                jpm_list.append((jp,jm))

    parameters = []
    for a in alpha_list:
        for gp in gamma_plus_list:
            for gm in gamma_minus_list:
                for s in sigma_list:
                    for jpm in jpm_list:
                            parameters.append((a,gp,gm,s,jpm))
                    
    for p in tqdm.tqdm(parameters):

        alpha = {}
        alpha['AtoB'] = p[0]
        alpha['BtoA'] = p[0]
        alpha['A'] = None
        alpha['B'] = None
    
        gamma_plus = {}
        gamma_plus['A'] = None
        gamma_plus['B'] = None
        gamma_plus['AB'] = p[1]
        
        gamma_minus = {}
        gamma_minus['A'] = None
        gamma_minus['B'] = None
        gamma_minus['AB'] = p[2]

        sigma = p[3]

        r_plus = {}
        r_plus['A'] = None
        r_plus['B'] = None
        r_plus['AB'] = p[4][0]
        
        r_minus = {}
        r_minus['A'] = None
        r_minus['B'] = None
        r_minus['AB'] = p[4][1]


        states = bipartite_simulation(node_path=args.node_feature_path, edge_path=args.network_structure_path, update_type= args.update_type, num_feature = args.num_feature,\
            alpha=alpha, sigma=sigma, gamma_plus=gamma_plus, gamma_minus=gamma_minus, r_plus=r_plus, r_minus=r_minus,\
            dt=args.dt, overall_time=5*args.T, seed=args.seed, device=args.device, mode=args.mode)


    
