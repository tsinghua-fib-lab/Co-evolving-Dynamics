import numpy as np
import pandas as pd
import torch
import argparse
import os
import time
import matplotlib.pyplot as plt
import tqdm
import pickle
from GPUSimulation_intervention import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-ct', '--city', type = str, default = 'newyork', help='city')
    parser.add_argument('--a', type = float, default = 0.0)
    parser.add_argument('--gp', type = float, default = 0.0)
    parser.add_argument('--gp_new', type = float, default = 0.0)
    parser.add_argument('--gm', type = float, default = 0.0)
    parser.add_argument('--gm_new', type = float, default = 0.0)
    parser.add_argument('--s', type = float, default = 0.0)
    parser.add_argument('--jp', type = float, default = 1.0)
    parser.add_argument('--jm', type = float, default = -1.0)
    parser.add_argument('--seed', type = int, default = 235)

    args = parser.parse_args()
    city = args.city
    a = args.a
    gp = [args.gp, args.gp_new]
    gm = [args.gm, args.gm_new]
    s = args.s
    jpm = (args.jp, args.jm)

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    node_feature_path = 'I://bipartite_dynamics_vector_version/Data/node_'+city+'_2D_PCA_rescale3_2018-01.csv'  # node feature path
    network_structure_path = './Data/edge_random.csv'  # network structure path ()
    update_type = 'Both+X'
    num_feature = 2    # 'number of features'
    dt = 0.1   # interval time
    T = 200000   # overall time
    change_round = 1000001
    device = 'cuda:0'
    mode = 'vector'  # experiment mode
    extended = city + '_%d' % seed  

    epsilon = 0
    p = (a,gp,gm,s,jpm)
        
    alpha = {}
    alpha['AtoB'] = p[0]
    alpha['BtoA'] = p[0] * epsilon
    alpha['A'] = None
    alpha['B'] = None

    gamma_plus = {}
    gamma_plus['A'] = None
    gamma_plus['B'] = None
    gamma_plus['AB'] = p[1][0]

    gamma_plus_new = {}
    gamma_plus_new['A'] = None
    gamma_plus_new['B'] = None
    gamma_plus_new['AB'] = p[1][1]

    gamma_minus = {}
    gamma_minus['A'] = None
    gamma_minus['B'] = None
    gamma_minus['AB'] = p[2][0]

    gamma_minus_new = {}
    gamma_minus_new['A'] = None
    gamma_minus_new['B'] = None
    gamma_minus_new['AB'] = p[2][1]

    sigma = [p[3] * epsilon, p[3]]

    r_plus = {}
    r_plus['A'] = None
    r_plus['B'] = None
    r_plus['AB'] = p[4][0]

    r_minus = {}
    r_minus['A'] = None
    r_minus['B'] = None
    r_minus['AB'] = p[4][1]

    states = bipartite_simulation(node_path=node_feature_path, edge_path=network_structure_path, update_type= update_type, num_feature = num_feature,\
alpha=alpha, sigma=sigma, gamma_plus=gamma_plus, gamma_minus=gamma_minus, r_plus=r_plus, r_minus=r_minus,\
dt=dt, overall_time=T, seed=seed, device=device, mode=mode, gamma_plus_new=gamma_plus_new, gamma_minus_new=gamma_minus_new, r_plus_new=r_plus, r_minus_new=r_minus, change_round=change_round, network_sim_F='linear', extended = extended)