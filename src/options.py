#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import argparse
from html import parser

def args_parser():
     parser = argparse.ArgumentParser()

     # federated arguments (Notation for the arguments followed from paper)
     parser.add_argument('--epochs', type=int, default=10,
                         help="number of rounds of training")
     parser.add_argument('--num_users', type=int, default=100,
                         help="number of users: K")
     parser.add_argument('--frac', type=float, default=0.2,
                         help='the fraction of clients: C')
     parser.add_argument('--local_ep', type=int, default=5,
                         help="the number of local epochs: E")
     parser.add_argument('--local_bs', type=int, default=5,
                         help="local batch size: B")
     parser.add_argument('--lr', type=float, default=0.001,
                         help='learning rate')
     parser.add_argument('--momentum', type=float, default=0.5,
                         help='SGD momentum (default: 0.5)')

     # NEW: Modern Dirichlet distribution parameters
     parser.add_argument('--alpha', type=float, default=0.5,
                         help='Dirichlet concentration parameter for non-IID data distribution. '
                              'Controls data heterogeneity level: '
                              '10.0+ (nearly IID), 1.0 (moderate), 0.5 (high), 0.1 (extreme), 0.01 (very extreme)')

     parser.add_argument('--min_samples', type=int, default=50,
                         help='Minimum samples per user to avoid empty clients in Dirichlet distribution')

     # NEW: Data analysis and visualization options
     parser.add_argument('--analyze_data', type=int, default=0,
                         help='Set to 1 to analyze and print detailed data distribution statistics')

     parser.add_argument('--plot_data', type=int, default=0,
                         help='Set to 1 to generate and display data distribution heatmap')

     parser.add_argument('--save_plot', type=str, default=None,
                         help='Path to save distribution plot (e.g., "./plots/distribution.png"). '
                              'Only works when --plot_data is set to 1')
                              
     # PUMB-specific arguments
     parser.add_argument('--pumb_exploration_ratio', type=float, default=0.4,
                         help='Exploration ratio for intelligent client selection in PUMB')
     parser.add_argument('--pumb_initial_rounds', type=int, default=15,
                         help='Number of initial rounds for cold start in PUMB (uses vanilla aggregation)')
     parser.add_argument('--quality_alpha', type=float, default=0.6, help='Alpha (loss weight) for quality metric')
     parser.add_argument('--quality_beta', type=float, default=0.3, help='Beta (consistency weight) for quality metric')
     parser.add_argument('--quality_gamma', type=float, default=0.1, help='Gamma (data weight) for quality metric')
     parser.add_argument('--quality_baseline', type=float, default=0.5, help='Baseline quality for quality metric')
     
     # ✅ ADD: FedProx-specific arguments
     parser.add_argument('--mu', type=float, default=0.01,
                         help='FedProx proximal term coefficient')

     # ✅ ADD: SCAFFOLD-specific arguments
     parser.add_argument('--scaffold_stepsize', type=float, default=1.0,
                    help='Server stepsize for SCAFFOLD (default: 1.0)')
     parser.add_argument('--scaffold_use_yogi', type=int, default=0,
                    help='Use Yogi optimizer for server updates in SCAFFOLD (default: 0)')


     # ✅ ADD: Power-of-Choice specific arguments
     parser.add_argument('--d', type=int, default=10,
                         help='Power-of-Choice: number of candidates to sample for each client selection')
     parser.add_argument('--power_strategy', type=str, default='largest_data',
                         choices=['largest_data', 'random', 'smallest_loss'],
                         help='Power-of-Choice selection strategy: largest_data, random, or smallest_loss')

     # ✅ ADD: FedNova specific arguments
     parser.add_argument('--fednovaaggmethod', type=str, default='baseline',
                         choices=['baseline', 'fedavg'],
                         help='FedNova aggregation method')
     parser.add_argument('--gm', type=float, default=1.0,
                         help='FedNova global momentum parameter')
     parser.add_argument('--tau', type=int, default=None,
                         help='FedNova local steps normalization (if None, uses local_ep)')

     # ✅ ADD: General comparison arguments
     parser.add_argument('--algorithm', type=str, default='fedavg',
                         choices=['fedavg', 'fedprox', 'scaffold', 'power_of_choice', 'pumb', 'fednova', 'lag'],
                         help='Federated learning algorithm to use')

     parser.add_argument('--comparison_mode', type=int, default=0,
                         help='Set to 1 to run in comparison mode (saves detailed metrics)')

     # model arguments
     parser.add_argument('--model', type=str, default='cnn', help='model name')
     parser.add_argument('--kernel_num', type=int, default=9,
                         help='number of each kind of kernel')
     parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                         help='comma-separated kernel size to use for convolution')
     parser.add_argument('--num_channels', type=int, default=3, 
                         help="number of channels of imgs")
     parser.add_argument('--norm', type=str, default='batch_norm',
                         help="batch_norm, layer_norm, or None")
     parser.add_argument('--num_filters', type=int, default=32,
                         help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.") 
     parser.add_argument('--max_pool', type=str, default='True',
                         help="Whether use max pooling rather than strided convolutions")

     # dataset and system arguments
     parser.add_argument('--dataset', type=str, default='cifar', 
                         help="name of dataset: 'cifar10', 'cifar100', 'femnist'")
     parser.add_argument('--num_classes', type=int, default=10, 
                         help="number of classes (10 for cifar10, 100 for cifar100)")
     parser.add_argument('--iid', type=int, default=1,
                         help='Default set to IID. Set to 0 for non-IID.')

     # system arguments
     parser.add_argument('--gpu_id', type=int, default=None, help="GPU ID to use")
     parser.add_argument('--gpu', default=None, 
                         help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
     parser.add_argument('--optimizer', type=str, default='adam', 
                         help="type of optimizer")
     parser.add_argument('--stopping_rounds', type=int, default=10,
                         help='rounds of early stopping')
     parser.add_argument('--verbose', type=int, default=0, help='verbose')
     parser.add_argument('--seed', type=int, default=42, help='random seed')

     args = parser.parse_args()
     return args
