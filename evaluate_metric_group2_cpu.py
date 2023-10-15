#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np

import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, NCTI_Score
from utils import fix_python_seed

imagenetacc_dict = {'byol': 74.30, 'deepcluster-v2': 75.20, 'infomin': 73.00, 'insdis': 59.50, 'moco-v1':60.60,
              'moco-v2': 71.10, 'pcl-v1':61.50, 'pcl-v2':67.60, 'sela-v2':71.80, 'swav':75.30, 'pirl': 61.70, 'sup':77.20}

# miou = {'byol': 0, 'deepcluster-v2': 36.51, 'infomin':23.13, 'insdis':24.03, 'moco-v1':25.85, 'moco-v2':22.08, 'pcl-v1':27.26, 'pcl-v2':27.76, 'sela-v2':37.16, 'swav':37.02, 'pirl': 37.16, 'sup': 32.42}
def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme', 
                        help='name of the method for measuring transferability')   
    parser.add_argument('--nleep-ratio', type=float, default=5, 
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='./results_metrics/group2', 
                        help='dir of output score')
    parser.add_argument('--percent', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    args = parser.parse_args()   
    pprint(args)
    fix_python_seed(1)


    score_dict = {}   
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)
    
    models_hub = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1',
              'moco-v2', 'pcl-v1', 'pcl-v2', 'sela-v2', 'swav']#'simclr-v2'
    start_time = time.time()
    for model in models_hub:
        # if exist_score(model, fpath):
        #     print(f'{model} has been calculated')
        #     continue
        args.model = model
        
        model_npy_feature = os.path.join('./results_f/group2', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('./results_f/group2', f'{args.model}_{args.dataset}_label.npy')
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)
        if args.percent<1:
            numsample = np.random.permutation(X_features.shape[0])[:int(X_features.shape[0] * args.percent)]
            X_features = X_features[numsample]
            y_labels = y_labels[numsample]
        if args.metric == 'src':
            model_npy_feature = os.path.join('./results_f/group2', f'{args.model}_imagenet_feature.npy')
            model_npy_label = os.path.join('./results_f/group2', f'{args.model}_imagenet_label.npy')
            src_features, src_labels = np.load(model_npy_feature), np.load(model_npy_label)

        print(f'x_trainval shape:{X_features.shape} and y_trainval shape:{y_labels.shape}')        
        print(f'Calc Transferabilities of {args.model} on {args.dataset}')
     
        if args.metric == 'logme':
            score_dict[args.model] = LogME_Score(X_features, y_labels)
        elif args.metric == 'leep':     
            score_dict[args.model] = LEEP(X_features, y_labels, model_name=args.model)
        elif args.metric == 'parc':           
            score_dict[args.model] = PARC_Score(X_features, y_labels, ratio=args.parc_ratio)
        elif args.metric == 'nleep':           
            ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
            score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)
        elif args.metric == 'sfda':
            score_dict[args.model] = SFDA_Score(X_features, y_labels) 
        elif args.metric == 'ncti':
            score_dict[args.model] = NCTI_Score(X_features, y_labels)
        else:
            raise NotImplementedError
        
        print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        # save_score(score_dict, fpath)
    if args.metric in['ncti']:
        all_score = []
        cls_score = []
        cls_compact = []

        for model in models_hub:
            all_score.append(score_dict[model][0])
            cls_score.append(score_dict[model][1])
            cls_compact.append(score_dict[model][2])
            
        all_score = np.array(all_score)
        cls_score = np.array(cls_score)
        cls_compact = np.array(cls_compact)

        all_score_min = all_score.min()
        all_score_div = all_score.max() - all_score.min()

        cls_score_min = cls_score.min()
        cls_score_div = cls_score.max() - cls_score.min()

        cls_compact_min = cls_compact.min()
        cls_compact_div = cls_compact.max() - cls_compact_min

        for model in models_hub:
            print(model)
            mascore = (score_dict[model][0] - all_score_min)/all_score_div
            mcscore = (score_dict[model][1] - cls_score_min)/cls_score_div
            cpscore = (score_dict[model][2] - cls_compact_min)/cls_compact_div
            print(mascore)
            print(mcscore)
            print(cpscore)
            score_dict[model] = mascore + mcscore - cpscore
        print("seli")
        
        print(((all_score - all_score_min)/all_score_div).var())

        print("ncc")
        
        print(((cls_score- cls_score_min)/cls_score_div).var())

        print("vc")
        print(((cls_compact- cls_compact_min)/cls_compact_div).var())
    end_time = time.time()
    score_dict['duration'] = end_time- start_time
    print(end_time-start_time)
    save_score(score_dict, fpath)
        
    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    pprint(results)
    results = {a[0]: a[1] for a in results}
    save_score(results, fpath)
