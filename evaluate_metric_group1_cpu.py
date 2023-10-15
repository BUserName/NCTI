#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import models.group1 as models
import numpy as np

import json
import time

from metrics import LEEP, NLEEP, LogME_Score, SFDA_Score, PARC_Score, NCTI_Score

from utils import fix_python_seed

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
    parser.add_argument('--output-dir', type=str, default='./results_metrics/group1', 
                        help='dir of output score')
    parser.add_argument('--percent', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sample', type=int, default=0)
    args = parser.parse_args()   
    pprint(args)
    fix_python_seed(args.seed)

    score_dict = {}   
    fpath = os.path.join(args.output_dir, args.metric)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    fpath = os.path.join(fpath, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            print(fpath)
            score_dict = json.load(f)
    
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
                    'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    
    start_time = time.time()
    for model in models_hub:
        # if exist_score(model, fpath):
        #     print(f'{model} has been calculated')
        #     continue
        args.model = model
        
        model_npy_feature = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_label.npy')
        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)
        
        # Sample specific percent of data 
        if args.percent<1:
            numsample = np.random.permutation(X_features.shape[0])[:int(X_features.shape[0] * args.percent)]
            X_features = X_features[numsample]
            y_labels = y_labels[numsample] 
        
        # Sample specific number of data 
        elif args.sample>0:
            C = np.unique(y_labels).shape[0]
            X_features_new = []#np.empty(C*args.sample, X_features.shape[1])
            y_labels_new = [] #np.empty(C*args.sample, y_labels.shape[1])
            for c in range(C):
                c_feature = X_features[(y_labels==c).flatten()]
                c_label = y_labels[(y_labels==c).flatten()]
                numsample = np.random.permutation(c_feature.shape[0])[:int(args.sample)]
                X_features_new.append(c_feature[numsample])
                y_labels_new.append(c_label[numsample])
            X_features = np.concatenate(X_features_new, axis=0)
            y_labels = np.concatenate(y_labels_new, axis=0)
            numsample = np.random.permutation(X_features.shape[0])
            X_features = X_features[numsample]
            y_labels = y_labels[numsample]
            print(X_features.shape)

        if args.metric == 'src':
            model_npy_feature = os.path.join('./results_f/group1', f'{args.model}_imagenet_feature.npy')
            model_npy_label = os.path.join('./results_f/group1', f'{args.model}_imagenet_label.npy')
            src_features, src_labels = np.load(model_npy_feature), np.load(model_npy_label)
        if args.metric == 'grad':
            model_npy_grad = os.path.join('./results_f/group1', f'{args.model}_{args.dataset}_grad.npy')
            model_npy_grad = np.load(model_npy_grad)

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
        elif args.metric == 'nce':
            score_dict[args.model] = NCE(X_features, y_labels, model_name=args.model)
        else:
            raise NotImplementedError
        
    print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
    
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
            score_dict[model] = mcscore  + mascore - cpscore
        print("seli")
        
        # print(((all_score - all_score_min)/all_score_div).var())
        print(all_score)
        print("ncc")
        
        # print(((cls_score- cls_score_min)/cls_score_div).var())
        print(cls_score)
        print("vc")
        # print(((cls_compact- cls_compact_min)/cls_compact_div).var())
        print(cls_compact)
    end_time = time.time()
    score_dict['duration'] = end_time- start_time
    print(end_time-start_time)
    save_score(score_dict, fpath)
        
    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    pprint(results)
    results = {a[0]: a[1] for a in results}
    save_score(results, fpath)