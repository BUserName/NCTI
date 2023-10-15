#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch

import numpy as np
from utils import load_model, forward_pass
from get_dataloader import prepare_data, get_data



# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract feature for supervised models.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=256, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--record_grad', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    # load dataset
    dset, data_dir, num_classes, metric = get_data(args.dataset)
    args.num_classes = num_classes
    if args.dataset in ['ADE']:
        from yacs.config import CfgNode as CN
        dataset_train = TrainDataset(
        "./segdata/ADE20K",
        "./segdata/ADE20K/training.odgt",
        CN(),
        batch_per_gpu=2)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=1,  # we have modified data_parallel
            shuffle=False,  # we do not use this param
            num_workers=4,
            pin_memory=True)
    else:
        train_loader, val_loader, trainval_loader, test_loader, all_loader = prepare_data(
            dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)
    
    print(f'Train:{len(train_loader.dataset)}, Val:{len(val_loader.dataset)},' 
            f'TrainVal:{len(trainval_loader.dataset)}, Test:{len(test_loader.dataset)} '
            f'AllData:{len(all_loader.dataset)}')
    
    # trainval_loader = all_loader

    fpath = os.path.join('./results_f', 'group1')
    if not os.path.exists(fpath):
        os.makedirs(fpath)  
    
    models_hub = ['inception_v3', 'mobilenet_v2', 'mnasnet1_0', 'densenet121', 'densenet169', 'densenet201', 
             'resnet34', 'resnet50', 'resnet101', 'resnet152', 'googlenet']
    for model in models_hub:
        args.model = model
        model_npy_feature = os.path.join(fpath, f'{args.model}_{args.dataset}_feature.npy')
        model_npy_output = os.path.join(fpath, f'{args.model}_{args.dataset}_output.npy')
        model_npy_label = os.path.join(fpath, f'{args.model}_{args.dataset}_label.npy')

        
        model, fc_layer, _ = load_model(args)
        if args.record_grad:
            model_npy_grad = os.path.join(fpath, f'{args.model}_{args.dataset}_grad.npy')
            grad = iterate_data_gradnorm(trainval_loader, model, fc_layer,args, temperature=1) 
            np.save(model_npy_grad, grad)
            print(f"Grad of {args.model} on {args.dataset} has been saved.")
            continue 
        else:
            if os.path.exists(model_npy_feature) and os.path.exists(model_npy_label) and os.path.exists(model_npy_output):
                print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
                continue
            if args.dataset in ['sun397']:
                X_trainval_feature, X_output, y_trainval = forward_pass(train_loader, model, fc_layer)   
            else:
                X_trainval_feature, X_output, y_trainval = forward_pass(trainval_loader, model, fc_layer)   
        
        #X_trainval_feature, y_trainval = forward_pass_feature(trainval_loader, model)   
        if args.dataset == 'voc2007':
            y_trainval = torch.argmax(y_trainval, dim=1)
        print(f'x_trainval shape:{X_trainval_feature.shape} and y_trainval shape:{y_trainval.shape}')
        
        np.save(model_npy_feature, X_trainval_feature.numpy())
        np.save(model_npy_output, X_output)
        np.save(model_npy_label, y_trainval.numpy())
        # np.save(model_npy_grad, grad)
        print(f"Features and Labels of {args.model} on {args.dataset} has been saved.")
      