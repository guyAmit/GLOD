import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data.sampler import SubsetRandomSampler

from ..models.Resnet import Resnet18, Resnet34
from ..utils.datasets import *
from ..utils.utils import get_test_valid_loader

seed = 15
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def search_thers(preds_in, level):
    step = 1
    val = -1
    eps = 0.001
    for _ in range(1000):
        TNR = (preds_in >= val).sum()/preds_in.shape[0]
        if TNR < level+eps and TNR > level-eps:
            return val
        elif TNR > level:
            val += step
        else:
            val = val - step
            step = step*0.1
    return val


def detection_accuracy(start, end, preds_in, preds_ood):
    step = (end-start)/10000
    val = start
    max_det = 0
    max_thres = val
    while val < end:
        TPR_in = (preds_in >= val).sum()/preds_in.shape[0]
        TPR_out = (preds_ood <= val).sum()/preds_ood.shape[0]
        detection = (TPR_in+TPR_out)/2
        if detection > max_det:
            max_det = detection
            max_thres = val
        val += step
    return max_thres, max_det


def calc_full_covs(net, trainloader, n_classes, layers):
    net.eval()
    layers_centers = []
    layers_precisions = []
    for l in range(layers):
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for (inputs, targets) in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net.intermediate_forward(inputs, layer_index=l)
                outputs_list.append(outputs)
                target_list.append(targets)
            outputs = torch.cat(outputs_list, axis=0)
            target_list = torch.cat(target_list)
            x_dim = outputs.size(1)
            centers = torch.zeros(n_classes, x_dim).cuda()
            normlized_outputs = []
            for c in range(n_classes):
                class_points = outputs[c == target_list]
                centers[c] = torch.mean(class_points, axis=0)
                normlized_outputs.append(
                    class_points-centers[c].unsqueeze(0).expand(
                        class_points.size(0), -1))
            normlized_outputs = torch.cat(normlized_outputs, axis=0).cpu()
            covs_lasso = EmpiricalCovariance(assume_centered=False)
            covs_lasso.fit(normlized_outputs.cpu().numpy())
            precision = torch.from_numpy(covs_lasso.precision_).float().cuda()
            layers_centers.append(centers)
            layers_precisions.append(precision)
    return layers_precisions, layers_centers


def calc_mahalanobis(x, precsion, centers):
    distance = torch.zeros(x.size(0), centers.size(0)).cuda()
    for c in range(centers.size(0)):
        diff = x - centers[c].unsqueeze(0).expand(x.size(0), -1)
        exp_log = -torch.mm(torch.mm(diff, precsion), diff.t()).diag()
        distance[:, c] = exp_log
    return distance


def predict_ensamble_batch(net, inputs, layers_precsions, layers_centers):
    net.eval()
    f_list = net.feature_list(inputs)
    preds = torch.zeros(inputs.size(0), len(layers_centers)).cuda()
    for l in range(len(layers_centers)):
        preds[:, l] = calc_mahalanobis(
            f_list[l], layers_precsions[l], layers_centers[l]).max(1)[0]
    return preds


def predict_mahalanobis_ensamble(net, loader, layers_precsions,
                                 layers_centers, device):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if len(inputs) == 2:
                inputs = inputs[0]
            if type(inputs) == list:
                inputs = inputs[0]
            inputs = inputs.to(device)
            preds = predict_ensamble_batch(
                net, inputs, layers_precsions, layers_centers)
            predictions.append(preds)
    predictions = torch.cat(predictions).cuda()
    return predictions


if __name__ == '__main__':
    dataset = 'cifar10'

    ood_datasets = []
    if dataset == 'cifar10':
        ood_dataset1 = 'svhn'
        ood_dataset2 = 'cifar100'

        n_classes = 10
        in_train_loader = cifar10_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128,
                                                                test_dataset=cifar10_test,
                                                                random_seed=seed,
                                                                valid_size=0.1,
                                                                num_workers=2,
                                                                pin_memory=False)

        out_test_loader1, out_valid_loader1 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=svhn_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)
        out_test_loader2, out_valid_loader2 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=cifar100_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)
    if dataset == 'cifar100':
        ood_dataset1 = 'svhn'
        ood_dataset2 = 'cifar10'

        n_classes = 100
        in_train_loader = cifar100_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128,
                                                                test_dataset=cifar100_test,
                                                                random_seed=seed,
                                                                valid_size=0.1,
                                                                num_workers=2,
                                                                pin_memory=False)

        out_test_loader1, out_valid_loader1 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=svhn_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)
        out_test_loader2, out_valid_loader2 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=cifar10_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)
    if dataset == 'svhn':
        ood_dataset1 = 'cifar10'
        ood_dataset2 = 'cifar100'
        n_classes = 10
        in_train_loader = svhn_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128,
                                                                test_dataset=svhn_test,
                                                                random_seed=seed,
                                                                valid_size=0.1,
                                                                num_workers=2,
                                                                pin_memory=False)

        out_test_loader1, out_valid_loader1 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=cifar10_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)

        out_test_loader2, out_valid_loader2 = get_test_valid_loader(batch_size=128,
                                                                    test_dataset=cifar100_test,
                                                                    random_seed=seed,
                                                                    valid_size=0.1,
                                                                    num_workers=2,
                                                                    pin_memory=False)
    #  create validation sets
    ood_datasets.append((ood_dataset1, out_test_loader1, out_valid_loader1))
    ood_datasets.append((ood_dataset2, out_test_loader2, out_valid_loader2))

    lsun_test_loader, lsun_valid_loader = get_test_valid_loader(batch_size=128,
                                                                test_dataset=lsun_testset,
                                                                random_seed=seed,
                                                                valid_size=0.1,
                                                                num_workers=2,
                                                                pin_memory=False)

    imagenet_test_loader, imagenet_valid_loader = get_test_valid_loader(batch_size=128,
                                                                        test_dataset=imagenet_testset,
                                                                        random_seed=seed,
                                                                        valid_size=0.1,
                                                                        num_workers=2,
                                                                        pin_memory=False)

    iSUN_test_loader, iSUN_valid_loader = get_test_valid_loader(batch_size=128,
                                                                test_dataset=iSUN_testset,
                                                                random_seed=seed,
                                                                valid_size=0.1,
                                                                num_workers=2,
                                                                pin_memory=False)

    ood_datasets.extend([('lsun', lsun_test_loader, lsun_valid_loader),
                         ('imagenet', imagenet_test_loader, imagenet_valid_loader),
                         ('iSUN', iSUN_test_loader, iSUN_valid_loader)])

    net = Resnet34(num_c=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    results = np.zeros((4, 5, 5))
    times = []
    for net_num in range(1, 6):
        #  Laod network
        checkpoint = torch.load(
            f'../resnet/{dataset}_resnet{net_num}.ckpt.pth')
        if list(checkpoint['net'].keys())[0].split('.')[0] == 'module':
            net = nn.DataParallel(net, device_ids=[0])
            net.load_state_dict(checkpoint['net'])
            net = net.module
        else:
            net.load_state_dict(checkpoint['net'])

        layers_precsions, layers_centers = calc_full_covs(net, in_train_loader,
                                                          n_classes=n_classes,
                                                          layers=5)

        mahalanobis_in_dist = predict_mahalanobis_ensamble(net,
                                                           in_test_loader,
                                                           layers_precsions,
                                                           layers_centers,
                                                           device).cpu().numpy()

        preds_in = mahalanobis_in_dist.sum(1, keepdims=True)
        scaler = MinMaxScaler().fit(preds_in)

        start_time = time.time()
        mahalanobis_in_dist = predict_mahalanobis_ensamble(net, in_test_loader,
                                                           layers_precsions,
                                                           layers_centers,
                                                           device)

        preds_in = mahalanobis_in_dist.sum(1, keepdims=True).cpu().numpy()
        preds_in = scaler.transform(preds_in)
        times.append(time.time()-start_time)

        thres95 = search_thers(preds_in, 0.95)
        thres99 = search_thers(preds_in, 0.99)

        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader, ood_valid_loader = ood_loaders

            #  predict with feature ensamble and input preprocess
            mahalanobis_ood = predict_mahalanobis_ensamble(net,
                                                           ood_test_loader,
                                                           layers_precsions,
                                                           layers_centers,
                                                           device)

            mahalanobis_ood = mahalanobis_ood.cpu().numpy()
            preds_ood = mahalanobis_ood.sum(1, keepdims=True)
            preds_ood = scaler.transform(preds_ood)

            # TNR level 1
            TNR = (preds_ood < thres95).sum()/preds_ood.shape[0]
            results[0, data_idx, net_num-1] = TNR
            print(f'{ood_data_name}: {TNR}')

            # TNR level 2
            TNR = (preds_ood < thres99).sum()/preds_ood.shape[0]
            results[1, data_idx, net_num-1] = TNR

            # auroc
            y_true = np.concatenate(
                (np.zeros(preds_ood.shape[0]), np.ones(preds_in.shape[0])))
            preds = np.concatenate((preds_ood, preds_in))
            results[2, data_idx, net_num-1] = roc_auc_score(y_true, preds)

            # detectuin accuracy
            results[3, data_idx, net_num -
                    1] = detection_accuracy(0, 1, preds_in, preds_ood)[1]

        print(f'finished {net_num} networks')

    mean = results.mean(axis=-1)

    if dataset == 'svhn':
        print(
            f'TNR95: cifar10 {mean[0 ,0]}  |  cifar100 {mean[0 ,1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: cifar10 {mean[1, 0]}  |  cifar100 {mean[1 ,1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1 , 3]} |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: cifar10 {mean[2, 0]}  |  cifar100 {mean[2 ,1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: cifar10 {mean[3, 0]}  |  cifar100 {mean[3 ,1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['cifar10', 'cifar100', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    if dataset == 'cifar10':
        print(
            f'TNR95: svhn {mean[0, 0]}  |  cifar100 {mean[0 ,1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: svhn {mean[1, 0]}  |  cifar100 {mean[1 ,1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1, 3]}  |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: svhn {mean[2, 0]}  |  cifar100 {mean[2 ,1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: svhn {mean[3, 0]}  |  cifar100 {mean[3 ,1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['svhn', 'cifar100', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    if dataset == 'cifar100':
        print(
            f'TNR95: svhn {mean[0, 0]}  |  cifar10 {mean[0 ,1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: svhn {mean[1, 0]}  |  cifar10 {mean[1 ,1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1, 3]}  |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: svhn {mean[2, 0]}  |  cifar10 {mean[2 ,1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: svhn {mean[3, 0]}  |  cifar10 {mean[3 ,1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['svhn', 'cifar10', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    df.to_csv(f'./Mahalanobis_equal_weights_{dataset}_results.csv')
    print(f'Avg prediction time {np.mean(np.array(times))}')
