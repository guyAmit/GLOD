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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data.sampler import SubsetRandomSampler

from ..Gaussain_layer import GaussianLayer
from ..models.Resnet import Resnet18, Resnet34
from ..utils.datasets import *
from ..utils.training_utils import *
from ..utils.utils import get_test_valid_loader, search_thers

seed = 15
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def calc_likelihood(x, precsion, centers):
    covs = torch.div(1.0, precsion)
    n_classes = centers.size(0)
    input_dim = x.size(1)
    covs_t = covs.unsqueeze(0).expand(x.size(0), n_classes, input_dim)
    centers_t = centers.unsqueeze(0).expand(x.size(0), n_classes, input_dim)
    diff = x.unsqueeze(1).repeat(1, n_classes, 1) - centers_t
    Z_log = -0.5 * torch.sum(torch.log(covs), axis=-1) - \
        0.5 * input_dim * np.log(2 * np.pi)
    exp_log = -0.5 * \
        torch.sum(diff * (1 / (covs_t + np.finfo(np.float32).eps))
                  * diff, axis=-1)
    likelihood = Z_log + exp_log
    return likelihood


def predict_llr(net, loader, k):
    net.eval()
    llr_scores = []
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scores = net.llr_ood_score(inputs, k)
            llr_scores.append(scores)
        llr_scores = torch.cat(llr_scores)
    return llr_scores


def predict_ensamble_batch(net, inputs, layers_precsions,
                           layers_centers, n_classes):
    net.eval()
    f_list = net.feature_list(inputs)
    n_features = len(f_list)
    preds = torch.zeros(inputs.size(0), n_features).cuda()

    n_features -= 1
    likelihood = f_list[-1]
    topk, indecies = torch.topk(likelihood, n_classes, dim=1)
    llr_score = topk[:, 0] - topk[:, 1:n_classes].mean(1)
    preds[:, n_features] = llr_score

    for l in range(n_features):
        likelihood = calc_likelihood(
            f_list[l], layers_precsions[l], layers_centers[l])
        preds[:, l] = likelihood.max(1)[0]
    return preds


def predict_ensamble(net, loader, layers_precsions,
                     layers_centers, n_classes):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if len(inputs) == 2:
                inputs = inputs[0]
            if type(inputs) == list:
                inputs = inputs[0]
            inputs = inputs.to(device)
            preds = predict_ensamble_batch(net, inputs,
                                           layers_precsions,
                                           layers_centers, n_classes)
            predictions.append(preds)
    predictions = torch.cat(predictions).cuda()
    return predictions


def gen_adversarial_batch(net, inputs, n_classes, eps):
    net.eval()
    inputs = torch.autograd.Variable(inputs.cuda(device), requires_grad=True)

    likelihood = net(inputs)

    topk = torch.topk(likelihood, n_classes, dim=1)[0]
    preds = topk[:, 0] - topk[:, 1:n_classes].mean(1)
    loss = preds.mean()
    loss.backward()
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient[:, 0] = gradient[:, 0] / 0.2023
    gradient[:, 1] = gradient[:, 1] / 0.1994
    gradient[:, 2] = gradient[:, 2] / 0.2010
    pertubed_inputs = torch.add(inputs.data, -eps * gradient).detach()
    #     del outputs, inputs
    return pertubed_inputs


mu, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
lower = [0, 0, 0]
upper = [0, 0, 0]
for d in [0, 1, 2]:
    lower[d] = (0 - mu[d]) / std[d]
    upper[d] = (1 - mu[d]) / std[d]


def gen_adversarial_batch_bound(net, inputs, labels, n_classes,
                                init_eps, bound):
    net.eval()
    pertubed_inputs = []
    input_shape = (1, *(inputs[0].shape))
    for idx in range(inputs.shape[0]):
        sample = torch.autograd.Variable(
            inputs[idx].cuda(device).reshape(*input_shape),
            requires_grad=True)
        eps = init_eps
        lowest_llr = np.inf
        likelihood = net(sample)
        for i in range(8):
            topk = torch.topk(likelihood, n_classes, dim=1)[0]
            llr = topk[:, 0] - topk[:, 1:n_classes].mean(1)[0]
            loss = llr
            loss.backward()
            gradient = sample.grad.data
            sample = torch.add(sample.data, -eps * gradient).detach()
            for dim in [0, 1, 2]:
                sample[:, dim, :, :] = torch.clamp(sample[:, dim, :, :],
                                                   lower[dim], upper[dim])
            with torch.no_grad():
                likelihood = net(sample.cuda(device).reshape(*input_shape))
            topk = torch.topk(likelihood, n_classes, dim=1)[0]
            llr = topk[:, 0] - topk[:, 1:n_classes].mean(1)[0]
            if llr < lowest_llr:
                perturbed_lowest = sample
                lowest_llr = llr

            if llr < bound:
                perturbed_lowest = sample
                break

            sample = torch.autograd.Variable(sample, requires_grad=True)
            likelihood = net(sample)
        pertubed_inputs.append(perturbed_lowest)
    pertubed_inputs = torch.cat(pertubed_inputs, dim=0)
    return pertubed_inputs


def create_adv_loader_bound(net, loader, n_classes, eps, bound):
    net.eval()
    adv_samples = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        adv_inputs = gen_adversarial_batch_bound(
            net, inputs, labels, n_classes, eps, bound)
        # adv_inputs = gen_adversarial_batch(inputs, n_classes, eps)
        adv_samples.append(adv_inputs)

    adv_samples = torch.cat(adv_samples).to('cpu')
    fake_labels = torch.zeros(adv_samples.size(0))
    adv_dataset = torch.utils.data.TensorDataset(adv_samples, fake_labels)
    adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=128,
                                             shuffle=True, num_workers=4)
    return adv_loader


if __name__ == '__main__':
    dataset = 'cifar100'
    eps = 0.01
    thres = 0.95
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Resnet34(n_classes)
#     net = Resnet18(n_classes)

    net = Resnet_GaussianLayer(net, num_classes=n_classes)
    net = net.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    results = np.zeros((4, 5, 5))
    times = []

    #     _, tiny_images_valid_loader = get_test_valid_loader(batch_size=128,
    #                                                             test_dataset=tiny_images_dataset,
    #                                                             random_seed=seed,
    #                                                             valid_size=0.00002,
    #                                                             num_workers=2)

    for net_num in range(1, 6):
        print(f'Started network number {net_num}')
        #  Load network
        checkpoint = torch.load(f'./GLOD_models/resnet34/{dataset}_GLOD{net_num}_aug.ckpt',
                                map_location=device)['net']

        net.load_state_dict(checkpoint)

        layers_precsions, layers_centers = calc_params(
            net, in_train_loader, n_classes=n_classes, layers=5)

        llr_in_distribution = predict_llr(
            net, loader=in_train_loader, k=n_classes)
        bound = search_thers_torch(llr_in_distribution, thres)
        adv_loader = create_adv_loader_bound(
            net, in_valid_loader, n_classes, eps, bound)

        preds_in = predict_ensamble(net, in_valid_loader, layers_precsions,
                                    layers_centers, n_classes).cpu()
        preds_adv = predict_ensamble(net, adv_loader, layers_precsions,
                                     layers_centers, n_classes).cpu()

        logit_train = torch.cat((preds_adv, preds_in), dim=0).numpy()
        scaler = MinMaxScaler().fit(logit_train)
        logit_train = scaler.transform(logit_train)
        labels = np.concatenate(
            (np.zeros(preds_adv.size(0)), np.ones(preds_in.size(0))))
        regression = LogisticRegressionCV(n_jobs=2, max_iter=200)
        #regression = LogisticRegressionCV(n_jobs=2)

        regression.fit(logit_train, labels)

        start_time = time.time()
        features_in = predict_ensamble(net, in_test_loader, layers_precsions,
                                       layers_centers, n_classes).cpu()
        features_in = scaler.transform(features_in)
        preds_in = regression.predict_proba(features_in)[:, 1]
        times.append(time.time()-start_time)

        thres95 = search_thers(preds_in, 0.95)
        thres99 = search_thers(preds_in, 0.99)

        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader, ood_valid_loader = ood_loaders
            features_ood = predict_ensamble(net, ood_test_loader, layers_precsions,
                                            layers_centers, n_classes).cpu()
            features_ood = scaler.transform(features_ood)
            preds_ood = regression.predict_proba(features_ood)[:, 1]

            # TNR level 1
            TNR95 = (preds_ood < thres95).sum() / preds_ood.shape[0]
            results[0, data_idx, net_num - 1] = TNR95
            print(f'{ood_data_name}: {TNR95}')

            # TNR level 2
            TNR99 = (preds_ood < thres99).sum() / preds_ood.shape[0]
            results[1, data_idx, net_num - 1] = TNR99

            # auroc
            y_true = np.concatenate(
                (np.zeros(preds_ood.shape[0]), np.ones(preds_in.shape[0])))
            preds = np.concatenate((preds_ood, preds_in))
            results[2, data_idx, net_num - 1] = roc_auc_score(y_true, preds)

            # detectuin accuracy
            results[3, data_idx, net_num -
                    1] = detection_accuracy(0, 1, preds_in, preds_ood)[1]

    mean = results.mean(axis=-1)

    if dataset == 'svhn':
        print(
            f'TNR95: cifar10 {mean[0, 0]}  |  cifar100 {mean[0, 1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: cifar10 {mean[1, 0]}  |  cifar100 {mean[1, 1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1, 3]}  |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: cifar10 {mean[2, 0]}  |  cifar100 {mean[2, 1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: cifar10 {mean[3, 0]}  |  cifar100 {mean[3, 1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['cifar10', 'cifar100', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    if dataset == 'cifar10':
        print(
            f'TNR95: svhn {mean[0, 0]}  |  cifar100 {mean[0, 1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: svhn {mean[1, 0]}  |  cifar100 {mean[1, 1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1, 3]}  |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: svhn {mean[2, 0]}  |  cifar100 {mean[2, 1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: svhn {mean[3, 0]}  |  cifar100 {mean[3, 1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['svhn', 'cifar100', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    if dataset == 'cifar100':
        print(
            f'TNR95: svhn {mean[0, 0]}  |  cifar10 {mean[0, 1]}  |  lsun {mean[0, 2]}  |  imagenet {mean[0, 3]}  |  iSUN {mean[0, 4]}')
        print(
            f'TNR99: svhn {mean[1, 0]}  |  cifar10 {mean[1, 1]}  |  lsun {mean[1, 2]}  |  imagenet {mean[1, 3]}  |  iSUN {mean[1, 4]}')
        print(
            f'AUROC: svhn {mean[2, 0]}  |  cifar10 {mean[2, 1]}  |  lsun {mean[2, 2]}  |  imagenet {mean[2, 3]}  |  iSUN {mean[2, 4]}')
        print(
            f'Detection Accuracy: svhn {mean[3, 0]}  |  cifar10 {mean[3, 1]}  |  lsun {mean[3, 2]}  |  imagenet {mean[3, 3]}  |  iSUN {mean[3, 4]}')
        df = pd.DataFrame(mean, columns=['svhn', 'cifar10', 'lsun', 'imagenet', 'iSUN'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
    df.to_csv(f'./FOOD_{dataset}_resnet34_results.csv')
    print(f'Avg prediction time {np.mean(np.array(times))}')
