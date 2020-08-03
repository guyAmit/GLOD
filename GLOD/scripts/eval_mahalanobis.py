import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.autograd import Variable

from ..models.Resnet import get_ResNet34
from ..utils.training_utils import validation_split
from ..utils.ood_utils import (
    search_thers, auroc_score, detection_accuracy)

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010)), ])


cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=test_transform)

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True,
                                             transform=test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train,
                                                   batch_size=128,
                                                   shuffle=True,
                                                   num_workers=2)


svhn_train = torchvision.datasets.SVHN(root='./data/SVHN/', split='train',
                                       download=True, transform=test_transform)
svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=128,
                                                shuffle=True, num_workers=2)

svhn_test = torchvision.datasets.SVHN(root='./data/SVHN/', split='test',
                                      download=True, transform=test_transform)

cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True,
                                               transform=test_transform)

cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train,
                                                    batch_size=128,
                                                    shuffle=True,
                                                    num_workers=2)

cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True,
                                              transform=test_transform)

lsun_path = os.path.expanduser('/home/guy5/Likelihood_model/LSUN_resize')
lsun_testset = torchvision.datasets.ImageFolder(
    root=lsun_path, transform=test_transform)


imagenet_path = os.path.expanduser(
    '/home/guy5/Likelihood_model/Imagenet_resize')
imagenet_testset = torchvision.datasets.ImageFolder(
    root=imagenet_path, transform=test_transform)


def calc_full_covs(trainloader, n_classes, layers):
    layers_centers = []
    layers_precisions = []
    for layer in range(layers):
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for (inputs, targets) in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net.module.intermediate_forward(
                    inputs, layer_index=layer)
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


def predict_ensamble_batch(inputs, layers_precsions, layers_centers):
    f_list = net.module.feature_list(inputs)
    preds = torch.zeros(inputs.size(0), len(layers_centers)).cuda()
    for layer in range(len(layers_centers)):
        preds[:, layer] = calc_mahalanobis(
            f_list[layer], layers_precsions[layer],
            layers_centers[layer]).max(1)[0]
    return preds


def predict_mahalanobis_ensamble(loader, layers_precsions, layers_centers):
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
                inputs, layers_precsions, layers_centers)
            predictions.append(preds)
    predictions = torch.cat(predictions).cuda()
    return predictions


def gen_adversarial_batch(inputs, precsion, centers, eps):
    inputs = Variable(inputs.cuda(device), requires_grad=True)
    outputs = net.module.pen_forward(inputs)
    initial_preds = calc_mahalanobis(outputs, precsion, centers).max(1)[0]
    loss = initial_preds.mean()
    loss.backward()
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    gradient[:, 0] = gradient[:, 0] / 0.2023
    gradient[:, 1] = gradient[:, 1] / 0.1994
    gradient[:, 2] = gradient[:, 2] / 0.2010
    pertubed_inputs = torch.add(inputs.data, -eps * gradient).detach()
    del outputs, inputs
    return pertubed_inputs


def predict_preprocess_ensamble(loader, layers_precsions, layers_centers, eps):
    net.eval()
    predictions = []
    for batch_idx, inputs in enumerate(loader):
        if len(inputs) == 2:
            inputs = inputs[0]
        if type(inputs) == list:
            inputs = inputs[0]
        adv_inputs = gen_adversarial_batch(inputs, layers_precsions[len(
            layers_centers)-1], layers_centers[len(layers_centers)-1], eps)
        with torch.no_grad():
            adv_mahalanobis = predict_ensamble_batch(
                adv_inputs, layers_precsions, layers_centers)
        predictions.append(adv_mahalanobis)
    predictions = torch.cat(predictions).cuda()
    return predictions


def tune_mahalanobis(in_valid_loader, ood_valid_loader,
                     layers_precsions, layers_centers):
    M_list = [0.0, 0.01, 0.005, 0.0035, 0.002, 0.0014, 0.001, 0.0005]
    best_conf = ()
    best_tnr = -np.inf

    for eps in M_list:
        preds_in = predict_preprocess_ensamble(in_valid_loader,
                                               layers_precsions,
                                               layers_centers, eps).cpu()
        preds_ood = predict_preprocess_ensamble(ood_valid_loader,
                                                layers_precsions,
                                                layers_centers, eps).cpu()
        logit_train = torch.cat((preds_ood[:int(preds_ood.size(0)/2)],
                                 preds_in[:int(preds_in.size(0)/2)]),
                                dim=0).cpu()
        labels = np.concatenate((np.zeros(int(preds_ood.size(0)/2)),
                                 np.ones(int(preds_in.size(0)/2))))
        regression = LogisticRegressionCV(n_jobs=2)
        regression.fit(logit_train, labels)

        preds_in = predict_preprocess_ensamble(in_valid_loader,
                                               layers_precsions,
                                               layers_centers, eps).cpu()
        preds_ood = predict_preprocess_ensamble(ood_valid_loader,
                                                layers_precsions,
                                                layers_centers, eps).cpu()

        val_preds_in = regression.predict_proba(
            preds_in[int(preds_in.size(0)/2):])[:, 1]
        val_preds_ood = regression.predict_proba(
            preds_ood[int(preds_ood.size(0)/2):])[:, 1]

        thres = search_thers(val_preds_in, 0.95, -1)
        TNR = (val_preds_ood < thres).sum()/val_preds_ood.shape[0]
        print(f'Epsilon: {eps} TNR: {TNR}')
        if TNR > best_tnr:
            best_tnr = TNR
            best_conf = eps
    return best_conf


if __name__ == '__main__':
    dataset = 'svhn'

    ood_datasets = []
    if dataset == 'cifar10':
        n_classes = 10
        in_train_set = cifar10_train_loader
        in_test_loader, in_valid_loader = validation_split(
            batch_size=128, test_dataset=cifar10_test, valid_size=0.1)
        out_test_loader, out_valid_loader = validation_split(
            batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'cifar100':
        n_classes = 100
        in_train_loader = cifar100_train_loader
        in_test_loader, in_valid_loader = validation_split(
            batch_size=128, test_dataset=cifar100_test, valid_size=0.1)
        out_test_loader, out_valid_loader = validation_split(
            batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'svhn':
        n_classes = 10
        in_train_loader = svhn_train_loader
        in_test_loader, in_valid_loader = validation_split(
            batch_size=128, test_dataset=svhn_test, valid_size=0.1)
        out_test_loader, out_valid_loader = validation_split(
            batch_size=128, test_dataset=cifar10_test, valid_size=0.1)

    #  create validation sets
    ood_datasets.append(('shvn', out_test_loader, out_valid_loader))
    lsun_test_loader, lsun_valid_loader = validation_split(
        batch_size=128, test_dataset=lsun_testset, valid_size=0.1)
    imagenet_test_loader, imagenet_valid_loader = validation_split(
        batch_size=128, test_dataset=imagenet_testset, valid_size=0.1)
    ood_datasets.extend([('lsun', lsun_test_loader, lsun_valid_loader),
                         ('imagenet', imagenet_test_loader,
                          imagenet_valid_loader)])

    net = get_ResNet34(num_c=n_classes, gaussian_layer=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(net, device_ids=[0])
    net.to(device)

    results = np.zeros((4, 3, 5))

    for net_num in range(1, 6):
        #  Laod network
        checkpoint = torch.load(f'./resnet/{dataset}_resnet{net_num}.ckpt.pth')
        if list(checkpoint['net'].keys())[0].split('.')[0] == 'module':
            net.load_state_dict(checkpoint['net'])
        else:
            net.module.load_state_dict(checkpoint['net'])
        layers_precsions, layers_centers = calc_full_covs(
            in_train_loader, n_classes=n_classes, layers=5)

        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader, ood_valid_loader = ood_loaders

            #  train logistic regression
            eps = tune_mahalanobis(
                in_valid_loader, ood_valid_loader,
                layers_precsions, layers_centers)
#             eps = 0.001
            preds_in = predict_preprocess_ensamble(
                in_valid_loader, layers_precsions, layers_centers, eps).cpu()
            preds_ood = predict_preprocess_ensamble(
                ood_valid_loader, layers_precsions, layers_centers, eps).cpu()
            logit_train = torch.cat((preds_ood, preds_in), dim=0).cpu()
            labels = np.concatenate(
                (np.zeros(preds_ood.size(0)), np.ones(preds_in.size(0))))
            regression = LogisticRegressionCV(n_jobs=2)
            regression.fit(logit_train, labels)

            #  predict with feature ensamble and input preprocess
            distances_in_distribution = predict_preprocess_ensamble(
                in_test_loader, layers_precsions, layers_centers, eps).cpu()
            preds_in = regression.predict_proba(
                distances_in_distribution)[:, 1]
            distances_out_distribution = predict_preprocess_ensamble(
                ood_test_loader, layers_precsions, layers_centers, eps).cpu()
            preds_ood = regression.predict_proba(
                distances_out_distribution)[:, 1]

            # TNR level 1
            thres = search_thers(preds_in, 0.95, -1)
            TNR = (preds_ood < thres).sum()/preds_ood.shape[0]
            results[0, data_idx, net_num-1] = TNR
            print(TNR)

            # TNR level 2
            thres = search_thers(preds_in, 0.99, -1)
            TNR = (preds_ood < thres).sum()/preds_ood.shape[0]
            results[1, data_idx, net_num-1] = TNR

            # auroc
            results[2, data_idx, net_num-1] = auroc_score(preds_in, preds_ood)

            # detectuin accuracy
            results[3, data_idx, net_num -
                    1] = detection_accuracy(0, 1, preds_in, preds_ood)[1]

        print(f'finished {net_num} networks')

    mean = results.mean(axis=-1)

    if dataset == 'svhn':
        print(
            f'TNR95: cifar {mean[0, 0]} | lsun {mean[0, 1]} | \
            imagenet {mean[0, 2]}')
        print(
            f'TNR99: cifar {mean[1, 0]} | lsun {mean[1, 1]} | \
            imagenet {mean[1, 2]}')
        print(
            f'AUROC: cifar {mean[2, 0]} | lsun {mean[2, 1]} | \
            imagenet {mean[2, 2]}')
        print(
            f'Detection Accuracy: cifar {mean[3, 0]} | lsun {mean[3, 1]} \
            | imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns=['cifar10', 'lsun', 'imagenet'],
                          index=['TNR95', 'TNR99',
                                 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./Mahalanobis_{dataset}_results.csv')
    if dataset == 'cifar10' or dataset == 'cifar100':
        print(
            f'TNR95: svhn {mean[0, 0]} | lsun {mean[0, 1]} | \
            imagenet {mean[0, 2]}')
        print(
            f'TNR99: svhn {mean[1, 0]} | lsun {mean[1, 1]} | \
            imagenet {mean[1, 2]}')
        print(
            f'AUROC: svhn {mean[2, 0]} | lsun {mean[2, 1]} | \
            imagenet {mean[2, 2]}')
        print(
            f'Detection Accuracy: svhn {mean[3, 0]} | lsun {mean[3, 1]} \
            | imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns=['svhn', 'lsun', 'imagenet'], index=[
                          'TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./Mahalanobis_{dataset}_results.csv')
