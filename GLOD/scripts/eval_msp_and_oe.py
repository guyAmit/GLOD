import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegressionCV
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


def predict(loader):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            predictions.append(outputs.max(1)[0])
    predictions = torch.cat(predictions).cuda()
    return predictions


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
        #  msp
        checkpoint = torch.load(f'./resnet/{dataset}_resnet{net_num}.ckpt.pth')
        #  OE
        # checkpoint = torch.load(f'./Outlier_Exposer/{dataset}/{dataset}_resnet{net_num}.ckpt.pth')

        if list(checkpoint['net'].keys())[0].split('.')[0] == 'module':
            net.load_state_dict(checkpoint['net'])
        else:
            net.module.load_state_dict(checkpoint['net'])

        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_test_loader, ood_valid_loader = ood_loaders

            # predict
            preds_in = predict(in_test_loader).cpu()
            preds_ood = predict(ood_test_loader).cpu()

            # TNR level 1
            thres = search_thers(preds_in, 0.95, 0)
            TNR = (preds_ood < thres).sum().item()/preds_ood.size(0)
            results[net_num-1, 0, data_idx] = TNR

            # TNR level 2
            thres = search_thers(preds_in, 0.99, 0)
            TNR = (preds_ood < thres).sum().item()/preds_ood.size(0)
            results[net_num-1, 1, data_idx] = TNR

            # auroc
            results[net_num-1, 2, data_idx] = auroc_score(preds_in, preds_ood)

            # detectuin accuracy
            results[net_num-1, 3,
                    data_idx] = detection_accuracy(-3, 10,
                                                   preds_in, preds_ood)[1]

        print(f'finished {net_num} networks')
    mean = results.mean(axis=0)

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
        df.to_csv(f'./MSP_{dataset}_results.csv')
