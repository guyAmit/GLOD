import os
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010)), ])

seed = 15
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True,
                                            transform=test_transform)

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True,
                                             transform=test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train,
                                                   batch_size=128,
                                                   shuffle=True, num_workers=2)


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

iSUN_path = os.path.expanduser('/home/guy5/Likelihood_model/iSUN')
iSUN_testset = torchvision.datasets.ImageFolder(
    root=iSUN_path, transform=test_transform)

imagenet_path = os.path.expanduser(
    '/home/guy5/Likelihood_model/Imagenet_resize')
imagenet_testset = torchvision.datasets.ImageFolder(
    root=imagenet_path, transform=test_transform)
