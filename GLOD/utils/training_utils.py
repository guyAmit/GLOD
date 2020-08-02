import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from ..glod import GaussianLayer


def validation_split(batch_size,
                     test_transform,
                     random_seed,
                     test_dataset,
                     valid_size=0.1,
                     num_workers=2,
                     pin_memory=False):
    '''validation_split(batch_size, test_transform, random_seed,
                     test_dataset, valid_size=0.1, num_workers=2,
                     pin_memory=False)
    creates a validation_split from the test data of size 0.1*len(test_set)

    Parameters
    ----------------
    'batch_size': int
        batch size of data loaders
    'test_transform': torchvosion transform
        transform to be applied to the images
    'random_seed': int
        seed for sampeling
    'test_dataset': torch dataset
        the test set that will be splited into validation and test sets
    'valid_size': float
        fraction of data that should be used for validation
    'num_classes': int
        num of processes that should be used in each data loader
    'pin_memory': boolean
        pin_memory parameter of data loader

    Return
    -----------------
    (test_loader, valid_loader) - Two data loaders one for validation and
        one for test
    '''
    num_test = len(test_dataset)
    indices = list(range(num_test))
    split = int(np.floor(valid_size * num_test))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    test_idx, valid_idx = indices[split:], indices[:split]
    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (test_loader, valid_loader)


class pretrained_to_glod(nn.Module):
    '''pretrained_to_glod(net, num_classes, pen_representaion_size)
    converts a pretrained model with a penultimate forward function the a
    GLOD model

    Parameters
    ----------------
    'net': pytorch nn.Module
        the network to be converted Must have a 'pen_forward' function
    'num_classes': int
        number of predicted classes of the model
    'pen_representaion_size': int
        dimension of the penultimate representation

    Return
    -----------------
    a neural network with an untrained glod layer
    '''

    def __init__(self, net, num_classes, pen_representaion_size):
        super(pretrained_to_glod, self).__init__()
        self.gaussian_layer = GaussianLayer(
            input_dim=pen_representaion_size, n_classes=num_classes)
        self.net = net

    def forward(self, x):
        out = self.net.pen_forward(x)
        out = self.gaussian_layer(out)
        return out
