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
    def __init__(self, net, num_classes, pen_representaion_size):
        super(pretrained_to_glod, self).__init__()
        self.gaussian_layer = GaussianLayer(
            input_dim=pen_representaion_size, n_classes=num_classes)
        self.net = net

    def forward(self, x):
        out = self.net.pen_forward(x)
        out = self.gaussian_layer(out)
        return out
