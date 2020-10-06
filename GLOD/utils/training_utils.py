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


class ResNet_GaussianLayer(nn.Module):
    def __init__(self, net, num_classes=10):
        '''
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
        super(ResNet_GaussianLayer, self).__init__()
        self.net = nn.Sequential(*list(net.children())[:-1])
        self.avgpool = nn.AvgPool2d(
            4) if 'avgpool' not in net.state_dict().keys() else None
        self.gaussian_layer = GaussianLayer(
            input_dim=512, n_classes=num_classes)

        def forward(self, x):
            out = self.net(x)
            if self.avgpool is not None:
                out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.gaussian_layer(out)
            return out

        def penultimate_forward(self, x):
            out = self.net(x)
            if self.avgpool is not None:
                out = self.avgpool(out)
            return out.view(out.size(0), -1)

        def llr_ood_score(self, x, k=100):
            preds = self(x)
            topk = torch.topk(preds, k, dim=1)[0]
            llr = topk[:, 0] - topk[:, 1:k].mean(1)
            return llr.to('cpu')

    # function to extact the multiple features
        def feature_list(self, x):
            feature_list = list(self.net.children())
            feature5 = feature_list[5]
            feature4 = feature_list[4]
            feature3 = feature_list[3]
            feature2 = feature_list[2]
            feature1 = nn.Sequential(*feature_list[:2])

            out_list = []
            out = feature1(x)
            out_list.append(F.avg_pool2d(out, 32).view(out.size(0), -1))
            out = feature2(out)
            out_list.append(F.avg_pool2d(out, 32).view(out.size(0), -1))
            out = feature3(out)
            out_list.append(F.avg_pool2d(out, 16).view(out.size(0), -1))
            out = feature4(out)
            out_list.append(F.avg_pool2d(out, 8).view(out.size(0), -1))
            out = feature5(out)
            out = F.avg_pool2d(out, 4).view(out.size(0), -1)
            out_list.append(out)

            return out_list

        # function to extact a specific feature
        def intermediate_forward(self, x, layer_index):
            feature_list = list(self.net.children())
            feature5 = feature_list[5]
            feature4 = feature_list[4]
            feature3 = feature_list[3]
            feature2 = feature_list[2]
            feature1 = nn.Sequential(*feature_list[:2])

            out = feature1(x)
            if layer_index == 0:
                out = F.avg_pool2d(out, 32).view(out.size(0), -1)
                return out
            out = feature2(out)
            if layer_index == 1:
                out = F.avg_pool2d(out, 32).view(out.size(0), -1)
                return out
            out = feature3(out)
            if layer_index == 2:
                out = F.avg_pool2d(out, 16).view(out.size(0), -1)
                return out
            out = feature4(out)
            if layer_index == 3:
                out = F.avg_pool2d(out, 8).view(out.size(0), -1)
                return out
            out = feature5(out)
            if layer_index == 4:
                out = F.avg_pool2d(out, 4).view(out.size(0), -1)
                return out


def calc_gaussian_params(model, loader, device, n_classes):
    '''
    Calc manualy GLOD params.
    '''
    outputs_list = []
    target_list = []
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.penultimate_forward(inputs)
            outputs_list.append(outputs)
            target_list.append(targets)
        outputs = torch.cat(outputs_list, axis=0)
        target_list = torch.cat(target_list)
        x_dim = outputs.size(1)
        centers = torch.zeros(n_classes, x_dim).cuda()
        covs = torch.zeros(n_classes, x_dim).cuda()
        for c in range(n_classes):
            class_points = outputs[c == target_list]
            centers[c] = torch.mean(class_points, axis=0)
            covs[c] = torch.var(class_points, axis=0)
        return covs, centers
