import os
import math
import time
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import random


from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from utils import *

seed = 15
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=test_transform)

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128,
                                                   shuffle=True, num_workers=2)


svhn_train = torchvision.datasets.SVHN(root='./data/SVHN/', split='train',
                                       download=True, transform=test_transform)
svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=128,
                                                shuffle=True, num_workers=2)

svhn_test = torchvision.datasets.SVHN(root='./data/SVHN/', split='test',
                                      download=True, transform=test_transform)

cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True,
                                               download=True, transform=test_transform)

cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=128,
                                                    shuffle=True, num_workers=2)

cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False,
                                              download=True, transform=test_transform)

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


# class TinyImages(torch.utils.data.Dataset):

#     def __init__(self, transform=None):

#         data_file = open('./Outlier_Exposer/tiny_images.bin', "rb")

#         def load_image(idx):
#             data_file.seek(idx * 3072)
#             data = data_file.read(3072)
# #             return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")
#             return  np.frombuffer (data, dtype=np.uint8).reshape(32, 32, 3, order="F")

#         self.load_image = load_image
#         self.offset = 0     # offset index

#         self.transform = transform

#         self.cifar_idxs = []
#         with open('./Outlier_Exposer/80mn_cifar_idxs.txt', 'r') as idxs:
#             for idx in idxs:
#                 # indices in file take the 80mn database to start at 1, hence "- 1"
#                 self.cifar_idxs.append(int(idx) - 1)

#         # hash table option
#         self.cifar_idxs = set(self.cifar_idxs)
#         self.in_cifar = lambda x: x in self.cifar_idxs

#     def __getitem__(self, index):
#         index = (index + self.offset) % 79302016

#         while self.in_cifar(index):
#             index = np.random.randint(79302017)

#         img = self.load_image(index)
#         if self.transform is not None:
#             img = self.transform(img)

#         return img, 0  # 0 is the class

#     def __len__(self):
#         return 79302017

# tiny_images_dataset = TinyImages(transform=test_transform)


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.centers = nn.Parameter(
            0.5 * torch.randn(n_classes, input_dim).cuda())
        self.covs = nn.Parameter(
            0.2 + torch.tensor(np.random.exponential(scale=0.5, size=(n_classes, input_dim))).cuda())

    def forward(self, x):
        covs = self.covs.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        centers = self.centers.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        diff = x.unsqueeze(1).repeat(1, self.n_classes, 1) - centers
        Z_log = -0.5 * torch.sum(torch.log(self.covs), axis=-1) - \
            0.5 * self.input_dim * np.log(2 * np.pi)
        exp_log = -0.5 * \
            torch.sum(diff * (1 / (covs + np.finfo(np.float32).eps))
                      * diff, axis=-1)
        likelihood = Z_log + exp_log
        return likelihood

    def clip_convs(self):
        with torch.no_grad():
            self.covs.clamp_(np.finfo(np.float32).eps)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    #         self.gaussian_layer = GaussianLayer(input_dim=512, n_classes=num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def Resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def Resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class Resnet_GaussianLayer(nn.Module):
    def __init__(self, net, num_classes=10):
        super(Resnet_GaussianLayer, self).__init__()
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
        return llr

        # function to extact the multiple features

    def feature_list(self, x):
        feature_list = list(self.net.children())
        feature5 = feature_list[5]
        feature4 = feature_list[4]
        feature3 = feature_list[3]
        feature2 = feature_list[2]
        #feature1 = nn.Sequential(*feature_list[:2])
        feature1 = feature_list[1]
        feature0 = feature_list[0]

        out_list = []
        out = feature0(x)
        out_list.append(F.max_pool2d(out, 32).view(out.size(0), -1))
        out = feature1(out)
        out_list.append(F.max_pool2d(out, 32).view(out.size(0), -1))
        out = feature2(out)
        out_list.append(F.max_pool2d(out, 32).view(out.size(0), -1))
        out = feature3(out)
        out_list.append(F.avg_pool2d(out, 16).view(out.size(0), -1))
        out = feature4(out)
        out_list.append(F.avg_pool2d(out, 4).view(out.size(0), -1))

        out = feature5(out)
        out = F.avg_pool2d(out, 4).view(out.size(0), -1)
        out = self.gaussian_layer(out)
        out_list.append(out)

        return out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        feature_list = list(self.net.children())
        feature5 = feature_list[5]
        feature4 = feature_list[4]
        feature3 = feature_list[3]
        feature2 = feature_list[2]
        feature1 = feature_list[1]
        feature0 = feature_list[0]

        out = feature0(x)
        if layer_index == 0:
            out = F.max_pool2d(out, 32).view(out.size(0), -1)
            return out
        out = feature1(out)
        if layer_index == 1:
            out = F.max_pool2d(out, 32).view(out.size(0), -1)
            return out
        out = feature2(out)
        if layer_index == 2:
            out = F.max_pool2d(out, 32).view(out.size(0), -1)
            return out
        out = feature3(out)
        if layer_index == 3:
            out = F.avg_pool2d(out, 16).view(out.size(0), -1)
            return out
        out = feature4(out)
        if layer_index == 4:
            out = F.avg_pool2d(out, 4).view(out.size(0), -1)
            return out


def calc_params(net, trainloader, n_classes, layers):
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
            covs = torch.zeros(n_classes, x_dim).cuda()
            for c in range(n_classes):
                class_points = outputs[c == target_list]
                centers[c] = torch.mean(class_points, axis=0)
                covs[c] = torch.var(class_points, axis=0)
            precision = torch.div(1.0, covs + np.finfo(np.float32).eps)
            layers_centers.append(centers)
            layers_precisions.append(precision)
    return layers_precisions, layers_centers


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
                                           layers_precsions, layers_centers, n_classes)
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


def gen_adversarial_batch_bound(net, inputs, labels, n_classes, init_eps, bound):
    net.eval()
    pertubed_inputs = []
    samples_reached = 0
    input_shape = (1, *(inputs[0].shape))
    for idx in range(inputs.shape[0]):
        sample = torch.autograd.Variable(inputs[idx].cuda(device).reshape(*input_shape),
                                         requires_grad=True)
        eps = init_eps
        lowest_llr = np.inf
        likelihood = net(sample)
        prediction = likelihood.max(1)[1]
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
