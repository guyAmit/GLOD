import os
import math
import random
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


from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

seed = 15
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])


cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=test_transform)

cifar10_train = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128,
                                                   shuffle=True, num_workers=2)


svhn_train = torchvision.datasets.SVHN(root='../data/SVHN/', split='train',
                                       download=True, transform=test_transform)
svhn_train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=128,
                                                shuffle=True, num_workers=2)

svhn_test = torchvision.datasets.SVHN(root='../data/SVHN/', split='test',
                                      download=True, transform=test_transform)

cifar100_train = torchvision.datasets.CIFAR100(root='../data', train=True,
                                               download=True, transform=test_transform)

cifar100_train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=128,
                                                    shuffle=True, num_workers=2)

cifar100_test = torchvision.datasets.CIFAR100(root='../data', train=False,
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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
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
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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
        y = self.linear(out)
        return y

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(F.avg_pool2d(out, 32).view(out.size(0), -1))
        out = self.layer1(out)
        out_list.append(F.avg_pool2d(out, 32).view(out.size(0), -1))
        out = self.layer2(out)
        out_list.append(F.avg_pool2d(out, 16).view(out.size(0), -1))
        out = self.layer3(out)
        out_list.append(F.avg_pool2d(out, 8).view(out.size(0), -1))
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4).view(out.size(0), -1)
        out_list.append(out)
        return out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 0:
            out = F.avg_pool2d(out, 32).view(out.size(0), -1)
        if layer_index == 1:
            out = F.avg_pool2d(self.layer1(out), 32).view(out.size(0), -1)
        elif layer_index == 2:
            out = self.layer1(out)
            out = F.avg_pool2d(self.layer2(out), 16).view(out.size(0), -1)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = F.avg_pool2d(self.layer3(out), 8).view(out.size(0), -1)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(self.layer4(out), 4).view(out.size(0), -1)
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def Resnet34(num_c):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_c)


def get_test_valid_loader(batch_size,
                          test_dataset,
                          random_seed,
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
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory, shuffle=False,
    )

    return (test_loader, valid_loader)


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
                    class_points-centers[c].unsqueeze(0).expand(class_points.size(0), -1))
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


def predict_mahalanobis_ensamble(net, loader, layers_precsions, layers_centers):
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
                                                          n_classes=n_classes, layers=5)

        mahalanobis_in_dist = predict_mahalanobis_ensamble(net, in_test_loader,
                                                           layers_precsions,
                                                           layers_centers).cpu().numpy()

        preds_in = mahalanobis_in_dist.sum(1, keepdims=True)
        scaler = MinMaxScaler().fit(preds_in)

        start_time = time.time()
        mahalanobis_in_dist = predict_mahalanobis_ensamble(net, in_test_loader,
                                                           layers_precsions,
                                                           layers_centers)

        preds_in = mahalanobis_in_dist.sum(1, keepdims=True).cpu().numpy()
        preds_in = scaler.transform(preds_in)
        times.append(time.time()-start_time)

        thres95 = search_thers(preds_in, 0.95)
        thres99 = search_thers(preds_in, 0.99)

        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader, ood_valid_loader = ood_loaders

            #  predict with feature ensamble and input preprocess
            mahalanobis_ood = predict_mahalanobis_ensamble(net, ood_test_loader,
                                                           layers_precsions,
                                                           layers_centers).cpu().numpy()
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
