import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd


test_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])


cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)

cifar10_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=test_transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128,
                                          shuffle=True, num_workers=2)
                                                 

svhn_train  = torchvision.datasets.SVHN(root='./data/SVHN/', split='train',
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
lsun_testset = torchvision.datasets.ImageFolder(root=lsun_path, transform=test_transform)


imagenet_path = os.path.expanduser('/home/guy5/Likelihood_model/Imagenet_resize')
imagenet_testset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=test_transform)


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
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
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
    

def create_Resnet34(num_c):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)

def get_test_valid_loader(batch_size,
                           test_dataset,
                           valid_size=0.1,
                           num_workers=2,
                           pin_memory=False):    

    num_test = len(test_dataset)
    indices = list(range(num_test))
    split = int(np.floor(valid_size * num_test))

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


def search_thers(preds, level):
    step = 1
    val = -1
    eps = 0.001
    while True:
        TNR = (preds>=val).sum().item()/preds.size(0)
        if TNR < level+eps and TNR > level-eps:
            return val
        elif TNR > level:
            val += step
        else:
            val = val - step
            step = step*0.1
    return val

def detection_accuracy(start,end, preds_in, preds_ood):
    step = (end-start)/10000
    val = start
    max_det=0
    max_thres = val
    while val<end:
        TPR_in = (preds_in>=val).sum().item()/preds_in.size(0)
        TPR_out =(preds_ood<=val).sum().item()/preds_ood.size(0)
        detection = (TPR_in+TPR_out)/2
        if detection > max_det:
            max_det = detection
            max_thres = val
        val += step
    return max_thres, max_det


def predict(loader):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            inputs= inputs.to(device)
            outputs = net(inputs)
            predictions.append(outputs.max(1)[0])
    predictions = torch.cat(predictions).cuda()
    return predictions


if __name__ == '__main__':
    dataset = 'cifar100'

    ood_datasets = []
    if dataset == 'cifar10':
        n_classes = 10
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar10_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'cifar100':
        n_classes = 100
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar100_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'svhn':
        n_classes = 10
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar10_test, valid_size=0.1) 

    #  create validation sets
    ood_datasets.append((out_test_loader, out_valid_loader))
    lsun_test_loader, lsun_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=lsun_testset, valid_size=0.1)
    imagenet_test_loader, imagenet_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=imagenet_testset, valid_size=0.1)
    ood_datasets.extend([(lsun_test_loader, lsun_valid_loader), (imagenet_test_loader, imagenet_valid_loader)])

    net = create_Resnet34(num_c=n_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net =  nn.DataParallel(net, device_ids=[0])
    net.to(device)
    results = np.zeros((5, 4, 3))

    for net_num in range(1, 6):
        #  Laod network
        checkpoint = torch.load(f'./Outlier_Exposer/{dataset}/{dataset}_resnet{net_num}.ckpt.pth')
        if list(checkpoint['net'].keys())[0].split('.')[0] == 'module':
            net.load_state_dict(checkpoint['net'])
        else:
            net.module.load_state_dict(checkpoint['net'])
        
        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_test_loader, _ = ood_loaders
            
            #  predict oe
            preds_in = predict(in_test_loader).cpu()
            preds_ood = predict(ood_test_loader).cpu()
            
            thres = search_thers(preds_in, 0.95)
            TNR = (preds_ood < thres).sum().item()/preds_ood.size(0)
            results[net_num-1, 0, data_idx] = TNR

            # TNR level 2
            thres = search_thers(preds_in, 0.99)
            TNR = (preds_ood < thres).sum().item()/preds_ood.size(0)
            results[net_num-1, 1, data_idx] = TNR
            
            # auroc
            y_true = np.concatenate((np.zeros(preds_ood.size(0)), np.ones(preds_in.size(0))))
            preds = np.concatenate((preds_ood.numpy(), preds_in.numpy()))
            results[net_num-1, 2, data_idx] = roc_auc_score(y_true, preds)
            
            # detectuin accuracy
            results[net_num-1, 3, data_idx] = detection_accuracy(-3, 10, preds_in, preds_ood)[1]
            
        print(f'finished {net_num} networks')
    mean = results.mean(axis=0)
    if dataset == 'svhn':
        print(f'TNR95: cifar {mean[0 ,0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: cifar {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1 , 2]}')
        print(f'AUROC: cifar {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: cifar {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['cifar10', 'lsun', 'imagenet'], index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./OE_{dataset}_results.csv')
    if dataset == 'cifar10' or dataset == 'cifar100':
        print(f'TNR95: svhn {mean[0, 0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: svhn {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1, 2]}')
        print(f'AUROC: svhn {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: svhn {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['svhn', 'lsun', 'imagenet'], index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./OE_{dataset}_results.csv')
        
        
















