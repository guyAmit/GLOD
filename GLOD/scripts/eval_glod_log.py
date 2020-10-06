import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parameter import Parameter
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import random


# seed = 15
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

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


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.centers = nn.Parameter(0.5*torch.randn(n_classes, input_dim).cuda())
        self.log_covs = nn.Parameter(torch.zeros(n_classes, input_dim).cuda())
        
    def forward(self, x):
        exp_log_covs = torch.exp(self.log_covs)
        covs = exp_log_covs.unsqueeze(0).expand(x.size(0), self.n_classes, self.input_dim)
        centers = self.centers.unsqueeze(0).expand(x.size(0), self.n_classes, self.input_dim)
        diff = x.unsqueeze(1).repeat(1, self.n_classes ,1) - centers
        Z_log = -0.5*torch.sum(torch.log(exp_log_covs+np.finfo(np.float32).eps),
                               axis=-1) -0.5*self.input_dim*np.log(2*np.pi)
        exp_log = -0.5*torch.sum(diff*(1/(covs+np.finfo(np.float32).eps))*diff, axis=-1)
        likelihood = Z_log+exp_log
        return likelihood


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
#         self.gaussian_layer = GaussianLayer(input_dim=512, n_classes=num_classes)

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


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes= num_classes)
        
class ResNet_GaussianLayer(nn.Module):
    def __init__(self, net, num_classes=10):
        super(ResNet_GaussianLayer, self).__init__()
        self.net = nn.Sequential(*list(net.children())[:-1])
        self.avgpool = nn.AvgPool2d(4) if 'avgpool' not in net.state_dict().keys() else None
        self.gaussian_layer = GaussianLayer(input_dim=512, n_classes=num_classes)
        
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
        feature5 = list(self.net.children())[5]
        feature4 = list(self.net.children())[4]
        feature3 = list(self.net.children())[3]
        feature2 = list(self.net.children())[2]
        feature1 = nn.Sequential(*list(self.net.children())[:2])
        
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
        feature5 = list(self.net.children())[5]
        feature4 = list(self.net.children())[4]
        feature3 = list(self.net.children())[3]
        feature2 = list(self.net.children())[2]
        feature1 = nn.Sequential(*list(self.net.children())[:2])
        
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
        
def get_test_valid_loader(batch_size,
                           test_dataset,
                           valid_size=0.1,
                           num_workers=2,
                           pin_memory=False):    

    num_test = len(test_dataset)
    indices = list(range(num_test))
    split = int(np.floor(valid_size * num_test))
    
    np.random.seed(5)
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



def search_thers(preds_in, level):
    step = 10
    val = -100
    eps = 0.0001
    while True:
        TNR = (preds_in>=val).sum()/preds_in.shape[0]
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
    max_det=0
    max_thres = val
    while val<end:
        TPR_in = (preds_in >= val).sum()/preds_in.shape[0]
        TPR_out = (preds_ood <= val).sum()/preds_ood.shape[0]
        detection = (TPR_in+TPR_out)/2
        if detection > max_det:
            max_det = detection
            max_thres = val
        val += step
    return max_thres, max_det


def predict_llr(model, loader, device, k):
    model.eval()
    llr_scores = []
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scores = model.llr_ood_score(inputs, k)
            llr_scores.append(scores)
        llr_scores = torch.cat(llr_scores)
    return llr_scores


if __name__ == '__main__':
    dataset = 'cifar100'

    ood_datasets = []
    if dataset == 'cifar10':
        ood_dataset = 'svhn'
        n_classes = 10
        in_train_set = cifar10_train_loader
        in_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=128,
                                          shuffle=False, num_workers=2)
        out_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=128,
                                          shuffle=False, num_workers=2)
    if dataset == 'cifar100':
        ood_dataset = 'svhn'
        n_classes = 100
        in_train_loader = cifar100_train_loader
        in_test_loader =  torch.utils.data.DataLoader(cifar100_test, batch_size=128,
                                          shuffle=False, num_workers=2)
        out_test_loader =  torch.utils.data.DataLoader(svhn_test, batch_size=128,
                                          shuffle=False, num_workers=2)
    if dataset == 'svhn':
        ood_dataset = 'cifar10'
        n_classes = 10
        in_train_loader = svhn_train_loader
        in_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=128,
                                          shuffle=False, num_workers=2)
        out_test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=128,
                                          shuffle=False, num_workers=2)
    #  create validation sets
    ood_datasets.append((ood_dataset, out_test_loader))
    lsun_test_loader =  torch.utils.data.DataLoader(lsun_testset, batch_size=128, shuffle=False, num_workers=2)
    imagenet_test_loader = torch.utils.data.DataLoader(imagenet_testset, batch_size=128, shuffle=False, num_workers=2)
    ood_datasets.extend([('lsun', lsun_test_loader), ('imagenet', imagenet_test_loader)])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResNet34(n_classes)

    net = ResNet_GaussianLayer(net, num_classes=n_classes)
    net = net.to(device)

#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)
        
    results = np.zeros((4, 3, 5))

    for net_num in range(1, 6):
        #  Laod network
        checkpoint = torch.load(f'./GLOD_models/log_cov/{dataset}_GLOD{net_num}.ckpt', map_location=device)['net']
    
        net.load_state_dict(checkpoint)
        
        llr_in_distribution = predict_llr(model=net.module,
                              loader=in_test_loader,
                              device=device,
                              k=n_classes).detach().cpu().numpy()
        
        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader = ood_loaders
            
            
            llr_out_distribution = predict_llr(model=net.module,
                                               loader=ood_test_loader,
                                               device=device,
                                               k=n_classes).detach().cpu().numpy()  

            thres99 = search_thers(llr_in_distribution, 0.99)
            thres95 = search_thers(llr_in_distribution, 0.95)
            
            # TNR level 1
            TNR95 = (llr_out_distribution < thres95).sum()/llr_out_distribution.shape[0]
            results[0, data_idx, net_num-1] = TNR95
            print(TNR95)

            # TNR level 2
            TNR99 = (llr_out_distribution < thres99).sum()/llr_out_distribution.shape[0]
            results[1, data_idx, net_num-1] = TNR99
            
            # auroc
            preds = np.concatenate((llr_out_distribution, llr_in_distribution))
            y_true = np.concatenate((np.zeros(llr_out_distribution.shape[0]), np.ones(llr_in_distribution.shape[0])))
            results[2, data_idx, net_num-1] = roc_auc_score(y_true, preds)
            
            # detectuin accuracy
            results[3, data_idx, net_num-1] = detection_accuracy(-5000, 10000,
                                                                 llr_in_distribution,
                                                                 llr_out_distribution)[1]
            
            
        print(f'finished {net_num} networks')
        
    mean = results.mean(axis=-1)
    
    if dataset == 'svhn':
        print(f'TNR95: cifar {mean[0 ,0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: cifar {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1 , 2]}')
        print(f'AUROC: cifar {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: cifar {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['cifar10', 'lsun', 'imagenet'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./GLOD_log_{dataset}_results.csv')
    if dataset == 'cifar10' or dataset == 'cifar100':
        print(f'TNR95: svhn {mean[0, 0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: svhn {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1, 2]}')
        print(f'AUROC: svhn {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: svhn {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['svhn', 'lsun', 'imagenet'],
                          index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./GLOD_log_{dataset}_results.csv')
        
        
        
















