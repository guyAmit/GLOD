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
    step = 1
    val = -1
    eps = 0.001
    for _ in  range(1000):
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

def calc_full_covs(trainloader, n_classes, layers):
    layers_centers = []
    layers_precisions = []
    for l in range(layers):
        outputs_list = []
        target_list = []
        with torch.no_grad():
            for (inputs, targets) in trainloader:
                inputs, targets = inputs, targets#.to(device),.to(device)
                outputs = net.module.intermediate_forward(inputs, layer_index=l)
                outputs_list.append(outputs)
                target_list.append(targets)
            outputs = torch.cat(outputs_list, axis=0)
            target_list = torch.cat(target_list)
            x_dim = outputs.size(1)
            centers = torch.zeros(n_classes, x_dim)#.cuda()
            normlized_outputs = []
            for c in range(n_classes):
                class_points = outputs[c == target_list]
                centers[c] = torch.mean(class_points, axis=0)
                normlized_outputs.append(class_points-centers[c].unsqueeze(0).expand(class_points.size(0), -1))
            normlized_outputs = torch.cat(normlized_outputs, axis=0).cpu()
            covs_lasso = EmpiricalCovariance(assume_centered=False)
            covs_lasso.fit(normlized_outputs.cpu().numpy())
            precision = torch.from_numpy(covs_lasso.precision_).float()#.cuda()
            layers_centers.append(centers)
            layers_precisions.append(precision)
    return layers_precisions, layers_centers

def calc_mahalanobis(x, precsion, centers):
    distance = torch.zeros(x.size(0), centers.size(0))#.cuda()
    for c in range(centers.size(0)):
        diff = x - centers[c].unsqueeze(0).expand(x.size(0), -1)
        exp_log = -torch.mm(torch.mm(diff,precsion),diff.t()).diag()
        distance[:, c] = exp_log
    return distance


def predict_ensamble_batch(inputs, layers_precsions, layers_centers):
    f_list = net.module.feature_list(inputs)
    preds = torch.zeros(inputs.size(0), len(layers_centers))#.cuda()
    for l in range(len(layers_centers)):
        preds[:, l] = calc_mahalanobis(f_list[l], layers_precsions[l], layers_centers[l]).max(1)[0]
    return preds

def predict_mahalanobis_ensamble(loader, layers_precsions, layers_centers):
    net.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if len(inputs)==2:
                inputs = inputs[0]
            if type(inputs) == list:
                inputs = inputs[0]
            inputs = inputs#.to(device)
            preds = predict_ensamble_batch(inputs, layers_precsions, layers_centers)
            predictions.append(preds)
    predictions = torch.cat(predictions)#.cuda()
    return predictions


def gen_adversarial_batch(inputs, precsion, centers, eps):
    inputs = torch.autograd.Variable(inputs, requires_grad=True)#.cuda(device)
    outputs = net.module.penultimate_forward(inputs)
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
    return pertubed_inputs.to('cpu')


def predict_mahalanobis_preprocess_ensamble(loader, layers_precsions, layers_centers, eps):
    net.eval()
    predictions = []
    for batch_idx, inputs in enumerate(loader):
        if len(inputs)==2:
            inputs = inputs[0]
        if type(inputs) == list:
            inputs = inputs[0]
        adv_inputs = gen_adversarial_batch(inputs, layers_precsions[len(layers_centers)-1], layers_centers[len(layers_centers)-1], eps)
        with torch.no_grad():
            adv_mahalanobis = predict_ensamble_batch(adv_inputs, layers_precsions, layers_centers)#.cuda()
        predictions.append(adv_mahalanobis)
    predictions = torch.cat(predictions).detach().cpu()
    return predictions    

def create_adv_loader(loader, layers_precsions, layers_centers, eps):
    net.eval()
    adv_samples = []
    for batch_idx, (inputs, _) in enumerate(loader):
        adv_inputs = gen_adversarial_batch(inputs, layers_precsions[len(layers_centers)-1], layers_centers[len(layers_centers)-1], eps)
        adv_samples.append(adv_inputs)
#         torch.cuda.empty_cache()
    
    adv_samples = torch.cat(adv_samples).to('cpu')
    fake_labels = torch.zeros(adv_samples.size(0))
    adv_dataset = torch.utils.data.TensorDataset(adv_samples, fake_labels)
    adv_loader =  torch.utils.data.DataLoader(adv_dataset, batch_size=128,
                                          shuffle=True, num_workers=4)
    return adv_loader



def tune_mahalanobis(in_valid_loader, ood_valid_loader,
                     layers_precsions, layers_centers):
    M_list = [0.0, 0.01, 0.005, 0.0035, 0.002, 0.0014, 0.001, 0.0005]
    best_conf = ()
    best_tnr = -np.inf
    
    for eps in M_list:
        preds_in = predict_mahalanobis_preprocess_ensamble(in_valid_loader,
                                                           layers_precsions,
                                                           layers_centers, eps).cpu() 
        preds_ood = predict_mahalanobis_preprocess_ensamble(ood_valid_loader,
                                                            layers_precsions,
                                                            layers_centers, eps).cpu()
        logit_train = torch.cat((preds_ood[:int(preds_ood.size(0)/2)],
                                 preds_in[:int(preds_in.size(0)/2)]), dim=0).cpu()
        labels = np.concatenate((np.zeros(int(preds_ood.size(0)/2)),
                                 np.ones(int(preds_in.size(0)/2))))
        regression = LogisticRegressionCV(n_jobs=2)
        regression.fit(logit_train, labels)

        preds_in = predict_mahalanobis_preprocess_ensamble(in_valid_loader, 
                                                           layers_precsions, 
                                                           layers_centers, eps).cpu() 
        preds_ood = predict_mahalanobis_preprocess_ensamble(ood_valid_loader, 
                                                            layers_precsions,
                                                            layers_centers, eps).cpu()

        val_preds_in = regression.predict_proba(preds_in[int(preds_in.size(0)/2):])[:, 1]
        val_preds_ood = regression.predict_proba(preds_ood[int(preds_ood.size(0)/2):])[:, 1]
    
        thres = search_thers(val_preds_in, 0.95)
        TNR = (val_preds_ood < thres).sum()/val_preds_ood.shape[0]    
        print(f'Epsilon: {eps} TNR: {TNR}')
        if TNR > best_tnr:
            best_tnr = TNR
            best_conf = eps
    return best_conf


if __name__ == '__main__':
    dataset = 'cifar100'

    ood_datasets = []
    if dataset == 'cifar10':
        n_classes = 10
        in_train_loader = cifar10_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar10_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'cifar100':
        n_classes = 100
        in_train_loader = cifar100_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar100_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
    if dataset == 'svhn':
        n_classes = 10
        in_train_loader = svhn_train_loader
        in_test_loader, in_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=svhn_test, valid_size=0.1)
        out_test_loader, out_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=cifar10_test, valid_size=0.1) 

    #  create validation sets
    ood_datasets.append(('shvn', out_test_loader, out_valid_loader))
    lsun_test_loader, lsun_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=lsun_testset, valid_size=0.1)
    imagenet_test_loader, imagenet_valid_loader = get_test_valid_loader(batch_size=128, test_dataset=imagenet_testset, valid_size=0.1)
    ood_datasets.extend([('lsun', lsun_test_loader, lsun_valid_loader), ('imagenet', imagenet_test_loader, imagenet_valid_loader)])

    net = create_Resnet34(num_c=n_classes)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net =  nn.DataParallel(net) 
#                            device_ids=[0])
#     net.to(device)
        
    results = np.zeros((4, 3, 5))

    for net_num in range(1, 6):
        #  Laod network
        checkpoint = torch.load(f'./resnet/{dataset}_resnet{net_num}.ckpt.pth')
        if list(checkpoint['net'].keys())[0].split('.')[0] == 'module':
            net.load_state_dict(checkpoint['net'])
        else:
            net.module.load_state_dict(checkpoint['net'])
        layers_precsions, layers_centers = calc_full_covs(in_train_loader, n_classes=n_classes, layers=5)
        adv_loader = create_adv_loader(in_valid_loader, layers_precsions, layers_centers, 0.1)
            #  train logistic regression
        eps = tune_mahalanobis(in_valid_loader, adv_loader, layers_precsions, layers_centers)
            
        for data_idx, ood_loaders in enumerate(ood_datasets):
            ood_data_name, ood_test_loader, ood_valid_loader = ood_loaders
           
            #             eps = 0.001
            preds_in = predict_mahalanobis_preprocess_ensamble(in_valid_loader, layers_precsions, layers_centers, eps).cpu() 
            preds_ood = predict_mahalanobis_preprocess_ensamble(adv_loader, layers_precsions, layers_centers, eps).cpu()
            logit_train = torch.cat((preds_ood, preds_in), dim=0).cpu()
            labels = np.concatenate((np.zeros(preds_ood.size(0)), np.ones(preds_in.size(0))))
            regression = LogisticRegressionCV(n_jobs=2)
            regression.fit(logit_train, labels)

            #  predict with feature ensamble and input preprocess
            mahalanobis_distances_in_distribution = predict_mahalanobis_preprocess_ensamble(in_test_loader, layers_precsions, layers_centers, eps).cpu()
            preds_in = regression.predict_proba(mahalanobis_distances_in_distribution)[:, 1]
            mahalanobis_distances_out_distribution = predict_mahalanobis_preprocess_ensamble(ood_test_loader, layers_precsions, layers_centers, eps).cpu()
            preds_ood = regression.predict_proba(mahalanobis_distances_out_distribution)[:, 1]
            
             # TNR level 1
            thres = search_thers(preds_in, 0.95)
            TNR = (preds_ood < thres).sum()/preds_ood.shape[0]
            results[0, data_idx, net_num-1] = TNR
            print(TNR)

            # TNR level 2
            thres = search_thers(preds_in, 0.99)
            TNR = (preds_ood < thres).sum()/preds_ood.shape[0]
            results[1, data_idx, net_num-1] = TNR
            
            # auroc
            y_true = np.concatenate((np.zeros(preds_ood.shape[0]), np.ones(preds_in.shape[0])))
            preds = np.concatenate((preds_ood, preds_in))
            results[2, data_idx, net_num-1] = roc_auc_score(y_true, preds)
            
            # detectuin accuracy
            results[3, data_idx, net_num-1] = detection_accuracy(0, 1, preds_in, preds_ood)[1]
            
            del adv_loader.dataset, regression
#             torch.cuda.empty_cache()
        print(f'finished {net_num} networks')
        
    mean = results.mean(axis=-1)
    
    if dataset == 'svhn':
        print(f'TNR95: cifar {mean[0 ,0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: cifar {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1 , 2]}')
        print(f'AUROC: cifar {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: cifar {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['cifar10', 'lsun', 'imagenet'], index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./Mahalanobis_{dataset}_results.csv')
    if dataset == 'cifar10' or dataset == 'cifar100':
        print(f'TNR95: svhn {mean[0, 0]}  |  lsun {mean[0, 1]}  |  imagenet {mean[0, 2]}')
        print(f'TNR99: svhn {mean[1, 0]}  |  lsun {mean[1, 1]}  |  imagenet {mean[1, 2]}')
        print(f'AUROC: svhn {mean[2, 0]}  |  lsun {mean[2, 1]}  |  imagenet {mean[2, 2]}')
        print(f'Detection Accuracy: svhn {mean[3, 0]}  |  lsun {mean[3, 1]}  |  imagenet {mean[3, 2]}')
        df = pd.DataFrame(mean, columns = ['svhn', 'lsun', 'imagenet'], index=['TNR95', 'TNR99', 'AUROC', 'Detection Accuracy'])
        df.to_csv(f'./Mahalanobis_{dataset}_results.csv')
        
        
        
















