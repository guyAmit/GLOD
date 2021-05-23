import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from ..Gaussian_layer import GaussianLayer


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


def calc_params(net, trainloader, n_classes,
                layers, device):
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
