import torch
import torch.nn as nn
import numpy as np


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


class FoodLoss(nn.Module):
    def __init__(self, lambd=0.003):
        super(FoodLoss, self).__init__()
        self.lambd = lambd
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        ce = self.cross_entropy(x, y)
        likelihood = -x.gather(1, y.unsqueeze(1))
        return ce+(self.lambd/x.size(0))*likelihood.sum()


def predict_llr(net, loader, k, device):
    net.eval()
    llr_scores = []
    with torch.no_grad():
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            scores = net.llr_ood_score(inputs, k)
            llr_scores.append(scores)
        llr_scores = torch.cat(llr_scores)
    return llr_scores
