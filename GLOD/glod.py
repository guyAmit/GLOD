import torch
import torch.nn as nn
import numpy as np


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.centers = nn.Parameter(
            0.5*torch.randn(n_classes, input_dim).cuda())
        self.covs = nn.Parameter(
            0.2+torch.tensor(np.random.exponential(scale=0.3,
                                                   size=(n_classes,
                                                         input_dim))).cuda())

    def forward(self, x):
        covs = self.covs.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        centers = self.centers.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        diff = x.unsqueeze(1).repeat(1, self.n_classes, 1) - centers
        Z_log = -0.5*torch.sum(torch.log(self.covs+np.finfo(np.float32).eps),
                               axis=-1) - \
            0.5*self.input_dim*np.log(2*np.pi)
        exp_log = -0.5 * \
            torch.sum(diff*(1/(covs+np.finfo(np.float32).eps))*diff, axis=-1)
        likelihood = Z_log+exp_log
        return likelihood

    def clip_convs(self):
        '''
        Cliping the convariance matricies to be alaways positive. \n
        Use: call after optimizer.step()
        '''
        with torch.no_grad():
            self.covs.clamp_(np.finfo(np.float32).eps)

    def cov_regulaizer(self, beta=0.01):
        '''
        Covarianvce regulzer \n
        Use: add to the loss if used for OOD detection
        '''
        return beta*(torch.norm(self.covs, p=2))


class GlodLoss(nn.Module):
    def __init__(self, lambd=0.003):
        super(GlodLoss, self).__init__()
        self.lambd = lambd
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        ce = self.cross_entropy(x, y)
        likelihood = -x.gather(1, y.unsqueeze(1))
        return ce+(self.lambd/x.size(0))*likelihood.sum()


def calc_llr(preds, k):
    top_k = preds.topk(k).values.squeeze()
    avg_ll = np.mean(top_k[:, 1:k].cpu().detach().numpy())
    llr = top_k[:, 0].cpu()-avg_ll
    return llr
