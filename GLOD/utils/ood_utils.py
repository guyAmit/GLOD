import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def search_thers(preds_in, level):
    step = 1
    val = -25
    eps = 0.0005
    for _ in range(1000):
        TNR = ((preds_in >= val).sum().item() /
               preds_in.size(0))
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
        TPR_in = (preds_in >= val).sum().item()/preds_in.size(0)
        TPR_out = (preds_ood <= val).sum().item()/preds_ood.size(0)
        detection = (TPR_in+TPR_out)/2
        if detection > max_det:
            max_det = detection
            max_thres = val
        val += step
    return max_thres, max_det


def auroc_score(preds_in, preds_ood):
    y_true = np.concatenate((np.zeros(preds_ood.size(0)),
                             np.ones(preds_in.size(0))))
    preds = np.concatenate((preds_ood.numpy(), preds_in.numpy()))
    return roc_auc_score(y_true, preds)


def aurpr_score(preds_in, preds_ood):
    y_true = np.concatenate((np.zeros(preds_ood.size(0)),
                             np.ones(preds_in.size(0))))
    preds = np.concatenate((preds_ood.numpy(), preds_in.numpy()))
    precision, recall, _ = precision_recall_curve(y_true, preds)
    return auc(precision, recall)
