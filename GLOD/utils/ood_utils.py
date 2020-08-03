import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def search_thers(preds_in, level, start_val=-25):
    '''search_thresh(preds_in, level, start_val)
    search the threshold for a spesific tnr level
    Parameters
    ----------------
    'preds_in': Tensor
        thensor containing the predictions for in distribution datasets

    'level': float
        tnr level e.g. 0.95, 0.99

    Return
    -----------------
    threshold value, such that (preds_in <= val).sum() <=level
    '''
    step = 1
    val = start_val
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
    '''detection_accuracy(start, end, preds_in, preds_ood)
    search for the threshold which will yeilds in the maximum detection
    accuracy

    Parameters
    ----------------
    'start, end': floats
        floats values that represent the start, end of the predictions range
    'preds_in, preds_ood': Tensor
        Tensors representing the in distribution and the out distribution predictions

    Return
    -----------------
    threshold value that maxmize the detection accuracy

    the detection accuracy
    '''
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
    '''auroc_score(preds_in, preds_ood)
    calculate the aucroc score of the detector

    Parameters
    ----------------
    'preds_in, preds_ood': Tensor
        Tensors representing the in distribution and the out distribution
        predictions

    Return
    -----------------
    the aucroc score
    '''
    y_true = np.concatenate((np.zeros(preds_ood.size(0)),
                             np.ones(preds_in.size(0))))
    preds = np.concatenate((preds_ood.numpy(), preds_in.numpy()))
    return roc_auc_score(y_true, preds)


def aupr_score(preds_in, preds_ood):
    '''aupr_score(preds_in, preds_ood)
    calculate the area under precision recall curve of the detector predictions

    Parameters
    ----------------
    'preds_in, preds_ood': Tensor
        Tensors representing the in distribution and the out distribution
        predictions

    Return
    -----------------
    the aucroc score
    '''
    y_true = np.concatenate((np.zeros(preds_ood.size(0)),
                             np.ones(preds_in.size(0))))
    preds = np.concatenate((preds_ood.numpy(), preds_in.numpy()))
    precision, recall, _ = precision_recall_curve(y_true, preds)
    return auc(precision, recall)
