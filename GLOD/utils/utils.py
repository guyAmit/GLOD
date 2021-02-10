import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def get_test_valid_loader(batch_size,
                          test_dataset,
                          random_seed,
                          valid_size=0.1,
                          num_workers=2,
                          pin_memory=False):
        '''validation_split(batch_size, test_transform, random_seed,
                     test_dataset, valid_size=0.1, num_workers=2,
                     pin_memory=False)
    creates a validation_split from the test data of size 0.1*len(test_set)

    Parameters
    ----------------
    'batch_size': int
        batch size of data loaders
    'test_dataset': torch dataset
        the test set that will be splited into validation and test sets
    'random_seed': int
        seed for sampeling
    'valid_size': float
        fraction of data that should be used for validation
    'num_workers': int
        num of processes that should be used in each data loader
    'pin_memory': boolean
        pin_memory parameter of data loader

    Return
    -----------------
    (test_loader, valid_loader) - Two data loaders one for validation and
        one for test
    '''

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
    '''search_thresh(preds_in, level, start_val)
    search the threshold for a spesific tnr level
    Parameters
    ----------------
    'preds_in': numpy array
        array containing the predictions for in distribution datasets

    'level': float
        tnr level e.g. 0.95, 0.99

    Return
    -----------------
    threshold value, such that (preds_in <= val).sum() <=level
    '''
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
        TPR_in = (preds_in >= val).sum()/preds_in.shape[0]
        TPR_out = (preds_ood <= val).sum()/preds_ood.shape[0]
        detection = (TPR_in+TPR_out)/2
        if detection > max_det:
            max_det = detection
            max_thres = val
        val += step
    return max_thres, max_det


def search_thers_torch(preds_in, level, eps=0.0001):
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
    step = 10
    val = -10
    while True:
        TNR = (preds_in >= val).sum().item() / preds_in.shape[0]
        if TNR < level + eps and TNR > level - eps:
            return val
        elif TNR > level:
            val += step
        else:
            val = val - step
            step = step * 0.1


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
