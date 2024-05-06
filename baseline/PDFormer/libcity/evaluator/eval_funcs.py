import numpy as np
import torch


def mse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MSE'
    return np.mean(sum(pow(loc_pred - loc_true, 2)))


def mae(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAE'
    return np.mean(sum(loc_pred - loc_true))


def rmse(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'RMSE'
    return np.sqrt(np.mean(sum(pow(loc_pred - loc_true, 2))))


def mape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'MAPE'
    assert 0 not in loc_true, "MAPE:"
    return np.mean(abs(loc_pred - loc_true) / loc_true)


def mare(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "MARE"
    assert np.sum(loc_true) != 0, "MARE"
    return np.sum(np.abs(loc_pred - loc_true)) / np.sum(loc_true)


def smape(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), 'SMAPE'
    assert 0 in (loc_pred + loc_true), "SMAPE"
    return 2.0 * np.mean(np.abs(loc_pred - loc_true) / (np.abs(loc_pred) +
                                                        np.abs(loc_true)))

def msis(preds, labels, a=0.05, c=1.96, h=12):
    assert len(preds) == len(labels), 'msis'
    # L, T, N
    if labels.shape[-1] in [170, 883, 307, 358]:
        m = 288
    elif labels.shape[-1] in [627, 524]:
        m = 144

    sigma = []
    scale = np.mean(np.abs(labels[m:] - labels[:-m]), axis=0) # T, N
    scale = np.mean(scale, axis=0)  # N
    
    if pres.shape[1] == 1:
        sigma.append(np.std(pres[:, 0] - labels[:, 0], axis=0, keepdims=True) * 1)#((index + 1) ** 0.5))
    else:
        for i in range(pres.shape[1]):
            sigma.append(np.std(pres[:, i] - labels[:, i], axis=0, keepdims=True) * 1)#(i + 1) ** 0.5)  # 1, N
    
    sigma = np.array(sigma).transpose(1,0,2)

    U = pres + c * sigma
    L = pres - c * sigma
    
    IS = (U - L) + np.where(labels < L, 1, 0) * (a/2) * (L-labels) + np.where(labels > U, 1, 0) * (a/2) * (labels-U) # L, T, N
    
    IS = np.mean(IS, axis=1) # L, N
    IS = np.mean(IS, axis=0) # N
    msis = np.mean(IS / scale)

    return msis


def acc(loc_pred, loc_true):
    assert len(loc_pred) == len(loc_true), "accuracy"
    loc_diff = loc_pred - loc_true
    loc_diff[loc_diff != 0] = 1
    return loc_diff, np.mean(loc_diff == 0)


def top_k(loc_pred, loc_true, topk):
    assert topk > 0, "top-k ACC"
    loc_pred = torch.FloatTensor(loc_pred)
    val, index = torch.topk(loc_pred, topk, 1)
    index = index.numpy()
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):
        target = loc_true[i]
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg
