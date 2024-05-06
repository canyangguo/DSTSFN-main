import numpy as np


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def masked_smape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs(y_pred - y_true) / ((y_true+np.abs(y_pred))/2)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_wmape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.sum(np.abs(y_pred - y_true)*mask) / np.sum(y_true*mask)
        #mape = np.nan_to_num(mask * mape)
        return mape * 100


def masked_msis_np(labels, pres, null_val=np.nan, index=0, a=0.05, c=1.96, h=12):

    if labels.shape[-1] in [170, 883, 307, 358]:
        m = 288
    elif labels.shape[-1] in [627, 524]:
        m = 144

    # L, T, N
    mask = mask_np(labels, null_val)
    mask /= mask.mean()
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
    
    IS = np.mean(np.nan_to_num(mask * IS), axis=1) # L, N
    IS = np.mean(IS, axis=0) # N
    msis = np.mean(IS / scale)

    return msis
    
    
    