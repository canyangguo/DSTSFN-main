import torch
import numpy as np
from sklearn.metrics import r2_score, explained_variance_score


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)




def masked_msis_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, a=0.05, c=1.96, h=12) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.

    preds = preds.squeeze(-1)
    labels = labels.squeeze(-1)
    
    if preds.shape[-1] in [170, 883, 307, 358]:
        m = 288
    elif preds.shape[-1] in [627, 524]:
        m = 144
    
    if len(preds.shape) == 2:
        preds = preds.unsqueeze(1)
        labels = labels.unsqueeze(1)
        
    
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    
    
    
    
    # TODO fix very large values
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    sigma = []
    scale = (1/(labels.shape[0]-m)) * torch.sum(torch.abs(labels[m:] - labels[:-m]), dim=0) # T, N
    scale = torch.mean(scale, dim=0)  # N
    for i in range(preds.shape[1]):
        sigma.append(torch.std(preds[:, i]-labels[:, i], dim=0, keepdims=True)) # 1, N

    sigma = torch.stack(sigma, dim=0).permute(1,0,2)

    U = preds + c * sigma
    L = preds - c * sigma
    
    IS = (U - L) + torch.where(labels < L, 1, 0) * (a/2) * (L-labels) + torch.where(labels > U, 1, 0) * (a/2) * (labels-U) # L, T, N

    IS = torch.mean(mask * IS, dim=1) # L, N
    IS = torch.mean(IS, dim=0) # N

    msis = torch.mean(IS / scale)
    
    return msis


def msis_torch(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, a=0.05, c=1.96, h=12) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.

    preds = preds.squeeze(-1)
    labels = labels.squeeze(-1)
    
    if preds.shape[-1] in [170, 883, 307, 358]:
        m = 288
    elif preds.shape[-1] in [627, 524]:
        m = 144
    
    if len(preds.shape) == 2:
        preds = preds.unsqueeze(1)
        labels = labels.unsqueeze(1)
        
    
    labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    
    
    
    
    # TODO fix very large values
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    sigma = []
    scale = (1/(labels.shape[0]-m)) * torch.sum(torch.abs(labels[m:] - labels[:-m]), dim=0) # T, N
    scale = torch.mean(scale, dim=0)  # N
    for i in range(preds.shape[1]):
        sigma.append(torch.std(preds[:, i]-labels[:, i], dim=0, keepdims=True)) # 1, N

    sigma = torch.stack(sigma, dim=0).permute(1,0,2)

    U = preds + c * sigma
    L = preds - c * sigma
    
    IS = (U - L) + torch.where(labels < L, 1, 0) * (a/2) * (L-labels) + torch.where(labels > U, 1, 0) * (a/2) * (labels-U) # L, T, N

    IS = torch.mean(IS, dim=1) # L, N
    IS = torch.mean(IS, dim=0) # N

    msis = torch.mean(IS / scale)
    
    return msis





def log_cosh_loss(preds, labels):
    loss = torch.log(torch.cosh(preds - labels))
    return torch.mean(loss)


def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def quantile_loss(preds, labels, delta=0.25):
    condition = torch.ge(labels, preds)
    large_res = delta * (labels - preds)
    small_res = (1 - delta) * (preds - labels)
    return torch.mean(torch.where(condition, large_res, small_res))


def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    #labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    #labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    #labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))

def masked_smape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    
    
    #labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    
    
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/((torch.abs(preds)+labels)/2)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_wmape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """

    # fix very small values of labels, which should be 0. Otherwise, nan detector will fail.
    
    
    #labels = torch.where(labels < 1e-4, torch.zeros_like(labels), labels)
    
    
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.sum(torch.abs(preds-labels)* mask)/torch.sum(labels * mask)
    return loss

def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)


def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels,
                   null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_smape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= mask.mean()
        mape = np.abs(y_pred - y_true) / ((y_true+np.abs(y_pred))/2)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_wmape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= mask.mean()
        mape = np.sum(np.abs(y_pred - y_true)*mask) / np.sum(y_true*mask)
        #mape = np.nan_to_num(mask * mape)
        return mape * 100


def r2_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return r2_score(labels, preds)


def explained_variance_score_np(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    return explained_variance_score(labels, preds)
