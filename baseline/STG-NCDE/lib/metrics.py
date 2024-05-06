'''
Always evaluate the model with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
'''
import numpy as np
import torch

def masked_msis(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, idx=0, a=0.05, c=1.96, h=12) -> torch.Tensor:
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
    
    if preds.shape[1] == 1:
        sigma.append(torch.std(preds[:, 0] - labels[:, 0], dim=0, keepdims=True) * 1)#((index + 1) ** 0.5))
    else:
        for i in range(preds.shape[1]):
            sigma.append(torch.std(preds[:, i] - labels[:, i], dim=0, keepdims=True) * 1)#(i + 1) ** 0.5)  # 1, N
    


    sigma = torch.stack(sigma, dim=0).permute(1,0,2)

    U = preds + c * sigma
    L = preds - c * sigma
    
    IS = (U - L) + torch.where(labels < L, 1, 0) * (a/2) * (L-labels) + torch.where(labels > U, 1, 0) * (a/2) * (labels-U) # L, T, N

    IS = torch.mean(mask * IS, dim=1) # L, N
    IS = torch.mean(IS, dim=0) # N

    msis = torch.mean(IS / scale)
    
    
    
    
    '''
    loss = torch.abs(preds-labels)
    loss = loss * mask
    
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    '''
    return msis


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))
    
  


def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        temp = torch.abs(true-pred)/(torch.abs(true)*0.5+torch.abs(pred)*0.5)
    return torch.mean(temp)

   
    
def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sum(torch.abs(true-pred)) / torch.sum(true)


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))
    
def SMAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    temp = np.divide(np.absolute(true - pred), (np.absolute(true)+np.absolute(pred))/2)
    return np.mean(temp)
    
    
def WMAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.absolute(np.divide(np.sum(np.absolute(true - pred)), np.sum(true)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation




def msis_np(labels, pres, null_val=np.nan, index=0, a=0.05, c=1.96, h=12, m=288):
    # L, T, N
    if labels.shape[-1] in [170, 883, 307, 358]:
        m = 288
    elif labels.shape[-1] in [627, 524]:
        m = 144
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


def All_Metrics(pred, true, mask1, mask2, i=0):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)

    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        smape = SMAPE_np(pred, true, mask2)
        wmape = WMAPE_np(pred, true, mask2)
        msis = msis_np(pred, true, 0)
        rrse = RRSE_np(pred, true, mask1)
        corr = 0
        #corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae  = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        mape = MAPE_torch(pred, true, mask2)
        smape = SMAPE_torch(pred, true, mask2)
        wmape = WMAPE_torch(pred, true, mask2)
        msis = masked_msis(pred, true, 0)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape, smape, wmape, msis

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr

if __name__ == '__main__':
    pred = torch.Tensor([1, 2, 3,4])
    true = torch.Tensor([2, 1, 4,5])
    print(All_Metrics(pred, true, None, None))

