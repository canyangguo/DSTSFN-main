import torch
import numpy as np





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
            sigma.append(torch.std(pres[:, i] - labels[:, i], dim=0, keepdims=True) * 1)#(i + 1) ** 0.5)  # 1, N
    


    sigma = torch.stack(sigma, dim=0).permute(1,0,2)

    U = preds + c * sigma
    L = preds - c * sigma
    
    IS = (U - L) + torch.where(labels < L, 1, 0) * (a/2) * (L-labels) + torch.where(labels > U, 1, 0) * (a/2) * (labels-U) # L, T, N

    IS = torch.mean(mask * IS, dim=1) # L, N
    IS = torch.mean(IS, dim=0) # N

    msis = torch.mean(IS / scale)