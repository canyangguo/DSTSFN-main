import time
import torch
import numpy as np
from utils.evaluation import masked_mae_np, masked_mape_np, masked_mse_np, masked_smape_np, masked_wmape_np, masked_msis_np
from utils.setting import log_string


def param_init(net, log):
    num_params = 0

    for param in net.parameters():
        log_string(log, str(param.shape), p=False)
        num_params += param.numel()
    return num_params


def training(net,
             train_loader,
             val_loader,
             test_loader,
             epochs,
             training_samples,
             val_samples,
             learning_rate,
             global_epoch,
             num_for_predict,
             num_of_vertices,
             params_filename,
             lamda,
             log,
             drop_noise=False):
    # if torch.cuda.device_count() > 1:
    # net = net.cuda()
    # net = torch.nn.DataParallel(net.cuda(), device_ids=[0,1])  # n GPUs

    criterion = torch.nn.HuberLoss(delta=1)
    mae_criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5,
                                 
                                 )

    lr_decay = False
    lr_decay_steps = [15, 40, 70, 105, 145]
    lr_decay_rate = 0.3
    if lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=lr_decay_rate)


    all_info = []
    lowest_val_loss = 1e6
    info_train = []
    info_val = []
    '''
    for ep in range(epochs):

        if ep == 150:
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=learning_rate,
                                         weight_decay=1e-5
                                         )


        t = time.time()

        mae = 0
        net.train()
        with torch.enable_grad():
            for idx, (data, label) in enumerate(train_loader):
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                pre = net(data)
                if drop_noise == True:
                    error = torch.abs(pre - label)
                    pos = torch.where(error < torch.mean(error) + lamda * torch.std(error))
                    loss_tra = criterion(pre[pos], label[pos])
                else:
                    loss_tra = criterion(pre, label)
                # print(error.shape, pos[0].shape[0], error.shape[0]*error.shape[1]*error.shape[2]-pos[0].shape[0])

                mae += mae_criterion(pre, label).item() * (pre.shape[0] / training_samples)
                optimizer.zero_grad()
                loss_tra.backward()
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=5, norm_type=2) # shenzhen overfit
                optimizer.step()
        if lr_decay:
            lr_scheduler.step()

        log_string(log, 'training: epoch: {}, mae: {:.2f}, time: {:.2f}s'.format(
            global_epoch, mae, time.time() - t))
        net.eval()
        with torch.no_grad():
            loss = 0
            for idx, (data, label) in enumerate(val_loader):
                data = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                pre = net(data)

                loss += mae_criterion(pre, label).item() * (pre.shape[0] / val_samples)

            log_string(log, 'validation: epoch: {}, loss: {:.2f}, time: {:.2f}s'.format(
                global_epoch, loss, time.time() - t))

            if loss < lowest_val_loss:
                log_string(log, 'update params...\n')
                best_epoch = global_epoch
                torch.save(net.state_dict(), params_filename)
                lowest_val_loss = loss
        info_train.append(mae)
        info_val.append(loss)
        global_epoch += 1
    '''
    net.load_state_dict(torch.load(params_filename))

    net.eval()
    with torch.no_grad():
        pres = np.zeros((0, num_for_predict, num_of_vertices))
        labels = np.zeros((0, num_for_predict, num_of_vertices))
        for idx, (data, label) in enumerate(test_loader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            pre = net(data)

            pres = np.r_[pres, pre.to('cpu').detach().numpy()]
            labels = np.r_[labels, label.to('cpu').detach().numpy()]

        tmp_info = []
        for idx in range(num_for_predict):
            y, x = labels[:, idx: idx + 1, :], pres[:, idx: idx + 1, :]
            tmp_info.append((
                masked_mae_np(y, x, 0),
                masked_mape_np(y, x, 0),
                masked_mse_np(y, x, 0) ** 0.5,
                masked_smape_np(y, x, 0),
                masked_wmape_np(y, x, 0),
                masked_msis_np(y, x, 0, idx)
            ))

        tmp_info.append((
            masked_mae_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mape_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mse_np(labels[:, : 12, :], pres[:, : 12, :], 0) ** 0.5,
            masked_smape_np(labels[:,: 12,:], pres[:,: 12,:], 0),
            masked_wmape_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_msis_np(labels[:, : 12, :], pres[:, : 12, :], 0)
        ))
        mae, mape, rmse, smape, wmape, msis = tmp_info[-1]

        log_string(log, 'mae: {:.3f}, mape: {:.3f}, rmse: {:.3f}, smape: {:.3f}, wmape: {:.3f}, msis: {:.3f}\n'.format(

            mae, mape, rmse, smape, wmape, msis))
    return tmp_info, info_train, info_val

