# -*- coding:utf-8 -*-
import json
from utils.data_process import load_data
import argparse
from utils.evaluation import masked_mae_np, masked_mape_np, masked_mse_np, masked_smape_np, masked_wmape_np, masked_msis_np
import numpy as np
import pandas as pd
from sklearn.svm import SVR

def log_string(log, string, p=True):  # p decide print
    log.write(string + '\n')
    log.flush()
    if p:
        print(string)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--model_name", type=str, default='new', help="model_name")
args = parser.parse_args()

config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

num_for_predict = config['num_for_predict']    
num_of_vertices = config['num_of_vertices']

log = open(config['log_path'] + '_' + args.model_name + '.txt', 'w')
log_string(log, 'let us begin! traning ' + args.model_name + ' ○（*￣︶￣*）○\n')
#setup_seed(args.seed)






if args.model_name == 'SVR':
    loaders = load_data(config['graph_signal_matrix_filename'], config['batch_size'], log, data_name=config['data_name'])
    training_data, validation_data, testing_data = loaders
    # training data: L, T, N
    pres = []
    tmp_info = []
    L, T, N = training_data[0].shape
    for i in range(N):
        train_x, train_y = training_data
        test_x, labels = testing_data
        train_x = train_x[:, :, i]
        train_y = train_y[:, :, i]
        test_x = test_x[:, :, i]
        train_y = np.mean(train_y, axis=1)
        svr_model = SVR(kernel='linear')
        svr_model.fit(train_x, train_y)
        pre = svr_model.predict(test_x)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(num_for_predict, axis=1)
        pres.append(pre)
        print(str(i) + "/" + "N")
    pres = np.array(pres).transpose(1, 2, 0) # N



if 'PEMS' in config['graph_signal_matrix_filename']:
    data = np.load(config['graph_signal_matrix_filename'])['data'][...,  0]
else:
    data = np.load(config['graph_signal_matrix_filename'])[...,  0]
test = data[int(0.8*len(data)):]
length = len(test)

if args.model_name == 'HA':
    pres = []
    labels = []
    for i in range(length-23):
        pres.append(test[i: i+12])
        labels.append(test[i+12: i+12+12])
    pres = np.array(pres)
    labels = np.array(labels)
    pres = np.mean(pres, axis=1, keepdims=True).repeat(num_for_predict, axis=1)

if args.model_name == 'ARIMA':
    test_ = test
    test = pd.DataFrame(data[int(0.8*len(data)):])
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    rng = pd.date_range('1/3/2012', periods=len(test), freq='5min')
    a1 = pd.DatetimeIndex(rng)
    test.index = a1
    num = test.shape[1]
    pres = []
    labels = []

    for i in range(num):
        print('{}/{}'.format(i+1, num))
        ts = test.iloc[:, i]
        #ts_log = ts#np.log(ts)
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log, order=[3, 0, 2])
        Model = model.fit()
        pre = []
        label = []
        for j in range(length-23):
            predict = Model.predict(j+12, j+12+11, dynamic=True)#当设置为True时，预测会使用模型之前步骤的预测值作为输入，而不是原始数据。这通常用于滚动预测或样本外预测。   
            predict = np.exp(predict)
            #ts = ts[predict.index]
            pre.append(predict)
            label.append(test_[j+12: j+12+12, i])
        labels.append(label)
        pres.append(pre)
    
    pres = np.array(pres)
    labels = np.array(labels).transpose(1,2,0)
    pres = np.array(pres).transpose(1,2,0)

    print('pres', pres.shape)
    print('labels', labels.shape)
    



# =====result=====
all_info = []
for idx in range(num_for_predict):
    y, x = labels[:, idx: idx + 1, :], pres[:, idx: idx + 1, :]
    all_info.append((
                masked_mae_np(y, x, 0),
                masked_mape_np(y, x, 0),
                masked_mse_np(y, x, 0) ** 0.5,
                masked_smape_np(y, x, 0),
                masked_wmape_np(y, x, 0),
                masked_msis_np(y, x, 0, idx)
            ))
all_info.append((
            masked_mae_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mape_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_mse_np(labels[:, : 12, :], pres[:, : 12, :], 0) ** 0.5,
            masked_smape_np(labels[:,: 12,:], pres[:,: 12,:], 0),
            masked_wmape_np(labels[:, : 12, :], pres[:, : 12, :], 0),
            masked_msis_np(labels[:, : 12, :], pres[:, : 12, :], 0)
            ))

mae, mape, rmse, smape, wmape, msis = all_info[-1]
log_string(log, 'mae: {:.3f}, rmse: {:.3f}, mape: {:.3f}, smape: {:.3f}, wmape: {:.3f}, msis: {:.3f}\n'.format(
            mae, rmse, mape, smape, wmape, msis))
k = 1
for i in all_info:
    if k <= num_for_predict:
        log_string(log, '@t' + str(k) + ' {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(*i))
    if k ==num_for_predict+1:
        log_string(log, 'avg {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(*i))
    k = k+1
    

log_string(log, 'end!!! ' + args.model_name)
log.close()



