import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'chengdu':
        data_path = os.path.join('../data/chengdu/dataset.npy')
        data = np.load(data_path)[:, :, 0:1]
    elif dataset == 'shenzhen':
        data_path = os.path.join('../data/shenzhen/dataset.npy')
        data = np.load(data_path)[:, :, 0:1]
    elif dataset == 'PEMSD4':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        # data_path = os.path.join('../data/PeMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD7L':
        data_path = os.path.join('../data/PEMS07L/PEMS07L.npz')
        data = np.load(data_path)['data'][:, :, 0:1]  #onley the first dimension, traffic flow data
    elif dataset == 'Decentraland':
        data_path = os.path.join('../token_data/Decentraland_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0:1]  #1 dimension, degree
    elif dataset == 'Bytom':
        data_path = os.path.join('../token_data/Bytom_node_features.npz')
        data = np.load(data_path)['arr_0'][:, :, 0:1]  #1 dimension, degree
    else:
        raise ValueError
        
    print(data.shape)
    data = data[:, :, 0: 1]
    index = np.arange(data.shape[0]).reshape(data.shape[0], 1, 1).repeat(data.shape[1], axis=1)
    
    if len(data) % 288 == 0:
        print(len(data) / 288)
        time_index = (index % 288) / 288
        day_index = (index // 288) % 7
    else:
        print(len(data) / 144)
        time_index = (index % 144) / 144
        day_index = (index // 144) % 7


    data = np.concatenate([data, time_index, day_index], axis=-1)
        
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data[:, :, 0: 1].max(), data[:, :, 0: 1].min(), data[:, :, 0: 1].mean(), np.median(data[:, :, 0: 1]))
    return data
