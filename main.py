# -*- coding:utf-8 -*-
import json
from utils.data_process import load_data, load_adj
from utils.model_fit import setup_seed, training, param_init
from utils.logs import log_string
import numpy as np
import torch
import random
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--gpu", type=str, default='0', help="gpu ID")
parser.add_argument("--dropout_rate", type=float, default=0.80, help="dropout_rate")
parser.add_argument("--missing_rate", type=float, default=0.)
parser.add_argument("--model_name", type=str, default='DPSTGCN', help="model_name")
parser.add_argument("--num_of_latents", type=int, default=32, help="num_of_latents")
parser.add_argument("--test", default=False)
parser.add_argument("--num_of_layers", type=int, default=3, help="random seed")
parser.add_argument("--S2L", type=int, default=5, help="epoch per step")
parser.add_argument("--S2D", type=int, default=80, help="switch static to dynamic")
args = parser.parse_args()

config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

model_name = args.model_name
data_name = config['data_name']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
num_of_vertices = config['num_of_vertices']
id_filename = config['id_filename']
batch_size = config['batch_size']
input_length = config['input_length']
d_model = config['d_model']
filters = config['filters']
num_of_times = config['num_of_times']
num_of_days = config['num_of_days']
use_mask = config['use_mask']
temporal_emb = config['temporal_emb']
spatial_emb = config['spatial_emb']
output_length = config['output_length']
num_of_features = config['num_of_features']
num_of_outputs = config['num_of_outputs']
receptive_length = config['receptive_length']
num_of_layers = args.num_of_layers
dropout_rate = args.dropout_rate
num_of_latents = args.num_of_latents
epochs = config['epochs']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
missing_rate = args.missing_rate
epoch_per_step = args.S2L
static_to_dynamic = args.S2D

config_name = '_' + model_name + ', layers_' + str(args.num_of_layers) \
              + ', hidden_dim_' + str(num_of_latents) + ', seed_' + str(args.seed)

log = open(config['log_path'] + config_name + '.txt', 'w')
param_file = config['params_filename'] + '_' + config_name


from models.DSTSFN import make_model



log_string(log, 'let us begin! traning ' + model_name + ' ○（*￣︶￣*）○\n')
log_string(log, 'param file: {}'.format(param_file))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
setup_seed(args.seed)
log_string(log, 'gpu ID: {}, random seed: {}\n'.format(args.gpu, args.seed))

log_string(log, str(json.dumps(config, sort_keys=True, indent=4)), pnt=False)

log_string(log, '*****data loading*****\n')
# adj_st, mask_init_value_st = load_adj(adj_dtw_filename, graph_signal_matrix_filename, adj_filename, num_of_vertices, id_filename, model_name, log)
train_loader, val_loader, test_loader, training_samples, val_samples, test_sample = load_data(graph_signal_matrix_filename, num_of_times, num_of_days, batch_size, args.test, log, missing_rate, data_name)

log_string(log, '*****make model*****\n')
net = make_model(input_length, num_of_vertices, d_model, filters, use_mask,
            temporal_emb, spatial_emb, output_length, num_of_features, num_of_outputs, receptive_length, dropout_rate,
            num_of_latents, num_of_layers, num_of_times, num_of_days).cuda()
num_params = param_init(net, log)
log_string(log, "num of parameters: {}".format(num_params))

log_string(log, '*****start training model*****')
all_info, train_loss, val_loss, train_time = training(net, num_of_layers, train_loader, val_loader, test_loader, epochs,
                                          training_samples, val_samples, learning_rate, weight_decay, output_length,
                                          num_of_vertices, param_file, log, static_to_dynamic, epoch_per_step)

log_string(log, '*****loss curve*****')
log_string(log, 'train_loss:\n' + str(train_loss))
log_string(log, 'val_loss:\n' + str(val_loss))
log_string(log, 'train_time:\n' + str(train_time))
log_string(log, '\n')
log_string(log, 'total_train_time:\n' + str(sum(train_time)))

log_string(log, '*****multi step prediction*****')

k = 1
for i in all_info:
    if k <= 12:
        log_string(log, '@t' + str(k) + ' {:.3f} {:.3f} {:.3f}'.format(*i))
    if k ==13:
        log_string(log, 'avg {:.3f} {:.3f} {:.3f}'.format(*i))
    k = k+1

log_string(log, 'end!!! ' + model_name
           + ', layers_' + str(args.num_of_layers)
           + ', hidden_dim_' + str(num_of_latents)
           + ', seed_' + str(args.seed))
log.close()



