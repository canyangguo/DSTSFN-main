# -*- coding:utf-8 -*-
import json
from utils.setting import setup_seed, log_string, parser_set, select_GPU
# import hiddenlayer as h
# from torchviz import make_dot
from utils.data_process import load_data, load_adj
from utils.model_fit import training, param_init


model_name = 'STSGCN'


if model_name == 'STSGCN':
    from models.STSGCN import _main



args = parser_set()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())
log = open(config['log_path'] + '_' + args.model_name + '_' + str(args.st_dropout_noise_rate) + '_' + str(args.st_dropout_rate) + '_' + str(args.num_of_latents) + '.txt', 'w')

param_name = config['params_filename'] + '_' + args.model_name + '_' + str(args.st_dropout_noise_rate) + '_' + str(args.st_dropout_rate) + '_' + str(args.num_of_latents)
print(param_name)
log_string(log, 'let us begin! traning ' + model_name + ' ○（*￣︶￣*）○\n')
log_string(log, '----------parameter setting----------\n')
setup_seed(args.seed)
select_GPU(config['ctx'])
log_string(log, str(json.dumps(config, sort_keys=True, indent=4)), p=False)

log_string(log, '----------data loading----------')
adj_st, mask_init_value_st = load_adj(config['adj_dtw_filename'],
                                      config['graph_signal_matrix_filename'],
                                      config['adj_filename'],
                                      config['num_of_vertices'],
                                      config['id_filename'],
                                      model_name,
                                      data_name=config['data_name'])

adj_st = adj_st.cuda()
train_loader, val_loader, test_loader, training_samples, val_samples, test_sample = load_data(config['graph_signal_matrix_filename'],
                                                                                              config['batch_size'],
                                                                                              args.test,
                                                                                              log, data_name=config['data_name'])

log_string(log, 'adj shape:{}'.format(adj_st.shape))


log_string(log, '----------model setting----------\n')
net = _main(adj_st, config['points_per_hour'], config['num_of_vertices'], config['first_layer_embedding_size'], config['filters'],
            config['use_mask'], mask_init_value_st, config['temporal_emb'], config['spatial_emb'], config['num_for_predict'],
            config['num_of_features'], config['receptive_length'], args.st_dropout_rate, args.num_of_latents,
            config['num_of_gcn_filters']).cuda()

total = sum([param.nelement() for param in net.parameters()])
log_string(log, 'the number of parameters:{}'.format(total))

num_params = param_init(net, log)
log_string(log, "num of parameters: {}".format(num_params))

log_string(log, '----------model training----------')
global_train_steps = training_samples // config['batch_size'] + 1
if args.test:
    config['epochs'] = 5
all_info, train_loss, val_loss = training(net, train_loader, val_loader, test_loader,
                                          config['epochs'], training_samples, val_samples,
                                          config['learning_rate'], config['global_epoch'], config['num_for_predict'],
                                          config['num_of_vertices'], param_name, args.st_dropout_noise_rate, log, model_name)

log_string(log, '----------multi step prediction----------')
k = 1
num_for_predict=12
for i in all_info:
    if k <= num_for_predict:
        log_string(log, '@t' + str(k) + ' {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(*i))
    if k ==num_for_predict+1:
        log_string(log, 'avg {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(*i))
    k = k+1

log_string(log, 'end!!! ' + model_name + ', seed_' + str(args.seed))

log.close()
print(param_name)


