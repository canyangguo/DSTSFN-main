import os
import argparse
import random
from libcity.pipeline import run_model
from libcity.utils import general_arguments, str2bool, str2float
import numpy as np
import torch
from libcity.config import ConfigParser
def add_other_args(parser):
    for arg in general_arguments:
        if general_arguments[arg] == 'int':
            parser.add_argument('--{}'.format(arg), type=int, default=None)
        elif general_arguments[arg] == 'bool':
            parser.add_argument('--{}'.format(arg),
                                type=str2bool, default=None)
        elif general_arguments[arg] == 'str':
            parser.add_argument('--{}'.format(arg),
                                type=str, default=None)
        elif general_arguments[arg] == 'float':
            parser.add_argument('--{}'.format(arg),
                                type=str2float, default=None)
        elif general_arguments[arg] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+',
                                type=int, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str)
    
    parser.add_argument('--task', type=str,
                        default='traffic_state_pred', help='the name of task')
    parser.add_argument('--model', type=str,
                        default='GRU', help='the name of model')
                        
    parser.add_argument('--source', type=str,
                        help='the name of dataset')
    parser.add_argument('--target', type=str,
                        help='the name of dataset')
    parser.add_argument('--cuda', type=str, default='1')                    

                        
    parser.add_argument('--saved_model', type=str2bool,
                        default=True, help='whether save the trained model')
    parser.add_argument('--train', type=str2bool, default=True,
                        help='whether re-train model if the model is \
                             trained before')
    parser.add_argument("--local_rank", default=1, type=int)
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
                        
    
                        
                        
    add_other_args(parser)
    args = parser.parse_args()
    
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    
    
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda))

    config = ConfigParser(args.task, args.model, args.source,args.source, args.saved_model, args.train, other_args)
    config['stage'] = 's'
    
    run_model(config, stage='s', task=args.task, model_name=args.model, dataset_name=args.source, source=args.source,
              config_file=args.source, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
       
    '''   
    config = ConfigParser(args.task, args.model, args.target,
                          args.target, args.saved_model, args.train, other_args)  
    config['stage'] = 't'
    config['max_epoch'] = 230
    config['epoch'] = 200
    config['learning_rate'] = 2e-4
    config['task_level'] = 12
    config['use_curriculum_learning'] = 'false'
    config['cand_key_days'] = 3
    run_model(config, stage='t', task=args.task, model_name=args.model, dataset_name=args.target, source=args.source,
              config_file=args.target, saved_model=args.saved_model,
              train=args.train, other_args=other_args)
    '''
