import os
import sys
from argparse import ArgumentParser
import torch
import numpy as np
import random
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
from basicts import launch_training

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # for CPU
    torch.cuda.manual_seed(seed)  # for current GPU
    torch.cuda.manual_seed_all(seed)  # for all GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")
    # parser.add_argument("-c", "--cfg", default="examples/DGCRN/DGCRN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_METR-LA.py", help="training config")
    parser.add_argument("-c", "--cfg", default="examples/DCRNN/DCRNN_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/GTS/GTS_PEMS03.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/STID/STID_PEMS-BAY.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/HI/HI_METR-LA.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Autoformer/Autoformer_PEMS04_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/FEDformer/FEDformer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Informer/Informer_METR-LA_in96_out96.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="examples/Pyraformer/Pyraformer_METR-LA_in96_out96.py", help="training config")
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    setup_seed(10)
    args = parse_args()

    launch_training(args.cfg, args.gpus)
