import math
import functools
from typing import Tuple, Union, Optional
import copy
import torch
import numpy as np
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl
from ..metrics import masked_mae, masked_mape, masked_rmse, masked_smape, masked_wmape, masked_msis
import numpy as np
import torch
import random

class BaseTimeSeriesForecastingRunner(BaseRunner):
    """
    Runner for short term multivariate time series forecasting datasets.
    Typically, models predict the future 12 time steps based on historical time series.
    Features:
        - Evaluate at horizon 3, 6, 12, and overall.
        - Metrics: MAE, RMSE, MAPE. The best model is the one with the smallest mae at validation.
        - Loss: MAE (masked_mae). Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg["TRAIN"].get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg["DATASET_TYPE"]

        # read scaler for re-normalization
        self.scaler = load_pkl("datasets/" + self.dataset_name + "/scaler_in{0}_out{1}.pkl".format(
                                                cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"]))
        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        # define metric
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "SMAPE": masked_smape, "WMAPE": masked_wmape, "MSIS": masked_msis}
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg.TRAIN.get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg.TRAIN.CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg.TRAIN.CL.get("CL_EPOCHS")
            self.prediction_length = cfg.TRAIN.CL.get("PREDICTION_LENGTH")
            self.cl_step_size = cfg.TRAIN.CL.get("STEP_SIZE", 1)
        # evaluation horizon
        self.evaluation_horizons = [_ - 1 for _ in cfg["TEST"].get("EVALUATION_HORIZONS", range(1, 13))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 0."

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.4f}")
        self.test_process(train_epoch=200)
    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
        index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item



    


    def svt(self, mat, tau):
        u, s, v = torch.linalg.svd(mat, full_matrices = False)
        idx = torch.sum(s > tau)
        return u[:, : idx] @ torch.diag(s[: idx] - tau) @ v[: idx, :]

    def lrmc_imputer(self, mat, rho0, epsilon, maxiter):
        dim1, dim2 = mat.shape
        pos_missing = torch.where(mat == 0)
        last_mat = copy.deepcopy(mat)
        snorm = torch.linalg.norm(mat, 'fro')
        T = torch.zeros(dim1, dim2)
        Z = copy.deepcopy(mat)

        Z[pos_missing] = torch.mean(Z[Z != 0])
        it = 0
        rho = rho0
        while True:
            rho = min(rho * 1.05, 1e5)
            X = self.svt(Z - T / rho, 1 / rho)
            Z[pos_missing] = (X + T / rho)[pos_missing]
            T = T + rho * (X - Z)
            tol = torch.linalg.norm((X - last_mat), 'fro') / snorm
            last_mat = copy.deepcopy(X)
            it += 1
            # if it % 1 == 0:
            #     print('Iter: {}'.format(it))
            #     print('Tolerance: {:.6}'.format(tol))
            #     print()
            if (tol < epsilon) or (it >= maxiter):
                break
        return X

    def start_imputer(self, x, mask, mean, std):
        
        #x = (x * std + mean) * mask
        '''
        #x = (x * mean_std[1] + mean_std[0]) * mask

        rho = 1e-2
        epsilon = 1e-4
        maxiter = 10
        mat_hat = self.lrmc_imputer(x, rho, epsilon, maxiter)
        x = x * mask + (1-mask) * mat_hat
        '''
        #x = x * mask + (1-mask) * mean#torch.where(mask==1, mean, x)
        #x = (x - mean) / std
        
        return x * mask



    def get_sparse_data(self, data, mean, std, rate):

        y, x = data  # 奇葩，future在前，his在后     
        missing_rate = torch.rand(x.shape[0]) * rate  #  缺失范围 0 - rate
        #missing_rate = torch.ones(x.shape[0]) * rate #固定缺失率 rate%

        dim1, dim2, dim3, dim4 = x[:, :, :, :1].shape
        mask = [torch.round(torch.rand(dim2, dim3, dim4) + 0.5 - missing_rate[i]) for i in range(dim1)]  # 0-1随机数
        mask = torch.stack(mask, dim=0)
        
        #impute = True
        #if impute:
            #x[..., :1] = torch.stack([self.start_imputer(x[i, :, :, 0], mask[i, :, :, 0].squeeze(-1), mean, std).unsqueeze(-1)
                        #for i in range(x.shape[0])], dim=0)
        #x[..., :1] = x[..., :1] * std + mean

        delta = torch.randn(x[..., :1].shape) * 1
        x[..., :1] = x[..., :1] + (mask * delta)

        
        #x[..., :1] = (x[..., :1] - mean) / std

        data = tuple([y, x])
        return data

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """
        
        
        #data = self.get_sparse_data(data)  # new add

        iter_num = (epoch-1) * self.iter_per_epoch + iter_index
        forward_return = list(self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True))
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return[0] = prediction_rescaled[:, :cl_length, :, :]
            forward_return[1] = real_value_rescaled[:, :cl_length, :, :]
        else:
            forward_return[0] = prediction_rescaled
            forward_return[1] = real_value_rescaled
        loss = self.metric_forward(self.loss, forward_return)
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return[:2])
            self.update_epoch_meter("train_"+metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """
        #data = self.get_sparse_data(data)  # new add
        forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            self.update_epoch_meter("val_"+metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    
    
    def setup_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # for CPU
        torch.cuda.manual_seed(seed)  # for current GPU
        torch.cuda.manual_seed_all(seed)  # for all GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
    
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop
        for rate in range(10, 11):
            rate = rate / 10

            prediction = []
            real_value = []
            #self.setup_seed(0)
            for _, data in enumerate(self.test_data_loader):
                #data = self.get_sparse_data(data, self.scaler["args"]['mean'], self.scaler["args"]['std'], rate)  # new add
                forward_return = self.forward(data, epoch=None, iter_num=None, train=False)
                prediction.append(forward_return[0])  # preds = forward_return[0]
                real_value.append(forward_return[1])  # testy = forward_return[1]
            prediction = torch.cat(prediction, dim=0)
            real_value = torch.cat(real_value, dim=0)
            # re-scale data
            prediction = SCALER_REGISTRY.get(self.scaler["func"])(prediction, **self.scaler["args"])
            real_value = SCALER_REGISTRY.get(self.scaler["func"])(
                real_value, **self.scaler["args"])
            # summarize the results.
            # test performance of different horizon
            for i in self.evaluation_horizons:
                # For horizon i, only calculate the metrics **at that time** slice here.
                pred = prediction[:, i, :, :]
                real = real_value[:, i, :, :]
                # metrics
                metric_results = {}
                for metric_name, metric_func in self.metrics.items():
                    metric_item = self.metric_forward(metric_func, [pred, real])
                    metric_results[metric_name] = metric_item.item()
                log = "Evaluate best model on test data for horizon " + \
                      "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SMAPE: {:.4f}, Test WMAPE: {:.4f}, Test MSIS: {:.4f}"
                log = log.format(
                    i + 1, metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"],
                    metric_results["SMAPE"], metric_results["WMAPE"], metric_results["MSIS"])
                self.logger.info(log)

            # test performance overall
            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, [prediction, real_value])
                self.update_epoch_meter("test_" + metric_name, metric_item.item())
                metric_results[metric_name] = metric_item.item()
                    
            
            
        

    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """
        
        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)