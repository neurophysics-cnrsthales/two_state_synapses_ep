import datetime
import os

import pandas as pd
import torch

COLUMNS = ['Train_Error', 'Train_Loss', 'Weight_Norm', 'Test_Error', 'Min_Test_Error']


def def_rho(activ_type):
    """Activation function rho and its derivative rhop (smooth hard-sigmoid, the only one used)."""
    if activ_type != 'smooth_hardsigm':
        raise ValueError(f"{activ_type} activation function is not defined!")

    def rho(x):
        return torch.clamp(0.25 * x, min=0, max=1)

    def rhop(x):
        return ((x >= 0) & (x <= 4)) * 0.25

    return rho, rhop


def define_optimizer(net, lr, optimizer_type, momentum=0.0, nesterov=False, weight_decay=0.0):
    """One parameter group per weight layer, each with its own learning rate (no bias).
    The defaults give plain SGD / default Adam."""
    net_params = [{'params': [net.W[i].weight], 'lr': lr[i]} for i in range(len(net.W))]

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net_params, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net_params, weight_decay=weight_decay)
    else:
        raise ValueError("{} Optimizer is not defined ".format(optimizer_type))

    return net_params, optimizer


def new_run_dir(prefix='ep-'):
    """DATA-0/<date>/<prefix><n> with the next free n (used when --out_dir is not given)."""
    base_path = os.path.join(os.getcwd(), 'DATA-0', datetime.datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(base_path, exist_ok=True)
    indices = [int(f.split('-')[1]) for f in os.listdir(base_path) if f.startswith(prefix) and f.split('-')[1].isdigit()]
    new_path = os.path.join(base_path, f'{prefix}{max(indices, default=0) + 1}')
    os.mkdir(new_path)
    return new_path


def init_dataframe(path, dataframe_to_init='results.csv'):
    csv_path = os.path.join(path, dataframe_to_init)
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path, sep=',', index_col=0)
    dataframe = pd.DataFrame({}, columns=COLUMNS)
    dataframe.to_csv(csv_path)
    return dataframe


def update_dataframe(path, dataframe, train_error, weight_norm, test_error, loss, filename='results.csv'):
    """Append the last epoch to results.csv. (The first column is a row index that is 0 in every row; the row
    order is the epoch order.)"""
    row = pd.DataFrame([[train_error[-1], loss[-1], weight_norm[-1], test_error[-1], min(test_error)]], columns=COLUMNS)
    dataframe = row if dataframe.empty else pd.concat([dataframe, row])
    dataframe.to_csv(os.path.join(path, filename))
    return dataframe
