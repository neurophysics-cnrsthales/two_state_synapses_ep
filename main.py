import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from data import mnist_gpu_loaders
from network import MacDynamic
from tool import def_rho, define_optimizer, new_run_dir, init_dataframe, update_dataframe
from train_test import train_dyn_ep, train_dyn_sequential_ep, test_dyn_ep

parser = argparse.ArgumentParser(description='Train the dynamic-synapse network from a json configuration')
parser.add_argument('--json_path', type=str, required=True, help='folder that contains config.json')
parser.add_argument('--device', type=int, default=None, help='override the "device" (GPU index, -1 = CPU) of the configuration')
parser.add_argument('--seed', type=int, default=None, help='override the "seed" of the configuration')
parser.add_argument('--out_dir', type=str, default=None,
                    help='write the results to this folder (must be empty or new); default: DATA-0/<date>/ep-<n>')
args = parser.parse_args()

with open(args.json_path + '/config.json') as f:
    jparams = json.load(f)
if args.device is not None:
    jparams['device'] = args.device
if args.seed is not None:
    jparams['seed'] = args.seed
if jparams['dataset'] != 'mnist':
    raise ValueError(f"{jparams['dataset']} as dataset is not defined!")

# val_split > 0: train on 60000 - val_split images and evaluate on the held-out validation images
# (reported in the Test_Error column); the real test set is then never used.
data_device = torch.device(f"cuda:{jparams['device']}" if jparams['device'] >= 0 and torch.cuda.is_available() else "cpu")
train_loader, test_loader = mnist_gpu_loaders('./data', jparams['batchSize'], jparams['test_batchSize'],
                                              jparams['fcLayers'][-1], data_device, val_split=jparams.get('val_split', 0))
rho, rhop = def_rho(jparams['activation_function'])

if __name__ == '__main__':
    jparams['fcLayers'].reverse()  # output layer first, input layer last
    jparams['lr'].reverse()        # lr = [first hidden, ..., output] in the configuration

    if args.out_dir:
        BASE_PATH = os.path.abspath(args.out_dir)
        os.makedirs(BASE_PATH, exist_ok=True)
    else:
        BASE_PATH = new_run_dir('ep-')
    with open(os.path.join(BASE_PATH, "config.json"), "w") as outfile:
        json.dump(jparams, outfile)

    torch.manual_seed(jparams['seed'])
    if jparams['device'] >= 0:
        torch.cuda.manual_seed(jparams['seed'])

    net = MacDynamic(jparams, rho, rhop)
    torch.save(net.state_dict(), os.path.join(BASE_PATH, 'model_state_dict_0.pt'))  # untrained network

    _, optimizer = define_optimizer(net, lr=jparams['lr'], optimizer_type=jparams['optimizer'],
                                    momentum=jparams.get('momentum', 0.0), nesterov=jparams.get('nesterov', False),
                                    weight_decay=jparams.get('weight_decay', 0.0))

    scheduler = None
    if jparams.get('lr_scheduler') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=jparams['epochs'],
                                                               eta_min=jparams.get('lr_min', 0.0))
    elif jparams.get('lr_scheduler') is not None:
        raise ValueError(f"lr_scheduler '{jparams['lr_scheduler']}' is not defined (use 'cosine' or null)")

    train_epoch = train_dyn_sequential_ep if jparams['continuous_update'] else train_dyn_ep
    train_error_list, train_loss_list, weight_norm_list, test_error_list = [], [], [], []
    data_frame = init_dataframe(BASE_PATH)
    best_test_error = 1.0

    for epoch in tqdm(range(jparams['epochs'])):
        train_error, train_loss = train_epoch(net, train_loader, optimizer)
        if scheduler is not None:
            scheduler.step()
        train_error_list.append(train_error.item())
        train_loss_list.append(train_loss)

        weight_norm = torch.norm(net.W[0].weight) / np.sqrt(jparams['fcLayers'][0] * jparams['fcLayers'][1])
        weight_norm_list.append(weight_norm.item())

        test_error = test_dyn_ep(net, test_loader)
        test_error_list.append(test_error.item())
        if test_error.item() < best_test_error:
            best_test_error = test_error.item()
            torch.save(net.state_dict(), os.path.join(BASE_PATH, 'model_state_dict_best.pt'))

        data_frame = update_dataframe(BASE_PATH, data_frame, train_error_list, weight_norm_list, test_error_list,
                                      train_loss_list)
        torch.save(net.state_dict(), os.path.join(BASE_PATH, 'model_state_dict.pt'))
