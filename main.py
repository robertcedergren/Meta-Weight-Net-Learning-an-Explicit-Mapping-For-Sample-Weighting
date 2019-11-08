import os
import argparse
import platform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datetime import datetime
from loading_data import loading_data
from data_preprocessing import *
from models import *
from perform_experiment import perform_experiment


def create_directories(path):
    lst_directories = []
    while os.path.split(path)[1] != "":
        temp_path = os.path.split(path)
        path = temp_path[0]
        lst_directories.append(temp_path[1])
    lst_directories.reverse()
    for n in range(len(lst_directories)):
        path = os.path.join(*lst_directories[:n + 1])
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar_type", type=int, default=10)  # 100
    parser.add_argument("--experiment_type", type=str, default="Imbalance")  # "Uniform noise" # "Flip noise"
    parser.add_argument("--model_type", type=str, default="MWN")  # Baseline, FineTune
    parser.add_argument("--factor", type=float, default=200.0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=12345)
    args_out = parser.parse_args()

    if platform.system() == "Windows":
        args_out.num_workers = 0

    if not os.path.exists("Logs"):
        os.mkdir("Logs")

    """## Arguments In"""
    parser_in = argparse.ArgumentParser()
    args = parser_in.parse_args()


    args.tau = 1e-6
    args.momentum = 0.9
    args.nesterov = True

    args.dataset = 'CIFAR'
    args.cifar_type = args_out.cifar_type  # 10  # 100
    args.experiment_type = args_out.experiment_type  # "Imbalance"  # "Uniform noise" # "Flip noise"

    args.model_type = args_out.model_type  # "MWN"  # Baseline, FineTune
    args.seed = args_out.seed
    print(f"seed:\t{args.seed}")
    args.num_meta_per_class = 10
    args.num_of_corrupted = 10

    args.num_workers = args_out.num_workers  # 6
    args.pin_memory = True
    args.cuda = "cuda:0" if torch.cuda.is_available() else "cpu"

    args.model_signature = str(datetime.now())[0:19].replace(':', '.').replace(" ", "_")

    args.factor = args_out.factor  # 200

    cifar_type = str(args.cifar_type) if args.dataset == 'CIFAR' else ""

    args.directory = os.path.join("Experiments",
                                  args.experiment_type.replace(" ", "_"),
                                  args.dataset + '_'.join(
                                      ['', cifar_type, args.experiment_type.replace(" ", "_"),
                                       str(args.factor), args.model_type]),
                                  args.model_signature
                                  )

    print(f"saving directory:\t{args.directory}")

    args.batch_size_data_dic = {'Imbalance': 100,
                                'Uniform noise': 100,
                                'Flip noise': 100,
                                'Clothing 1M': 32
                                }

    args.batch_size_meta_data_dic = {'Imbalance': 100,
                                     'Uniform noise': 100,
                                     'Flip noise': 100,
                                     'Clothing 1M': 32
                                     }

    args.lr_model_dic = {'Imbalance': {0: 0.1, 80: 0.01, 90: 1e-3},
                         'Uniform noise': {0: 0.1, 36: 0.01, 38: 1e-3},
                         'Flip noise': {0: 0.1, 40: 0.01, 50: 1e-3},
                         'Clothing 1M': {0: 0.01, 5: 0.001}
                         }

    args.lr_wnet_dic = {'Imbalance': 1e-5,
                        'Uniform noise': 1e-3,
                        'Flip noise': 1e-3,
                        'Clothing 1M': 1e-3
                        }

    args.epochs_dic = {'Imbalance': 100,
                       'Uniform noise': 40,
                       'Flip noise': 60,
                       'Clothing 1M': 10
                       }

    args.weight_decay_dic = {'Imbalance': 5e-4,
                             'Uniform noise': 5e-4,
                             'Flip noise': 5e-4,
                             'Clothing 1M': 1e-3
                             }

    args.lr_schedule = args.lr_model_dic[args.experiment_type]
    args.batch_size = args.batch_size_data_dic[args.experiment_type]

    kwargs_dataloader = {'batch_size': args.batch_size,
                         'shuffle': True}

    kwargs_optimizer = {'lr': args.lr_model_dic[args.experiment_type][0],
                        'momentum': args.momentum,
                        'nesterov': True,
                        'weight_decay': args.weight_decay_dic[args.experiment_type]}

    kwargs_optimizer_wnet = {'lr': args.lr_wnet_dic[args.experiment_type],
                             'momentum': args.momentum,
                             'nesterov': True,
                             'weight_decay': args.weight_decay_dic[args.experiment_type]}

    args.transformations = {'CIFAR': {'train': train_transform, 'test': test_transform},
                            'Clothing 1M': {'train': None, 'test': None}}

    args.get_dataset_function_dict = {'CIFAR': get_CIFAR_data, 'Clothing 1M': None}

    train_loader, test_loader, meta, corrupted_data = loading_data(args)

    if args.experiment_type == "Imbalance":
        args.num_meta_per_class = 10
    else:
        if args.cifar_type == 10:
            args.num_meta_per_class = 100
        else:
            args.num_meta_per_class = 10

    create_directories(args.directory)

    print(f"cifar type:\t{args_out.cifar_type}")
    print(f"experiment type:\t{args_out.experiment_type}")
    print(f"model type:\t{args_out.model_type}")
    print(f"factor:\t{args_out.factor}")
    print(f"number of workers:\t{args_out.num_workers}")

    meta_loader = None
    if args.factor == 0 or args_out.factor == 1:
        corrupted_data_loader = None
    else:
        corrupted_data_loader = DataLoader(corrupted_data,
                                           pin_memory=False,
                                           num_workers=0,
                                           # num_workers=args.num_workers,
                                           **kwargs_dataloader) if args.experiment_type != 'Imbalance' else None

    meta_model = None
    meta_weight_net = None

    if args.experiment_type == "Uniform noise":
        model = wide_resnet_28_10(args.cifar_type).to(args.cuda)
    else:
        model = resnet32(args.cifar_type).to(args.cuda)

    optimizers = {}
    optimizers['model'] = torch.optim.SGD(model.parameters(), **kwargs_optimizer)

    if args.model_type == "MWN":
        meta_loader = DataLoader(meta,
                                 num_workers=0,
                                 # pin_memory=True,
                                 **kwargs_dataloader)

        meta_weight_net = MLP().to(args.cuda)
        if args.experiment_type == "Uniform noise":
            meta_model = F_WideResNet(depth=28,
                                      widen_factor=10).to(args.cuda)
        else:
            meta_model = F_ResNet_32(F_BasicBlock, [5, 5, 5]).to(args.cuda)

        optimizers['meta_weight_net'] = torch.optim.SGD(meta_weight_net.parameters(), **kwargs_optimizer_wnet)

    loss_functions = {}
    loss_functions['model'] = nn.CrossEntropyLoss(reduction='none').to(args.cuda)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Perform experiment
    perform_experiment(args,
                       train_loader,
                       meta_loader,
                       test_loader,
                       corrupted_data_loader,
                       model,
                       meta_model,
                       meta_weight_net,
                       optimizers,
                       loss_functions)

