from copy import deepcopy
from data_preprocessing import *
from torch.utils.data import DataLoader


def loading_data(args):
    kwargs_dataloader = {'batch_size': args.batch_size,
                         'shuffle': True}

    train_data, test_data = args.get_dataset_function_dict[args.dataset](args.cifar_type,
                                                                         train_transform=
                                                                         args.transformations[args.dataset][
                                                                             'train'],
                                                                         test_transform=
                                                                         args.transformations[args.dataset][
                                                                             'test'])

    kwargs_data_functions = {'dataset': deepcopy(train_data),
                             'factor': args.factor,
                             'num_meta_per_class': args.num_meta_per_class,
                             'num_of_corrupted': args.num_of_corrupted,
                             'seed': args.seed}

    args.data_function = {'Imbalance': generate_imbalance_data,
                          'Uniform noise': generate_noise_data,
                          'Flip noise': generate_flip_data,
                          'Clothing 1M': None
                          }

    data, meta, corrupted_data = args.data_function[args.experiment_type](**kwargs_data_functions)

    train_loader = DataLoader(data, pin_memory=args.pin_memory,
                              num_workers=args.num_workers,
                              **kwargs_dataloader)

    test_loader = DataLoader(test_data, pin_memory=args.pin_memory,
                             num_workers=args.num_workers,
                             **kwargs_dataloader)

    return train_loader, test_loader, meta, corrupted_data
