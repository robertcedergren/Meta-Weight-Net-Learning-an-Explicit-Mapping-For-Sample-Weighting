
import time

import numpy as np

from tqdm import trange

from torch.autograd import Variable

import os

from data_preprocessing import *
from models import *
from plots import *
from train_eval import *


def perform_experiment(args, train_loader, meta_loader, test_loader, corrupted_data_loader, model, meta_model,
                       meta_weight_net, optimizers, loss_functions):
    def compute_accuracy(torch_predictions, torch_targets):
        predictions = Variable(torch.cat(torch_predictions)).cpu().numpy()
        targets = Variable(torch.cat(torch_targets)).cpu().numpy()
        accuracy = np.sum(np.where(targets == predictions, 1, 0)) / targets.size
        return accuracy

    history = {'train_losses': [],
               'train_losses_weighted': [],
               'meta_losses': [],
               'test_losses': [],
               'train_accuracy': [],
               'test_accuracy': [],
               'test_targets': []}

    best_test_model_acc = -np.inf

    best_model_info = {'best_test_model': None,
                       'best_test_model_predictions': None,
                       'best_test_model_targets': None,
                       'best_test_model_epoch': 0,
                       'best_mlp_model': None}

    if args.model_type == 'MWN':
        if args.experiment_type == "Uniform noise" or args.experiment_type == "Flip noise":
            w_previous = 0
            history['weight_variation_means'] = []
            history['weight_variation_stds'] = []

    epochs = trange(args.epochs_dic[args.experiment_type], position=1, leave=True)
    for epoch in epochs:
        print("\n\nEpoch lr(model)\n\n", epoch, optimizers['model'].param_groups[0]['lr'])

        if epoch + 1 in args.lr_schedule:
            optimizers['model'].param_groups[0]['lr'] = args.lr_schedule[epoch + 1]
            print("\n\nEpoch lr(model)\n\n", epoch, optimizers['model'].param_groups[0]['lr'])

        time_start = time.time()

        # training
        if args.model_type == 'MWN':
            model, meta_weight_net, train_loss, train_loss_weighted, meta_loss, train_predictions, train_targets = meta_training(
                args, train_loader, meta_loader, model, meta_model, meta_weight_net, optimizers, loss_functions)
        elif args.model_type == 'Baseline':
            model, train_loss, train_predictions, train_targets = base_training(args, train_loader, model, optimizers,
                                                                                loss_functions)
        elif args.model_type == 'FineTune':
            # fine_tune_training()
            pass

        train_accuracy = compute_accuracy(train_predictions, train_targets)

        # evaluation on test set
        test_loss, test_predictions, test_targets = evaluate(args, test_loader, model, loss_functions['model'])
        test_accuracy = compute_accuracy(test_predictions, test_targets)

        if args.model_type == 'MWN':
            # weight variation
            if args.experiment_type == "Uniform noise" or args.experiment_type == "Flip noise":
                if args.factor != 0:
                    w, w_variation_mean, w_variation_std = weight_variation(args, w_previous, corrupted_data_loader,
                                                                            model,
                                                                            meta_weight_net, loss_functions['model'])
                    history['weight_variation_means'].append(w_variation_mean)
                    history['weight_variation_stds'].append(w_variation_std)
                    w_previous = w

        time_for_epoch = time.time() - time_start

        # update progress bar
        epochs.set_description(
            "Time for epoch: {}\nTrain loss: {} Train loss weighted: {} Meta loss: {} Test loss: {}\nTrain acc: {} Test acc: {}".format(
                time_for_epoch, train_loss,
                train_loss_weighted if args.model_type == "MWN" else "",
                meta_loss if args.model_type == "MWN" else "",
                test_loss, train_accuracy, test_accuracy))

        with open(os.path.join(args.directory, "logs.txt"), "a") as fl:
            print(
                "\nTime for epoch: {}\nTrain loss: {} Train loss weighted: {} Meta loss: {} Test loss: {}\nTrain acc: {} Test acc: {}\n".format(
                    time_for_epoch, train_loss,
                    train_loss_weighted if args.model_type == "MWN" else "",
                    meta_loss if args.model_type == "MWN" else "",
                    test_loss, train_accuracy, test_accuracy), file=fl)

        if test_accuracy > best_test_model_acc:
            best_model_info['best_test_model'] = model.state_dict()
            best_model_info['best_test_model_epoch'] = epoch + 1
            best_model_info['best_test_model_predictions'] = torch.cat(test_predictions).cpu().numpy()
            best_model_info['best_test_model_targets'] = torch.cat(test_targets).cpu().numpy()
            best_model_info['best_test_model_accuracy'] = test_accuracy
            best_model_info['best_test_model_loss'] = test_loss
            if args.model_type == "MWN":
                best_model_info["best_mlp_model"] = meta_weight_net.state_dict()

            torch.save(best_model_info, os.path.join(args.directory, 'best_model_info'))

        torch.save(model.state_dict(), os.path.join(args.directory, 'current_model'))

        # append history
        history['train_losses'].append(train_loss)
        if args.model_type == 'MWN':
            history['train_losses_weighted'].append(train_loss_weighted)
            history['meta_losses'].append(meta_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)

        torch.save(history, os.path.join(args.directory, 'history'))
        torch.save(args, os.path.join(args.directory, 'args'))

    # create confusion matrix
    cm_counts_test, cm_percent_test = compute_confusion_matrix(true_targets=best_model_info['best_test_model_targets'],
                                                               pred_targets=best_model_info[
                                                                   'best_test_model_predictions'])
    # plot and saves confusion matrix
    plot_confusion_matrix(cm_percent_test, args, close=False)

    if args.model_type == 'MWN':
        # plot weight variation curves
        if args.experiment_type == "Uniform noise" or args.experiment_type == "Flip noise":
            plot_weight_variation_curves(np.array(history['weight_variation_means']),
                                         np.array(history['weight_variation_stds']),
                                         args, close=False)

        # plot MW-Net function learned
        plot_mwn_function(args=args,
                          meta_weight_net_state=best_model_info["best_mlp_model"],
                          close=False)

    # plot train and meta losses
    plot_train_and_meta_loss(train_loss=history['train_losses'],
                             meta_loss=history['meta_losses'] if args.model_type == "MWN" else [],
                             args=args, close=False)



