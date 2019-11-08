import matplotlib
matplotlib.use("Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from models import MLP

def plot_train_and_meta_loss(train_loss, meta_loss, args, close=True):
    cifar_type = args.cifar_type if args.dataset == 'CIFAR' else ""
    plt.figure()
    plt.title('{}{}-{}-{}'.format(args.dataset,
                                  cifar_type,
                                  args.experiment_type,
                                  args.factor))
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.plot(train_loss, label='training loss')
    if len(meta_loss) != 0:
        ## For Baseline model we do not plot the meta loss
        plt.plot(meta_loss, label='meta loss')
    plt.legend(loc='best')
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(os.path.join(args.directory, 'loss_plot.png'))
    if close: plt.close()


def plot_weight_variation_curves(weight_variation_means, weight_variation_stds, args, close=True):
    upper_bound = weight_variation_means + weight_variation_stds
    lower_bound = weight_variation_means - weight_variation_stds

    x = np.arange(1, len(weight_variation_means) + 1)

    cifar_type = args.cifar_type if args.dataset == 'CIFAR' else ""
    plt.figure()
    plt.title('{}{}-{}-{}'.format(args.dataset,
                                  cifar_type,
                                  args.experiment_type,
                                  args.factor))
    plt.ylabel('weights')
    plt.xlabel('epochs')
    plt.plot(x, weight_variation_means.tolist(), label='weight_variation', color='r')
    plt.fill_between(x=x,
                     y1=upper_bound,
                     y2=lower_bound,
                     facecolor='pink')
    plt.legend(loc='best')
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(os.path.join(args.directory, 'weight_variation_curve.png'))  # "factor_" + str(args.factor)
    if close: plt.close()


def compute_confusion_matrix(true_targets, pred_targets):
    matrix = confusion_matrix(y_true=true_targets.astype(int),
                              y_pred=pred_targets.astype(int))
    return matrix, matrix.astype('float') / np.sum(matrix, axis=1).reshape(-1, 1)


def plot_confusion_matrix(matrix, args, close=True):
    cifar_type = args.cifar_type if args.dataset == 'CIFAR' else ""
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='Blues', origin='lower')
    plt.colorbar(im)
    ax.set_title('{}-{}{}-{}-{}'.format(args.model_type,
                                        args.dataset,
                                        cifar_type,
                                        args.experiment_type,
                                        args.factor))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="r")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(os.path.join(args.directory, 'confusion_matrix.png'))
    if close: plt.close()


def plot_mwn_function(args, meta_weight_net_state, close=True):
    linspace_losses = torch.linspace(start=0, end=20, steps=100).to(args.cuda)
    temp_MWN = MLP().to(args.cuda)
    temp_MWN.load_state_dict(meta_weight_net_state)
    temp_MWN.eval()
    predicted_weights = temp_MWN(linspace_losses.reshape(-1, 1)).data.cpu()

    cifar_type = args.cifar_type if args.dataset == 'CIFAR' else ""
    plt.figure()
    plt.title('{}{}-{}-{}'.format(args.dataset,
                                  cifar_type,
                                  args.experiment_type,
                                  args.factor))
    plt.ylabel('weight')
    plt.xlabel('loss')
    plt.plot(linspace_losses.cpu(), predicted_weights, label='{}{}'.format(args.dataset,
                                                                           cifar_type))
    plt.legend(loc='best')
    plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.savefig(os.path.join(args.directory, 'MWN_function_plot.png'))
    if close: plt.close()

