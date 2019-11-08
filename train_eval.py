from copy import deepcopy

from tqdm.auto import tqdm

import torch
from torch.autograd import Variable


def base_training(args, train_loader, model, optimizers, loss_functions):
    train_loss = 0
    train_predictions = []
    train_targets = []

    model.train()

    for enum, (x, y) in tqdm(enumerate(train_loader), position=0, leave=True):
        x, y = x.to(args.cuda), y.to(args.cuda)
        y = y.long()

        # forward pass for meta-model using input data
        y_pred = model(x)
        loss = torch.mean(loss_functions['model'](y_pred, y))

        train_loss += loss.data.cpu().numpy()

        # backward pass for model
        optimizers['model'].zero_grad()
        loss.backward()

        # update parameters of model
        optimizers['model'].step()

        train_predictions.append(torch.argmax(y_pred, dim=1))
        train_targets.append(y)

    train_loss /= len(train_loader)

    return model, train_loss, train_predictions, train_targets


"""## Meta-training"""


def meta_training(args, train_loader, meta_loader, model, meta_model, meta_weight_net, optimizers, loss_functions):
    train_loss = 0
    train_loss_weighted = 0
    meta_loss = 0
    train_predictions = []
    train_targets = []

    model.train()

    eta = deepcopy(optimizers["model"].param_groups[0]['lr'])

    for enum, (x, y) in tqdm(enumerate(train_loader), position=0, leave=True):
        x_meta, y_meta = next(iter(meta_loader))
        y, y_meta = y.long(), y_meta.long()
        x, y, x_meta, y_meta = x.to(args.cuda), y.to(args.cuda), x_meta.to(args.cuda), y_meta.to(args.cuda)

        ### Step 5
        y_p = model(x)
        loss = loss_functions["model"](y_p, y)
        weights = meta_weight_net(loss.reshape(-1, 1))
        normalized_weights = (weights / (torch.sum(weights) if torch.sum(weights) != 0 else args.tau)).reshape(-1)
        weighted_loss = torch.sum(normalized_weights * loss)

        model.zero_grad()
        meta_weight_net.zero_grad()
        grads = torch.autograd.grad(weighted_loss, (model.parameters()), create_graph=True)

        ### Step 6
        y_meta_pred = meta_model(x_meta, model, eta, grads)
        loss_meta = torch.mean(loss_functions['model'](y_meta_pred, y_meta))

        meta_weight_net.zero_grad()
        meta_model.zero_grad()
        loss_meta.backward()
        optimizers["meta_weight_net"].step()

        meta_loss += loss_meta.data.cpu().numpy()

        ### Step 7
        y_pred = model(x)
        loss = loss_functions['model'](y_pred, y)

        with torch.no_grad():
            weights = meta_weight_net(loss.reshape(-1, 1))
        normalized_weights = (weights / (torch.sum(weights) if torch.sum(weights) != 0 else args.tau)).reshape(-1)
        weighted_loss = torch.sum(loss * normalized_weights)

        # backward pass for model
        optimizers['model'].zero_grad()
        weighted_loss.backward()

        # update parameters of model
        optimizers['model'].step()

        train_loss += torch.mean(loss).data.cpu().numpy()
        train_loss_weighted += weighted_loss.data.cpu().numpy()

        train_predictions.append(torch.argmax(y_pred, dim=1))
        train_targets.append(y)

    train_loss /= len(train_loader)
    train_loss_weighted /= len(train_loader)
    meta_loss /= len(train_loader)

    return model, meta_weight_net, train_loss, train_loss_weighted, meta_loss, train_predictions, train_targets


"""# Evaluation"""


def evaluate(args, data_loader, model, loss_function):
    predictions = []
    targets = []

    evaluation_loss = 0

    with torch.no_grad():
        model.eval()
        tqdm_data_loader = tqdm(data_loader, position=0, leave=True)
        for enum, (x_eval, y_eval) in enumerate(tqdm_data_loader):
            x_eval, y_eval = x_eval.to(args.cuda), y_eval.to(args.cuda)
            y_eval = y_eval.long()

            # forward pass
            y_pred = model(x_eval)

            # compute loss
            loss = torch.mean(loss_function(y_pred, y_eval))

            evaluation_loss += loss.data.cpu().numpy()

            predictions.append(torch.argmax(y_pred, dim=1))
            targets.append(y_eval)

    evaluation_loss /= len(data_loader)
    return evaluation_loss, predictions, targets


def weight_variation(args, w_previous, corrupted_data_loader, model, meta_weight_net, model_loss_function):
    x, y_noisy = next(iter(corrupted_data_loader))
    x, y_noisy = x.to(args.cuda), y_noisy.to(args.cuda)
    y_noisy = y_noisy.long()

    with torch.no_grad():
        model.eval()
        y_pred = model(x)
        loss = model_loss_function(y_pred, y_noisy)
        w = meta_weight_net(loss.reshape(-1, 1))

        w /= torch.sum(w) if torch.sum(w) != 0 else args.tau

    w_variation = w - w_previous
    w_variation_mean = Variable(torch.mean(w_variation)).cpu().numpy()
    w_variation_std = Variable(torch.std(w_variation)).cpu().numpy()

    return w, w_variation_mean, w_variation_std
