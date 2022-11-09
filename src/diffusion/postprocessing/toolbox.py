import sklearn.decomposition
import os
import numpy as np
from diffusion.models.recurrent import RNN
from diffusion.utils import *
from diffusion.blocks import *

torch.set_grad_enabled(False)

def load_model(result_dir, model_name, model_version='best_model.pth', device='cuda:0', loader=None):
    model_path = os.path.join(result_dir, model_name, model_version)
    state_dict = torch.load(model_path)

    config_path = os.path.join(result_dir, model_name, 'config.yaml')
    config = read_yaml(config_path)

    torch.manual_seed(config['seed'])

    if loader is None:
        loader = get_data_loader(config)

    model = get_model(config, in_dim=loader.data_dim, out_dim=loader.n_classes, vocab_size=vocab_size).to(device)

    model.load_state_dict(state_dict)
    torch.set_grad_enabled(False)

    return model, loader

def evaluate(model, dataset, recurrence=None):
    acc = 0
    ce = 0
    counter = 0
    for minibatch in dataset:
        batch_size = minibatch['target'].size(0)
        stats = model.evaluate(minibatch, recurrence=recurrence)
        acc += float(stats['accuracy']) * batch_size
        ce += float(stats['cross_entropy']) * batch_size
        counter += batch_size

    cost = get_computational_cost(model, dataset)
    acc = acc / counter * 100
    ce /= counter
    return acc, ce, cost

def get_timestamps(result_dir, model_dir):
    _, timestamps, _ = next(os.walk(os.path.join(result_dir, model_dir)))
    timestamps = sorted(timestamps)
    return timestamps

def load_and_evaluate_dir(result_dir, model_dir, crit_estim=None, use_embedding=False, full_return=False, print_best=False):
    timestamps = get_timestamps(result_dir, model_dir)

    acc_list = []
    ce_list = []
    step_list = []
    for timestamp in timestamps:
        model_name = os.path.join(model_dir, timestamp)
        model, loader = load_model(result_dir, model_name)
        model.use_embedding = use_embedding
        if crit_estim is not None and model.stop_crit == 'first_correct':
            crit_estim = get_recurrence_estimator(model, loader, crit_estim)
            model.crit_estim = crit_estim
        acc, ce, steps = evaluate(model, loader.valid)
        acc_list.append(acc)
        ce_list.append(ce)
        step_list.append(steps)

    if print_best:
        print(np.array(timestamps)[np.argsort(ce_list)[:2]])

    if full_return:
        return np.stack((acc_list, ce_list, step_list), axis=0)
    else:
        print('-' * 5)
        print(f'accuracy: {np.around(np.mean(acc_list), 2): .2f} +- {np.around(np.std(acc_list), 2): .2f}')
        print(f'cross entropy: {np.around(np.mean(ce_list), 3): .3f} +- {np.around(np.std(ce_list), 3): .3f}')
        if step_list[0] > 0:
            print(f'computational cost: {np.around(np.mean(step_list), 3): .3f} +- {np.around(np.std(step_list), 3): .3f}')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number parameters:', n_params)

    return acc_list

def get_final_steps(model, input, logits_stacked):
    final_steps = model.get_final_steps(input, logits_stacked, None)
    return final_steps

def get_recurrence_estimator(model, loader, estimator, n_samples=-1, verbose=False, standardize=False):
    if verbose:
        print('training estimator')

    x_data, y_data, _ = get_final_steps_data(model, loader.train)
    x_data = x_data.cpu()
    if standardize:
        scaler = sklearn.preprocessing.StandardScaler()
        x_data = scaler.fit_transform(x_data)
        model.scaler = scaler

    estimator.fit(x_data[:n_samples], y_data.cpu()[:n_samples])
    if verbose:
        print('testing estimator')
    x_data, y_data, _ = get_final_steps_data(model, loader.valid)
    x_data = x_data.cpu()
    if standardize:
        x_data = scaler.transform(x_data)

    acc = estimator.score(x_data, y_data.cpu())
    if verbose:
        print('recurrence estimator test accuracy:', acc)
    return estimator

def get_final_steps_data(model, dataset):
    x_data = []
    y_data = []
    c_data = []

    for i, minibatch in enumerate(dataset):
        target = minibatch['target']
        input = minibatch['input']
        logits_stacked = model(input)
        final_steps = model.get_final_steps(input, logits_stacked, target)
        if hasattr(model, 'use_embedding'):
            if model.use_embedding:
                input = model.fc_layers(input)


        x_data.append(input)
        y_data.append(final_steps)
        c_data.append(target)


    x_data = torch.cat(x_data, dim=0)
    y_data = torch.cat(y_data, dim=0)
    c_data = torch.cat(c_data, dim=0)

    return x_data, y_data, c_data

def get_computational_cost(model, dataset=None, verbose=False):
    n_ops_total = 0
    if not isinstance(model, RNN):
        for layer in model.get_logits:
            if hasattr(layer, 'weight'):
                n_ops_total += layer.weight.size(0) * layer.weight.size(1)
        return n_ops_total
    else:
        # get final steps
        _, final_steps, _ = get_final_steps_data(model, dataset)
        final_steps = final_steps.cpu()

        ### FEATURE EMBEDDINGS
        n_ops_fc = 0
        for layer in model.fc_layers:
            if hasattr(layer, 'weight'):
                n_ops_fc += layer.weight.size(0) * layer.weight.size(1)
        n_ops_total += n_ops_fc

        ### RECURRENT BLOCK
        # get final step distribution
        final_step_distr = torch.zeros(model.max_rec_lim)
        values, counts = torch.unique(final_steps, return_counts=True)
        final_step_distr[values] = counts / torch.sum(counts)

        # calculate operations
        n_ops_rec = 0
        if isinstance(model.rnn_layer, MyRNN):
            in_dim = model.rnn_layer.in_dim
            hidden_dim = model.rnn_layer.hidden_dim
            for t, final_step_frac in enumerate(final_step_distr):
                if t == 0:
                    n_ops_rec += in_dim**2 * final_step_frac
                elif t == 1:
                    n_ops_rec += in_dim**2 + 2 * in_dim * hidden_dim * final_step_frac
                else:
                    n_ops_rec += in_dim**2 + 2 * in_dim * hidden_dim + (t - 1) * (hidden_dim**2 + hidden_dim * in_dim)\
                        * final_step_frac
        else:
            for t, final_step_frac in enumerate(final_step_distr):
                n_ops_rec += model.rnn_dim**2 * (t + 1) * final_step_frac

        n_ops_total += n_ops_rec

        ### OUTPUT LAYER
        n_ops_out = 0
        # operations in layer
        for layer in model.get_logits:
            if hasattr(layer, 'weight'):
                n_ops_out += layer.weight.size(0) * layer.weight.size(1)
        # how often output layer is calculated
        if model.stop_crit == 'threshold':
            n_times_out = torch.arange(model.max_rec) + 1
            n_ops_total += torch.sum(n_times_out * final_step_distr) * n_ops_out
        else:
            n_ops_total += n_ops_out

        ### LEARNABLE FUNCTION
        n_ops_rec_fn = 0
        if model.stop_crit == 'learnable':
            for layer in model.get_final_step_probs:
                if hasattr(layer, 'weight'):
                    n_ops_rec_fn += layer.weight.size(0) * layer.weight.size(1)
                if isinstance(layer, RBF):
                    n_ops_rec_fn += layer.center.size(1) * layer.center.size(2)
            n_ops_total += n_ops_rec_fn

        if verbose:
            print('fc layer:', n_ops_fc)
            print('rnn layer:', n_ops_rec)
            print('out layer:', n_ops_out)
            print('rec_fn layer:', n_ops_rec_fn)

        return n_ops_total


def generate_latex_table_line(name, acc_list, ce_list, step_list):
    """name & acc & ce & cost & acc/cost"""
    line = name + ' &'
    line += f'{np.around(np.mean(acc_list), 2): .2f} \\pm {np.around(np.std(acc_list), 2): .2f} & '
    line += f'{np.around(np.mean(ce_list), 2): .2f} \\pm {np.around(np.std(ce_list), 2): .2f} & '
    line += f'{np.around(np.mean(step_list), 2): .2f} \\pm {np.around(np.std(step_list), 2): .2f} & '
    acc_cost_ratio = np.array(acc_list) / np.array(step_list)
    line += f'{np.around(np.mean(acc_cost_ratio), 2): .2f} \\pm {np.around(np.std(acc_cost_ratio), 2): .2f} \\\\'
    # best_index = np.argmax(acc_list)
    # line += f'{np.around(acc_list[best_index], 2): .2f} & '
    # line += f'{np.around(step_list[best_index], 2): .2f} & '
    # cheapest_index = np.argmin(step_list)
    # line += f'{np.around(acc_list[cheapest_index], 2): .2f} & '
    # line += f'{np.around(step_list[cheapest_index], 2): .2f} \\\\'

    print(line)



def make_table(result_dir, model_type, crit_estim=None, use_embedding=False):
    print('\\begin{center}')
    print('\\begin{tabular}{ |l||l|l|l|l| }')
    print('\\hline')
    print(' & & & & ACC/ \\\\')
    print('model & ACC & CE & OPS & OPS \\\\')
    print('\\hline')
    _, model_paths, _ = next(os.walk(os.path.join(result_dir, model_type)))
    model_paths = sorted(model_paths)
    for model_path in model_paths:
        model_dir = os.path.join(model_type, model_path)
        name = make_name(model_dir)
        metrics = load_and_evaluate_dir(result_dir, model_dir, crit_estim, use_embedding, full_return=True)
        generate_latex_table_line(name, *metrics)

        print('\\hline')

    print('\\end{tabular}')
    print('\\end{center}')

def make_name(path):
    name = ''
    if 'linear' in path:
        name += 'sRNN '
    elif 'myrnn1' in path:
        name += 'hRNN '

    if 'learnable' in path:
        name += 'learnable '
        if 'mlp' in path:
            name += 'MLP '
        elif 'rbf' in path:
            name += 'RBF '
        if 'none' in path:
            name += 'loss '
        elif 'first_correct' in path:
            name += 'first\_correct '
    elif 'first_correct' in path:
        name += 'reg '
    elif 'threshold' in path:
        name += 'treshold '
        if '0.5' in path:
            name += '0.5 '
        if '0.6' in path:
            name += '0.6 '
        if '0.7' in path:
            name += '0.7 '
        if '0.8' in path:
            name += '0.8 '
        if '0.9' in path:
            name += '0.9 '

    if 'max_rec_3' in path:
        name += '$T=3$'
    elif 'max_rec_5' in path:
        name += '$T=5$'
    else:
        name += '$T=2$'
    return name
