import matplotlib.pyplot as plt
import sklearn.decomposition
import os
import numpy as np
import torch
from diffusion.utils import *
from diffusion.models.entropy import *
from scipy.integrate import simpson


torch.set_grad_enabled(False)

def load_model(result_dir, model_name, model_version='best_model.pth', device='cuda:0', loader=None):
    model_path = os.path.join(result_dir, model_name, model_version)
    state_dict = torch.load(model_path)

    config_path = os.path.join(result_dir, model_name, 'config.yaml')
    config = read_yaml(config_path)

    torch.manual_seed(config['seed'])

    if loader is None:
        loader = get_data_loader(config)

    model = get_model(config, data_shape=loader.data_shape).to(device)

    model.load_state_dict(state_dict)
    torch.set_grad_enabled(False)

    return model, loader

def plot_things(model, loader, convergence_threshold=.01, do_wasserstein=False):
    current_list = []
    for minibatch in loader.test:
        data = minibatch['data'].to(model.device)  # [B,T,2,D]
        time_point = minibatch['time_point'].to(model.device)  # [B,T,2]
        current = model.forward((data, time_point))  # [B,T,2,D], [B,T,2] -> [B,T]
        current_list.append(current)

    current = torch.concat(current_list, dim=0).cpu()
    dt = (time_point[:, :, 1] - time_point[:, :, 0])[0].cpu()
    t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2
    entropy_production_rate = (torch.max(model.estimators['var'](current), dim=-1).values / dt)

    plt.figure(figsize=(20, 20), dpi=200)
    plt.rcParams['font.size'] = 15

    if hasattr(loader.train.dataset, 'loss'):
        loss = loader.train.dataset.loss
        entropy_production_rate_truncated, cutoff_idx = remove_constant_epr(entropy_production_rate.clone(), loss, convergence_threshold)
        #plt.plot(t_eval, torch.log(loss[:-1]), label=f'log loss')
        plt.plot(t_eval, loss[:-1], label=f'loss')
        plt.axvline(x=t_eval[cutoff_idx], color='violet', label='cutoff time')

    if hasattr(loader.train.dataset, 'mi'):
        mutual_information = loader.train.dataset.mi.cpu()
        plt.plot(t_eval.numpy(), mutual_information[:-1].numpy(),
                 label=f'mutual information')

    if hasattr(loader.train.dataset, 'bounds') and do_wasserstein:
        wasserstein_bounds = loader.train.dataset.bounds.cpu()
        plt.plot(t_eval.numpy(), wasserstein_bounds[:, 0].numpy(),
                 label=f'wasserstein bound 1')
        plt.plot(t_eval.numpy(), wasserstein_bounds[:, 1].numpy(),
                 label=f'wasserstein bound 2')
        plt.plot(t_eval.numpy(), wasserstein_bounds[:, 2].numpy(),
                 label=f'wasserstein bound 3')

    plt.plot(t_eval, entropy_production_rate, label=f'epr')
    plt.plot(t_eval, entropy_production_rate_truncated, label=f'epr truncated')

    total_entropy_production = []
    for i in range(1, len(t_eval)):
        total_entropy_production.append(simpson(entropy_production_rate_truncated[:i],
                                                t_eval[:i], even="avg"))
    plt.plot(t_eval[:-1], total_entropy_production,
             label=f'total ep')

    plt.ylim(0, np.max(total_entropy_production) + 3)
    plt.xlabel('time')
    plt.legend()

    print('mi over ep:', float(mutual_information[-1] / total_entropy_production[-1]))
    if hasattr(loader.train.dataset, 'bounds') and do_wasserstein:
        print('wasserstein 1 over ep:', float(wasserstein_bounds[-1, 0] / total_entropy_production[-1]))
        print('wasserstein 2 over ep:', float(wasserstein_bounds[-1, 1] / total_entropy_production[-1]))
        print('wasserstein 3 over ep:', float(wasserstein_bounds[-1, 2] / total_entropy_production[-1]))

def remove_constant_epr(entropy_production_rate, loss, convergence_threshold):
    # find convergence point in loss
    convergence_idx = torch.argmax((torch.abs(loss[:-10] - loss[10:]) < convergence_threshold).float())
    entropy_production_rate[convergence_idx:] = 0
    return entropy_production_rate, convergence_idx

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
