import os
import json
import torch
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from tqdm import tqdm

from diffusion.utils.helpers import *
from diffusion.postprocessing.utils import load_trainer, get_j_t






def plot_epr(trainer_list, label_list, zorder_list=None, estimator='var', plot_exact=True, linestyles=None, colors=None):
    if zorder_list is None:
        zorder_list = [None] * len(trainer_list)
    if linestyles is None:
        linestyles = [None] *( len(trainer_list) + 1)
    if colors is None:
        colors = ['grey', 'blue', 'red', 'black']

    for trainer, label, zorder, linestyle, color in zip(trainer_list, label_list, zorder_list, linestyles, colors):
        j, t = get_j_t(trainer)

        model, data_loader = trainer.model, trainer.data_loader

        t_eval = (t[0, :, 0] + t[0, :, 1]).cpu() / 2

        # EPR
        if estimator == 'var':
            plot_epr_var(model, j, t, t_eval, label=label, zorder=zorder, linestyle=linestyle, color=color)
        elif estimator == 'simple':
            plot_epr_simple(model, j, t, t_eval, label=label, linestyle=linestyle, color=color)
    if plot_exact:
        plot_exact_epr(data_loader, t_eval, linestyles[-1])

    trainer.model._set_loss_weight(t)
    if hasattr(trainer.model, 'loss_weight'):
        plt.plot(t_eval, trainer.model.loss_weight.cpu(), label='loss weight')


    plt.legend()
    plt.xlabel('time')
    plt.ylabel('entropy production rate')

def get_epr_error(trainer_list):
    for trainer in trainer_list:
        j, t = get_j_t(trainer)

        trainer.model.estimator_type = 'var'
        epr_var = trainer.model.get_entropy_production_rate(j, t)

        trainer.model.estimator_type = 'simple'
        epr_simple = trainer.model.get_entropy_production_rate(j, t)

        epr_exact = trainer.data_loader.train.dataset.exact_epr

        epr_var_error = torch.mean(torch.abs(epr_var - epr_exact))
        epr_simple_error = torch.mean(torch.abs(epr_simple - epr_exact))

        epr_var_error_relative = epr_var_error / epr_exact
        epr_simple_error_relative = epr_simple_error / epr_exact

        print('#####################')
        print('model:', trainer.name)
        print('avg epr error (simple):', epr_simple_error)
        print('avg epr error (var):', epr_var_error)
        print('avg relative epr error (simple):', epr_simple_error_relative)
        print('avg relative epr error (var):', epr_var_error_relative)

def get_trainer_list(result_dir, model_name_list):
    trainer_list = []
    for model_name in model_name_list:
        trainer = load_trainer(result_dir, model_name)
        trainer_list.append(trainer)
    return trainer_list


def plot_epr_var(model, j, t, t_eval, label='estimated epr (var)', zorder=None, linestyle='solid', color=None):
    model.estimator_type = 'var'
    epr_var = model.get_entropy_production_rate(j, t).cpu()
    plt.plot(t_eval, epr_var, label=label, linewidth=1, zorder=zorder, linestyle=linestyle, c=color)
    return epr_var

def plot_epr_simple(model, j, t, t_eval, label='estimated epr (var)', linestyle=None, color=None):
    model.estimator_type = 'simple'
    epr_simple = model.get_entropy_production_rate(j, t).cpu()
    plt.plot(t_eval, epr_simple, label=label, linewidth=1, linestyle=linestyle, c=color)
    return epr_simple

def plot_exact_epr(data_loader, t_eval, linestyle='solid'):
    if hasattr(data_loader.train.dataset, 'exact_epr'):
        exact_epr = data_loader.train.dataset.exact_epr
        plt.plot(t_eval, exact_epr, label=f'exact', linewidth=1, linestyle=linestyle, c='black')
        return exact_epr

def modified_var(j, t):
    dt = (t[:, :, 1] - t[:, :, 0])[0]
    estimate = .5 * torch.mean((j[:-1] + j[1:])**2, dim=0) + torch.mean(j[:-1] * j[1:], dim=0) / dt
    return torch.cat((torch.zeros(1, device=estimate.device), estimate))

def get_ml_solution(path_to_data):
    with open(os.path.join(path_to_data, 'max_likelihood_sol.json'), 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':

    result_dir = '/rdata/results/entropy/ppca'
    model_name = 'no_clip/_optimizer_lr_0.01/_model_layer_dims_(128, 128)/0915-213135'

    gaussian_path = '/raid/data/gaussian_linear'

    ml_dict = get_ml_solution(gaussian_path)
    data_cov = ml_dict['ground_truth_data_cov_matrix']
    data_mean = ml_dict['mean_ml']



