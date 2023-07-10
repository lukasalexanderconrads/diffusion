import os

import torch
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from tqdm import tqdm

from diffusion.utils.helpers import *


def load_trainer(result_dir, model_name, model_version='best_model.pth', device='cuda:0'):
    model_path = os.path.join(result_dir, model_name, model_version)
    state_dict = torch.load(model_path)

    config_path = os.path.join(result_dir, model_name, 'config.yaml')
    config = read_yaml(config_path)
    config['device'] = device
    config['trainer']['args']['no_training'] = True

    torch.manual_seed(config['seed'])

    trainer = get_trainer(config)
    trainer.model.load_state_dict(state_dict)

    torch.set_grad_enabled(False)

    return trainer

def plot_mi_epr(trainer, set='test'):

    j, t = get_j_t(trainer, set)
    print('current sum ensemble:', torch.mean(torch.sum(j, dim=1)))
    model, data_loader = trainer.model, trainer.data_loader

    t_eval = (t[0, :, 0] + t[0, :, 1]).cpu() / 2

    # EPR
    epr_var = plot_epr_var(model, j, t, t_eval)

    exact_epr = plot_exact_epr(data_loader, t_eval)

    plt.legend()
    plt.xlabel('time')
    plt.ylabel('entropy production rate')
    plt.show()

    # MI
    plot_mi(data_loader, t_eval)

    # CUM EPR
    cum_epr_var = []
    for i in range(1, len(t_eval)):
        cum_epr_var.append(simpson(epr_var[:i],
                                         t_eval[:i], even="avg"))
    plt.plot(t_eval[:-1], cum_epr_var, label=f'estimated cumulative entropy production (var)')

    # plot exact cumulative entropy production if contained in data set
    if hasattr(data_loader.train.dataset, 'exact_epr'):
        exact_cum_epr = []
        for i in range(1, len(t_eval)):
            exact_cum_epr.append(simpson(exact_epr[:i],
                                         t_eval[:i], even="avg"))
        plt.plot(t_eval[:-1], exact_cum_epr,
                 label=f'exact cumulative entropy production')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('total entropy production')
    plt.show()

    # print('total ep (simple):', cum_epr_simple[-1])
    print('total ep (var):', cum_epr_var[-1])


    # if hasattr(data_loader.train.dataset, 'loss'):
    #     loss = data_loader.train.dataset.loss
    #     #plt.plot(t_eval, torch.log(loss[:-1]), label=f'log loss')
    #     plt.plot(t_eval, loss[:-1], label=f'loss')
    #     plt.axvline(x=t_eval, color='violet', label='cutoff time')

def plot_epr(trainer_list, label_list):

    for trainer, label in zip(trainer_list, label_list):
        j, t = get_j_t(trainer)

        model, data_loader = trainer.model, trainer.data_loader

        t_eval = (t[0, :, 0] + t[0, :, 1]).cpu() / 2

        # EPR
        plot_epr_var(model, j, t, t_eval, label=label)

    plot_exact_epr(data_loader, t_eval)

    trainer.model._set_loss_weight(t)
    if hasattr(trainer.model, 'loss_weight'):
        plt.plot(t_eval, trainer.model.loss_weight.cpu(), label='loss weight')


    plt.legend()
    plt.xlabel('time')
    plt.ylabel('entropy production rate')


def get_trainer_list(result_dir, model_name_list):
    trainer_list = []
    for model_name in model_name_list:
        trainer = load_trainer(result_dir, model_name)
        trainer_list.append(trainer)
    return trainer_list

def get_j_t(trainer, set='test'):
    model, data_loader = trainer.model, trainer.data_loader
    j_list = []
    t_list = []
    for minibatch in tqdm(getattr(data_loader, set)):
        x = minibatch['data'].to(model.device)  # [B, T, 2, D]
        t = minibatch['time_point'].to(model.device)  # [B, T, 2]

        j = model.forward((x, t))  # [B, T]
        j_list.append(j)
        t_list.append(t)

    j = torch.cat(j_list, dim=0)
    t = torch.cat(t_list, dim=0)
    return j, t
def plot_epr_var(model, j, t, t_eval, label='estimated epr (var)'):
    model.estimator_type = 'var'
    epr_var = model.get_entropy_production_rate(j, t).cpu()
    plt.plot(t_eval, epr_var, label=label, linewidth=1)
    return epr_var

def plot_exact_epr(data_loader, t_eval):
    if hasattr(data_loader.train.dataset, 'exact_epr'):
        exact_epr = data_loader.train.dataset.exact_epr
        plt.plot(t_eval, exact_epr, label=f'exact epr')
        return exact_epr

def plot_mi(data_loader, t_eval):
    if hasattr(data_loader.train.dataset, 'mi'):
        mutual_information = data_loader.train.dataset.mi
        plt.plot(t_eval.cpu().numpy(), mutual_information.cpu().numpy(), label=f'mutual information')

def modified_var(j, t):
    dt = (t[:, :, 1] - t[:, :, 0])[0]
    estimate = .5 * torch.mean((j[:-1] + j[1:])**2, dim=0) + torch.mean(j[:-1] * j[1:], dim=0) / dt
    return torch.cat((torch.zeros(1, device=estimate.device), estimate))

if __name__ == '__main__':

    result_dir = '/rdata/results/entropy/ppca'
    # model_name = 'mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_100000/_model_layer_dims_(512, 512)/0623-205300'
    #
    # trainer = load_trainer(result_dir, model_name, device='cuda:0')
    #
    # # plot_mi_epr(trainer, 'train')
    # plot_mi_epr(trainer, 'test')

    # model_name_list = ['mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000/_model_layer_dims_(8, 8)/0627-085213',
    #                    #'mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000/_model_layer_dims_(8, 8)/0703-140359',
    #                    'mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000/_model_layer_dims_(8, 8)_model_time_step_separation_(0.1,)/0707-145148',
    #                    # 'mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000/_model_layer_dims_(4, 4)_model_time_step_separation_(0.05,)/0706-125711',
    #                    #'mv_ou_process/_dim_3_T_10_num_steps_10000_num_samples_1000/_model_layer_dims_(2, 2)_model_time_step_separation_(0.05, 0.1)/0707-135641',
    #                    ]
    # label_list = ['regular', #'exp weight',
    #               'tss .1',
    #               # 'tss .05',
    #               #'tss .05, .1',
    #               ]

    model_name_list = ['ppca/_optimizer_lr_0.0005/_model_layer_dims_(4, 4)_model_time_step_separation_(0.007, 0.04)/0707-191131']
    label_list = ['tss .007, .04']

    trainer_list = get_trainer_list(result_dir, model_name_list)
    plot_epr(trainer_list, label_list)
    plt.xlim(-.1, 10)
    plt.title('$dt = .001$')
    plt.show()
