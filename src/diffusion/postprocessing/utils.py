import os
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

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

def plot_epr_var(model, j, t, t_eval, label='estimated epr (var)', zorder=None):
    model.estimator_type = 'var'
    epr_var = model.get_entropy_production_rate(j, t).cpu()
    plt.plot(t_eval, epr_var, label=label, linewidth=1, zorder=zorder)
    return epr_var

def plot_mi(data_loader, t_eval):
    if hasattr(data_loader.train.dataset, 'mi'):
        mutual_information = data_loader.train.dataset.mi
        plt.plot(t_eval.cpu().numpy(), mutual_information.cpu().numpy(), label=f'mutual information')