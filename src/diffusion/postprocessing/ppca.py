import torch
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import math


from diffusion.postprocessing.utils import *

def plot_mi_epr(trainer, set='test'):

    j, t = get_j_t(trainer, set)
    print('current sum ensemble:', torch.mean(torch.sum(j, dim=1)))
    model, data_loader = trainer.model, trainer.data_loader

    t_eval = (t[0, :, 0] + t[0, :, 1]).cpu() / 2

    # EPR
    epr_var = plot_epr_var(model, j, t, t_eval)

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

def get_system_entropy(trainer):
    model = trainer.model

    sigma_x = model.ground_truth_cov

    W = model.linear_map.cpu()
    sigma_sq = torch.abs(model.sigma_square).cpu()

    M = W.T @ W + sigma_sq * torch.eye(model.latent_dim)
    M_inv = torch.inverse(M)

    cov_z = M_inv @ W.T @ sigma_x @ W @ M_inv.T + sigma_sq * M_inv

    entropy = .5 * cov_z.size(0) * (1 + math.log(2 * math.pi)) + .5 * torch.log(torch.det(cov_z))
    return entropy

def print_system_entropy_change(result_dir, model_name, last_epoch=None):
    trainer = load_trainer(result_dir, model_name, model_version='checkpoint-0-0.pth')

    initial_entropy = get_system_entropy(trainer)

    trainer = load_trainer(result_dir, model_name, model_version=f'checkpoint-{last_epoch}-0.pth' if \
        last_epoch is not None else 'best_model.pth')

    final_entropy = get_system_entropy(trainer)

    print('initial entropy:', initial_entropy)
    print('final entropy:', final_entropy)
    print('entropy change:', final_entropy - initial_entropy)





if __name__ == '__main__':

    # result_dir = '/raid/results/ppca_3'
    # model_name = 'no_clip/_optimizer_lr_0.001/1027-131459'
    #
    # print_system_entropy_change(result_dir, model_name)


    result_dir = '/rdata/results/entropy/ppca'
    model_name = 'mean/_optimizer_lr_0.005/mean/_model_layer_dims_(16, 16)/1109-230144'

    trainer = load_trainer(result_dir, model_name)
    plot_mi_epr(trainer)

