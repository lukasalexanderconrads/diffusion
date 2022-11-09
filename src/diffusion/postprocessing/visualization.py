import torch
import matplotlib.pyplot as plt
import numpy as np


def get_grid_points(x_min, x_max, y_min, y_max, step_size=.05):
    x1, x2 = torch.meshgrid(torch.arange(x_min, x_max, step_size), torch.arange(y_min, y_max, step_size))
    points = torch.stack([x1.flatten(), x2.flatten()], dim=1)
    return points

def make_violin_plot(population_list):
    plt.violinplot(population_list, range(len(population_list)), points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=False)
    for i, pop in enumerate(population_list):
        plt.scatter([i]*len(pop), pop, c='red', alpha=.6)

def make_histogram_plot(model, loader):
    final_steps_list = []
    torch.set_grad_enabled(False)
    for minibatch in loader.valid:
        input = minibatch['input']
        final_steps = model.get_final_steps(input, None, None, evaluate=True)

        final_steps_list.append(final_steps)

    final_steps_list.append(torch.arange(model.max_rec, device=model.device))
    final_steps_list = torch.cat(final_steps_list)
    n_recs, hgram = torch.unique(final_steps_list, return_counts=True)
    hgram = hgram / torch.sum(hgram)
    plt.scatter(n_recs.cpu(), hgram.cpu(), c='r')
    return hgram

def plot_acc_over_cost(metric_list_list, name_list):
    color_list = ['blue', 'red', 'green', 'violet', 'orange', 'indigo', 'darkolivegreen', 'midnightblue']
    for metric_list, name, color in zip(metric_list_list, name_list, color_list):
        acc = np.concatenate([metric[0] for metric in metric_list])
        cost = np.concatenate([metric[2] for metric in metric_list])
        plt.scatter(cost, acc, c=color, label=name, alpha=.5, s=15)
    plt.legend(loc='lower right')
    plt.xlabel('cost')
    plt.ylabel('accuracy')
    plt.tight_layout()
    plt.grid(visible=True, alpha=.5)

def plot_ce_over_cost(metric_list_list, name_list):
    color_list = ['blue', 'red', 'green', 'violet', 'orange', 'indigo', 'darkolivegreen', 'midnightblue']
    for metric_list, name, color in zip(metric_list_list, name_list, color_list):
        acc = np.concatenate([metric[1] for metric in metric_list])
        cost = np.concatenate([metric[2] for metric in metric_list])
        plt.scatter(cost, acc, c=color, label=name, alpha=.5, s=15)
    plt.legend(loc='lower right')
    plt.xlabel('computational cost')
    plt.ylabel('NLL test loss')
    plt.ylim(0, .6)
    plt.tight_layout()
    plt.grid(visible=True, alpha=.5)

