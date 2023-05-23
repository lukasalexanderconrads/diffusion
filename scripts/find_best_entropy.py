import gc
import os
import csv

import click
from pathlib import Path
import torch
from scipy.integrate import simpson


from diffusion.utils.helpers import read_yaml, get_trainer
from diffusion.utils.expand_config import expand_config


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))


def main(config_path: Path):
    configs = read_yaml(config_path)
    config_list = expand_config(configs)

    config = config_list[0]

    stopping_difference = config['stopping_difference']
    layer_multiplier = config['layer_multiplier']
    data_name = config['loader']['args']['path'].rsplit('/', 1)[-1]
    metrics_path = os.path.join(config['metrics_path'], data_name)

    done = False
    while not done:
        torch.manual_seed(config['seed'])

        print_experiment_info(config)

        trainer = get_trainer(config)
        train_metrics, valid_metrics = trainer.train(return_metrics=True)

        train_loss, valid_loss = float(train_metrics['loss'].detach()), float(valid_metrics['loss'].detach())

        if valid_loss - train_loss > stopping_difference:
            done = True

        # double layer sizes
        old_layer_dims = config['model']['args']['layer_dims']
        new_layer_dims = [dim * multiplier for dim, multiplier in zip(old_layer_dims, layer_multiplier)]
        config['name'] = config['name'].replace(f'model_layer_dims_{tuple(old_layer_dims)}',
                                                f'model_layer_dims_{tuple(new_layer_dims)}')
        config['model']['args']['layer_dims'] = new_layer_dims

        save_metrics(trainer, metrics_path, train_metrics, valid_metrics, old_layer_dims)


        # avoid memory fragmentation
        del trainer
        gc.collect()


def save_metrics(trainer, metrics_path, train_metrics, valid_metrics, layer_dims):

    file_path = os.path.join(metrics_path, 'metrics.csv')
    if not os.path.exists(file_path):
        os.mkdir(metrics_path)
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['layer_dims', 'train_loss', 'valid_loss', 'exact_ep', 'estimated_ep'])

    layer_dims = str(layer_dims).replace(',', ' ')

    train_loss = float(train_metrics['loss'])
    valid_loss = float(valid_metrics['loss'])

    current = valid_metrics['current']
    time_point = valid_metrics['time_point']
    current = torch.concat(current, dim=0).cpu()
    dt = (time_point[:, :, 1] - time_point[:, :, 0])[0].cpu()
    t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2
    estimated_epr = (torch.max(trainer.model.estimators['var'](current), dim=-1).values / dt)

    estimated_ep = float(simpson(estimated_epr, t_eval, even="avg"))

    if hasattr(trainer.data_loader.train.dataset, 'exact_epr'):
        exact_epr = trainer.data_loader.train.dataset.exact_epr
        exact_ep = float(simpson(exact_epr, t_eval, even="avg"))
    else:
        exact_ep = 'n/a'

    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([layer_dims, train_loss, valid_loss, exact_ep, estimated_ep])

def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
