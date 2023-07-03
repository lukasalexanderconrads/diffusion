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
        trainer.train()
        train_metrics, valid_metrics = trainer.evaluate_best_model()

        train_loss, valid_loss = float(train_metrics['loss'].detach()), float(valid_metrics['loss'].detach())

        if valid_loss - train_loss > stopping_difference:
            done = True

        # double layer sizes
        old_layer_dims = config['model']['args']['layer_dims']
        new_layer_dims = [int(dim * multiplier) for dim, multiplier in zip(old_layer_dims, layer_multiplier)]
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
        os.makedirs(metrics_path)
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['layer_dims', 'train loss', 'valid loss', 'exact_ep', 'train ep var', 'valid ep var', 'valid ep simple'])

    layer_dims = str(layer_dims).replace(',', ' ')

    train_loss = float(train_metrics['loss'])
    valid_loss = float(valid_metrics['loss'])

    valid_current = valid_metrics['current']
    train_current = train_metrics['current']
    time_point = valid_metrics['time_point']
    valid_current = torch.concat(valid_current, dim=0)
    train_current = torch.concat(train_current, dim=0)
    t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2

    trainer.model.estimator_type = 'var'
    valid_epr_var = trainer.model.get_entropy_production_rate(valid_current, time_point).cpu()
    valid_ep_var = float(simpson(valid_epr_var, t_eval, even="avg"))
    train_epr_var = trainer.model.get_entropy_production_rate(train_current, time_point).cpu()
    train_ep_var = float(simpson(train_epr_var, t_eval, even="avg"))

    trainer.model.estimator_type = 'simple'
    epr_simple = trainer.model.get_entropy_production_rate(valid_current, time_point).cpu()
    ep_simple = float(simpson(epr_simple, t_eval, even="avg"))


    if hasattr(trainer.data_loader.train.dataset, 'exact_epr'):
        exact_epr = trainer.data_loader.train.dataset.exact_epr
        exact_ep = float(simpson(exact_epr, t_eval, even="avg"))
    else:
        exact_ep = 'n/a'

    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([layer_dims, train_loss, valid_loss, exact_ep, train_ep_var, valid_ep_var, ep_simple])

def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
