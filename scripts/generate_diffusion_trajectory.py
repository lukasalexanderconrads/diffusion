import os.path

import click
from pathlib import Path
import numpy as np
import torch

from diffusion.utils.helpers import read_yaml, get_model
from diffusion.utils.expand_config import expand_config

from diffusion.models.trajectory_generator import DDPM
from diffusion.data.datasets import CIFAR10Dataset


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))

def main(config_path: Path):
    configs = read_yaml(config_path)
    config_list = expand_config(configs)
    for config in config_list:
        torch.set_grad_enabled(False)
        diff = get_model(config, data_shape=None)

        path = config['path']
        name = config['name']
        compute_metrics = config.get('compute_metrics', False)
        save_dir = os.path.join(path, name.replace('args.', ''))
        if compute_metrics:
            print('computing metrics...')
            dataset = CIFAR10Dataset(device=config['device'])
            images_true = dataset.data[:diff.ensemble_size]
            metrics = diff.metrics(images_true)
            print(f'inception score: {metrics["IS"][0]} +- {metrics["IS"][1]}')
            print(f'frechet inception distance: {metrics["FID"]}')

        diff.make_dataset(save_dir)


        print_experiment_info(config)


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\n', '-' * 10)


if __name__ == '__main__':
    main()
