import os.path

import click
from pathlib import Path
import numpy as np

from diffusion.utils import read_yaml
from diffusion.expand_config import expand_config

from diffusion.models.trajectory_generator import DDPM


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))

def main(config_path: Path):
    configs = read_yaml(config_path)
    config_list = expand_config(configs)
    for config in config_list:
        seed = config.get('seed', 1)
        device = config.get('device', 'cpu')
        name = config['name']

        kwargs = config['args']

        num_steps = kwargs['num_steps']
        num_samples = kwargs['num_samples']
        batch_size = kwargs['batch_size']

        diff = DDPM(device)

        path = kwargs['path']
        save_dir = os.path.join(path, name.replace('args.', ''))
        diff.make_dataset(save_dir, ensemble_size=num_samples, batch_size=batch_size)


        print_experiment_info(config)


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\n', '-' * 10)


if __name__ == '__main__':
    main()
