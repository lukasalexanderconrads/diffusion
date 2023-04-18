import os.path

import click
from pathlib import Path
import numpy as np

from diffusion.utils import read_yaml, get_trainer
from diffusion.expand_config import expand_config

from diffusion.models.trajectory_generator import MultivariateOUProcess
from diffusion.utils import get_random_hermitian, get_random_diagonal_matrix


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
        dim = kwargs.get('dim', 1)

        rng = np.random.default_rng(seed=seed)

        mean_0 = np.ones(dim)
        var_0 = np.eye(dim)

        diagonal = kwargs.get('diagonal', False)
        unitary = kwargs.get('unitary', False)
        if diagonal:
            A = get_random_diagonal_matrix(dim, rng=rng)
            B = get_random_diagonal_matrix(dim, allow_singular=False, rng=rng)
        else:
            A = get_random_hermitian(dim, rng=rng)
            B = get_random_hermitian(dim, allow_singular=False, unitary=unitary, rng=rng)

        T = kwargs['T']
        num_steps = kwargs['num_steps']
        num_samples = kwargs['num_samples']
        batch_size = kwargs['batch_size']

        ou = MultivariateOUProcess(mean_0, var_0, A, B, T, num_steps=num_steps, diagonal=diagonal, device=device)

        path = kwargs['path']
        save_dir = os.path.join(path, name.replace('args.', ''))
        ou.make_dataset(save_dir, ensemble_size=num_samples, batch_size=batch_size)


        print_experiment_info(config)


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\n', '-' * 10)


if __name__ == '__main__':
    main()
