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
    config = read_yaml(config_path)
    seed = config.get('seed', 1)

    rng = np.random.default_rng(seed=seed)

    dim = config['dim']

    mean_0 = np.ones(dim)
    var_0 = np.eye(dim)

    diagonal = config.get('diagonal', False)
    if diagonal or True:
        A = get_random_diagonal_matrix(dim, rng=rng)
        B = get_random_diagonal_matrix(dim, allow_singular=False, rng=rng)
    else:
        A = get_random_hermitian(dim, rng=rng)
        B = get_random_hermitian(dim, allow_singular=False, rng=rng)

    T = config['T']
    num_steps = config['num_steps']
    num_samples = config['num_samples']

    ou = MultivariateOUProcess(mean_0, var_0, A, B, T, num_steps=num_steps, diagonal=diagonal)

    path = config['path']
    ou.make_dataset(path, ensemble_size=num_samples)


    print_experiment_info(config)


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\n', '-' * 10)


if __name__ == '__main__':
    main()
