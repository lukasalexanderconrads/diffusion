import click
from pathlib import Path
import torch

from diffusion.utils.helpers import read_yaml, get_trainer
from diffusion.utils.expand_config import expand_config


@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))


def main(config_path: Path):
    configs = read_yaml(config_path)
    config_list = expand_config(configs)
    seed = configs.get('seed', 1)
    data_seed = configs['loader'].get('seed', 1)

    num_runs = len(config_list)
    for i, config in enumerate(config_list):
        print(f'model {i} out of {num_runs}')
        config['seed'] = seed
        if config.get('change_seed', False):
            seed += 1
        config['loader']['seed'] = data_seed
        #data_seed += 1

        torch.manual_seed(config['seed'])

        print_experiment_info(config)

        trainer = get_trainer(config)
        trainer.train()


def print_experiment_info(config):
    print('-' * 10,
          '\nexperiment:', config['name'],
          '\nmodel name:', config['model']['name'],
          '\n', '-' * 10)






if __name__ == '__main__':
    main()
