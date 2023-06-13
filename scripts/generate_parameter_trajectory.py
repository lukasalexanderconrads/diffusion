import os
from pathlib import Path
import math
import csv

from tqdm import tqdm
import click
import torch
import numpy as np

from diffusion.utils.expand_config import expand_config
from diffusion.utils.helpers import read_yaml, get_trainer

@click.command()
@click.option('-c', '--config', 'config_path', required=True, type=click.Path(exists=True))

def main(config_path: Path):

    configs = read_yaml(config_path)
    config_list = expand_config(configs)
    for config in config_list:
        model_dir = config['args']['model_dir']

        save_dir = config['args']['path']
        num_steps = config['args'].get('num_steps', torch.inf)

        # create save_dir if not exists
        model_name = os.path.join(*(model_dir.split('/')[3:]))
        Path(os.path.join(save_dir, model_name)).mkdir(parents=True, exist_ok=True)

        param_traj_list = []

        timestamp_list = os.scandir(model_dir)
        for timestamp in tqdm(timestamp_list):
            # parameters
            file_path = os.path.join(model_dir, timestamp, 'parameter_trajectory.csv')
            parameter_traj = torch.from_numpy(np.genfromtxt(file_path, delimiter=',', dtype=float))

            if parameter_traj.dim() == 1:
                parameter_traj = parameter_traj.unsqueeze(-1)
            elif parameter_traj.dim() == 0:
                continue
            if len(param_traj_list) > 0:
                if parameter_traj.size(0) != param_traj_list[-1].size(0):
                    continue
            param_traj_list.append(parameter_traj)

        param_traj = torch.stack(param_traj_list, dim=0)

        # learning rate
        file_path = os.path.join(model_dir, timestamp, 'lr_trajectory.csv')
        lr_traj = torch.from_numpy(np.genfromtxt(file_path, delimiter=',', dtype=float))
        lr_traj = torch.ones_like(lr_traj)

        print(f'saving parameters of shape {param_traj.size()} to', os.path.join(save_dir, model_name, 'latent.pt'))
        print(f'saving learning rates of shape {lr_traj.size()} to', os.path.join(save_dir, model_name, 'learn_rate.pt'))
        # print(f'saving loss of shape {loss_traj.size()} to', os.path.join(save_dir, model_name, 'loss.pt'))

        torch.save(param_traj, os.path.join(save_dir, model_name, 'latent.pt'))
        torch.save(lr_traj, os.path.join(save_dir, model_name, 'learn_rate.pt'))
        # torch.save(loss_traj.detach().cpu(), os.path.join(save_dir, model_name, 'loss.pt'))

        # if mi_traj[0] is not None:
        #     mi_traj = torch.tensor(mi_traj)
        #     print(f'saving mi of shape {mi_traj.size()} to', os.path.join(save_dir, model_name, 'mi.pt'))
        #     torch.save(mi_traj.detach().cpu(), os.path.join(save_dir, model_name, 'mi.pt'))

if __name__ == '__main__':
    main()