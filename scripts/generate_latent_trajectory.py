import os
from pathlib import Path

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

        model_config = read_yaml(os.path.join(model_dir, next(os.scandir(model_dir)), 'config.yaml'))

        batch_size = config['args']['batch_size']
        time_point_batch_size = config['args']['time_point_batch_size']
        model_config['trainer']['args']['no_training'] = True
        model_config['loader']['args']['shuffle'] = False
        model_config['loader']['args']['batch_size'] = batch_size
        model_config['device'] = config['device']

        trainer = get_trainer(model_config)
        data_loader = trainer.data_loader
        model = trainer.model

        save_dir = config['args']['path']
        max_n_time_points = config['args'].get('max_n_time_points', torch.inf)

        n_trajectories_per_data = config['args'].get('n_trajectories_per_data', 1)
        n_trajectories_per_model = config['args'].get('n_trajectories_per_model', None)
        if n_trajectories_per_model is None:
            n_trajectories_per_model = len(data_loader.train.dataset)
        cov_factor = config['args'].get('cov_factor', 1)

        # create save_dir if not exists
        model_name = os.path.join(*(model_dir.split('/')[3:]))
        Path(os.path.join(save_dir, model_name)).mkdir(parents=True, exist_ok=True)

        timestamp_list = [_ for _ in os.scandir(model_dir)]
        n_timestamps = len(timestamp_list)

        ### set up memory map ###
        # get trajectory length
        with open(os.path.join(model_dir, timestamp_list[0], 'checkpoint_names.txt')) as f:
            checkpoint_names = f.readlines()

        n_time_points = min(len(checkpoint_names), max_n_time_points)
        assert n_time_points % time_point_batch_size == 0, \
            f'time point batch size {time_point_batch_size} not divisible by number of time points {n_time_points}'
        n_trajects = n_timestamps * n_trajectories_per_model * n_trajectories_per_data
        data_dim = data_loader.train.dataset.latent_shape
        traject_shape = (n_trajects, n_time_points, data_dim)

        traject_file = os.path.join(save_dir, model_name, 'latent.dat')
        file_map = np.memmap(traject_file, dtype='float32', mode='w+', shape=traject_shape)

        # FOR EACH MODEL
        for model_number, timestamp in tqdm(enumerate(timestamp_list), total=n_timestamps):

            lr_traj = []
            loss_traj = []
            mi_traj = []

            # checkpoint file names are in checkpoint_names.txt
            with open(os.path.join(model_dir, timestamp, 'checkpoint_names.txt')) as f:
                checkpoint_names = f.readlines()

            # FOR EACH TIME POINT
            latent_time_step_batch = []
            for step, checkpoint_name in tqdm(enumerate(checkpoint_names), total=len(checkpoint_names), leave=False):
                if step >= max_n_time_points:
                    break

                # load the model from the checkpoint
                model.load_state_dict(torch.load(os.path.join(model_dir, timestamp, checkpoint_name[:-1])))

                # load the data
                latent = []
                loss = []
                # FOR EACH MINIBATCH
                for batch_number, minibatch in enumerate(data_loader.train):
                    data = minibatch['data'].to(model.device).float()

                    data = data.repeat_interleave(n_trajectories_per_data, dim=0)

                    # save latent code
                    latent_batch, mi, loss_batch = model.get_latent_mi_loss(data, cov_factor=cov_factor)
                    latent.append(latent_batch.detach())
                    # compute loss
                    loss.append(loss_batch)

                    # if (batch_number + 1) * latent_batch.size(0) * n_trajectories_per_data >= n_trajectories_per_model:
                    #     print()
                    #     break

                latent_time_step_batch.append(torch.cat(latent, dim=0))

                if len(latent_time_step_batch) == time_point_batch_size:
                    print(n_trajectories_per_model, n_trajectories_per_data)
                    batch_start_idx = model_number * n_trajectories_per_model * n_trajectories_per_data
                    batch_end_idx = (model_number + 1) * n_trajectories_per_model * n_trajectories_per_data
                    time_step_start_idx = step - time_point_batch_size + 1
                    time_step_end_idx = step + 1

                    latent_time_step_batch = torch.stack(latent_time_step_batch, dim=1)
                    file_map[batch_start_idx:batch_end_idx, time_step_start_idx:time_step_end_idx] = latent_time_step_batch.cpu().numpy().copy()

                    latent_time_step_batch = []

                mi_traj.append(mi)

                # lr
                lr_traj.append(trainer.optimizer.param_groups[0]['lr'])
                # loss
                loss = torch.mean(torch.tensor(loss))
                loss_traj.append(loss)



            lr_traj = torch.tensor(lr_traj)
            loss_traj = torch.tensor(loss_traj)



        print(f'saving latent codes of shape {traject_shape} to', os.path.join(save_dir, model_name, 'latent.dat'))
        print(f'saving learning rates of shape {lr_traj.size()} to', os.path.join(save_dir, model_name, 'learn_rate.pt'))
        print(f'saving loss of shape {loss_traj.size()} to', os.path.join(save_dir, model_name, 'loss.pt'))

        # close latent file map
        file_map.flush()

        np.save(os.path.join(save_dir, model_name, 'data_shape.npy'), np.array(traject_shape))
        torch.save(lr_traj.detach().cpu(), os.path.join(save_dir, model_name, 'learn_rate.pt'))
        torch.save(loss_traj.detach().cpu(), os.path.join(save_dir, model_name, 'loss.pt'))

        if mi_traj[0] is not None:
            mi_traj = torch.tensor(mi_traj)
            print(f'saving mi of shape {mi_traj.size()} to', os.path.join(save_dir, model_name, 'mi.pt'))
            torch.save(mi_traj.detach().cpu(), os.path.join(save_dir, model_name, 'mi.pt'))



        # if isinstance(model.model, CatVAE):
        #     print(f'saving mi of length {len(mi_traj)} to', os.path.join(save_dir, model_name, 'mi.pt'))
        #     torch.save(torch.tensor(mi_traj).detach().cpu(), os.path.join(save_dir, model_name, 'mi.pt'))


if __name__ == '__main__':
    main()