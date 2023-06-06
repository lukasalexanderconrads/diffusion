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
        model_config = read_yaml(os.path.join(model_dir, 'config.yaml'))

        model_config['trainer']['args']['no_training'] = True
        model_config['loader']['args']['shuffle'] = False
        model_config['loader']['args']['batch_size'] = config['args']['batch_size']

        trainer = get_trainer(model_config)
        data_loader = trainer.data_loader
        model = trainer.model

        save_dir = config['args']['path']
        num_steps = config['args'].get('num_steps', torch.inf)
        cutoff_metric = config['args'].get('cutoff_metric', 'loss')
        cutoff_var = config['args'].get('cutoff_var', 0)
        cutoff_n = config['args'].get('cutoff_n', 1)
        cutoff_value = config['args'].get('cutoff_value', torch.inf)

        # create save_dir if not exists
        model_name = os.path.join(*(model_dir.split('/')[3:]))
        Path(os.path.join(save_dir, model_name)).mkdir(parents=True, exist_ok=True)

        latent_traj = []
        lr_traj = []
        loss_traj = []
        mi_traj = []

        # checkpoint file names are in checkpoint_names.txt
        with open(os.path.join(model_dir, 'checkpoint_names.txt')) as f:
            checkpoint_names = f.readlines()

        # for each checkpoint
        for step, checkpoint_name in tqdm(enumerate(checkpoint_names), total=len(checkpoint_names)):
            if step > num_steps:
                break

            # cut off if variance of last cutoff_n values of cutoff_mmetric is less than cutoff_var
            cutoff_metric_list = mi_traj if cutoff_metric == 'mi' else loss_traj
            #var = torch.var(torch.tensor(cutoff_metric_list[-min(cutoff_n, len(mi_traj)):]))
            # if var < cutoff_var and len(cutoff_metric_list) > cutoff_n:
            #     break
            if cutoff_value < torch.inf:
                if torch.mean(torch.tensor(cutoff_metric_list[-min(cutoff_n, len(mi_traj)):])) > cutoff_value and len(cutoff_metric_list) > cutoff_n:
                    break

            # load the model from the checkpoint
            model.load_state_dict(torch.load(os.path.join(model_dir, checkpoint_name[:-1])))

            # load the data
            latent = []
            loss = []
            for i, minibatch in enumerate(data_loader.train):
                data = minibatch['data'].to(model.device).float()

                # save latent code
                latent_batch, mi, loss_batch = model.get_latent_mi_loss(data)
                latent.append(latent_batch)
                # compute loss
                loss.append(loss_batch)

            mi_traj.append(mi)

            # lr
            lr_traj.append(trainer.optimizer.param_groups[0]['lr'])


            latent = torch.cat(latent, dim=0)
            latent_traj.append(latent.detach().cpu())

            loss = torch.mean(torch.tensor(loss))
            loss_traj.append(loss)


        latent_traj = torch.stack(latent_traj, dim=1)
        lr_traj = torch.tensor(lr_traj)
        loss_traj = torch.tensor(loss_traj)

        print(f'saving latent codes of shape {latent_traj.size()} to', os.path.join(save_dir, model_name, 'latent.pt'))
        print(f'saving learning rates of shape {lr_traj.size()} to', os.path.join(save_dir, model_name, 'learn_rate.pt'))
        print(f'saving loss of shape {loss_traj.size()} to', os.path.join(save_dir, model_name, 'loss.pt'))

        torch.save(latent_traj.detach().cpu(), os.path.join(save_dir, model_name, 'latent.pt'))
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