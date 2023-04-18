import math
import os
import shutil
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import yaml

from torch.utils.tensorboard import SummaryWriter
from diffusion.utils import MetricAccumulator, get_data_loader, get_model, create_instance
from scipy.integrate import cumtrapz, simpson

class Trainer:

    def __init__(self, config, **kwargs):
        """
        :param config: dictionary from .yaml configuration file
        :param kwargs:
            n_epochs: int, number of epochs to train for
            early_stop_criterion: int, number of epochs after which there is no improvement in bm_metric, training is stopped
            bm_metric: str, name of the metric determining when the model is saved and when early stopping
            log_dir: str, path to the tensorboard logging directory
            save_dir: str, path to directory where the model .pth and config .yaml files are saved
            schedulers: list of
        """
        name = config['name']
        # data loader
        print('loading data...')
        self.data_loader = get_data_loader(config)
        # model
        print('creating model...')
        self.create_model(config)
        # optimizer
        self.optimizer = create_instance(config['optimizer']['module'], config['optimizer']['name'],
                                         config['optimizer']['args'], self.model.parameters())

        self.n_epochs = kwargs.get('n_epochs')
        assert self.n_epochs is not None, 'n_epochs not provided to trainer'

        # logging
        timestamp = self.get_timestamp()
        self.log_dir = kwargs.get('log_dir')
        assert self.log_dir is not None, 'log_dir not provided to trainer'
        log_dir = os.path.join(self.log_dir, name, timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.metric_avg = MetricAccumulator()

        # saving
        self.bm_metric = kwargs.get('bm_metric', 'loss')
        self.save_dir = kwargs.get('save_dir')
        self.save_dir = os.path.join(self.save_dir, name, timestamp)
        self.best_metric = None
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_config(config)

        self.early_stop_criterion = kwargs.get('early_stop_criterion', float('inf'))

        self.schedulers = []
        scheduler_dicts = kwargs.get('schedulers', [])
        for scheduler_dict in scheduler_dicts:
            scheduler = create_instance(scheduler_dict['module'], scheduler_dict['name'], scheduler_dict['args'])
            self.schedulers.append(scheduler)

        lr_scheduler = kwargs.get('lr_scheduler', None)
        if lr_scheduler is not None:
            self.lr_scheduler = create_instance(lr_scheduler['module'], lr_scheduler['name'], lr_scheduler['args'])

    def train(self):
        print('training parameters...')
        for epoch in tqdm(range(self.n_epochs), desc='epochs'):
            self.model.train()
            torch.set_grad_enabled(True)
            self.train_epoch(epoch)

            self.model.eval()
            torch.set_grad_enabled(False)
            self.valid_epoch(epoch)
            if self.check_early_stopping():
                break

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):
        self.update_scheduled_values(epoch)
        self.metric_avg.reset()
        for minibatch in tqdm(self.data_loader.train, desc='train set', leave=False):
            self.update_lr(epoch)
            metrics = self.model.train_step(minibatch, self.optimizer)
            self.metric_avg.update(metrics)
        metrics = self.metric_avg.get_average()
        self.writer.add_scalar('train/loss', metrics['loss'], epoch)
        self.log(metrics, epoch)

    def valid_epoch(self, epoch):
        self.metric_avg.reset()
        for minibatch in tqdm(self.data_loader.valid, desc='validation set', leave=False):
            metrics = self.model.valid_step(minibatch)
            self.metric_avg.update(metrics)
        metrics = self.metric_avg.get_average()
        self.log(metrics, epoch, 'valid')
        self.save_model(metrics)

    def log(self, metrics, epoch, split='train'):
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, epoch)

    def save_model(self, metrics):
        """
        saves the model if it is better according to bm_metric
        """
        metric = metrics[self.bm_metric]
        if self.best_metric is None or metric < self.best_metric:
            self.best_metric = metric
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
            self.early_stop_counter = 0

    @staticmethod
    def get_timestamp():
        dt_obj = datetime.now()
        timestamp = dt_obj.strftime('%m%d-%H%M%S')
        return timestamp

    def save_config(self, config):
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

    def check_early_stopping(self):
        self.early_stop_counter += 1
        if self.early_stop_counter > self.early_stop_criterion:
            self.early_stop_counter = 0
            return True
        else:
            return False

    def update_scheduled_values(self, epoch):
        for scheduler in self.schedulers:
            scheduler.update_scheduled_value(self.model, epoch)

    def update_lr(self, epoch):
        if hasattr(self, 'lr_scheduler'):
            lr = self.lr_scheduler.get_scheduled_variable_value(epoch)
            self.optimizer.param_groups[0]['lr'] = lr

    def create_model(self, config):
        self.model = get_model(config, data_shape=self.data_loader.data_shape)

class EntropyTrainer(Trainer):
    def __init__(self, config, **kwargs):
        super(EntropyTrainer, self).__init__(config, **kwargs)
        self.metric_avg.exclude_keys_from_average(['current'])

    def log(self, metrics, epoch, split='train'):
        if split == 'valid':
            current = metrics.pop('current')
            time_point = metrics.pop('time_point')
            if epoch % 50 == 0:
                self.log_entropy_production_rate_plot(current, time_point, epoch)
                self.log_current_plot(current, time_point, epoch)
                self.log_loss_plot(time_point, epoch)
        super(EntropyTrainer, self).log(metrics, epoch, split)

    def log_entropy_production_rate_plot(self, current, time_point, epoch):
        """
        adds plot of entropy production rate over time to tensorboard
        :param current: list of tensors [B, T, D]
        :param time_point: [B, T, 2]
        :return:
        """
        current = torch.concat(current, dim=0).cpu()
        dt = (time_point[:, :, 1] - time_point[:, :, 0])[0].cpu()
        t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2
        estimated_epr = (torch.max(self.model.estimators['var'](current), dim=-1).values / dt)
        # plot loss if contained in data set
        if hasattr(self.data_loader.train.dataset, 'loss'):
            loss = self.data_loader.train.dataset.loss

        # plot estimated_epr
        plt.plot(t_eval, estimated_epr, label=f'estimated epr')

        if hasattr(self.data_loader.train.dataset, 'exact_epr'):
            exact_epr = self.data_loader.train.dataset.exact_epr
            plt.plot(t_eval, exact_epr, label=f'exact epr')


        plt.legend()
        plt.xlabel('time')
        plt.ylabel('entropy production rate')
        fig = plt.gcf()
        self.writer.add_figure(tag='epr', figure=fig, global_step=epoch)

        # plot mutual information if contained in data set
        if hasattr(self.data_loader.train.dataset, 'mi'):
            mutual_information = self.data_loader.train.dataset.mi
            plt.plot(t_eval.cpu().numpy(), mutual_information[:-1].cpu().numpy(),
                     label=f'mutual information')

        # plot estimated estimated cumulative entropy production
        estimated_cum_epr = []
        for i in range(1, len(t_eval)):
            estimated_cum_epr.append(simpson(estimated_epr[:i],
                    t_eval[:i], even="avg"))
        plt.plot(t_eval[:-1], estimated_cum_epr,
                 label=f'estimated cumulative entropy production')

        # plot exact cumulative entropy production if contained in data set
        if hasattr(self.data_loader.train.dataset, 'exact_epr'):
            exact_cum_epr = []
            for i in range(1, len(t_eval)):
                exact_cum_epr.append(simpson(exact_epr[:i],
                                                        t_eval[:i], even="avg"))
            plt.plot(t_eval[:-1], exact_cum_epr,
                     label=f'exact cumulative entropy production')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('total entropy production')
        fig = plt.gcf()
        self.writer.add_figure(tag='ep', figure=fig, global_step=epoch)
        self.writer.add_scalar('valid/total_entropy_production_var', estimated_cum_epr[-1], epoch)


    def log_current_plot(self, current, time_point, epoch):
        """
        adds plot of entropy production rate over time to tensorboard
        :param current: list of tensors [B, T, D]
        :param time_point: [B, T, 2]
        :return:
        """
        current_avg = torch.mean(torch.concat(current, dim=0), dim=0).cpu() # [T, D]
        t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2

        for current_dim in range(current_avg.size(-1)):
            plt.plot(t_eval[50:], current_avg[50:, current_dim], label=f'dim {current_dim}')

        plt.legend()
        plt.xlabel('time')
        plt.ylabel('current')
        fig = plt.gcf()
        self.writer.add_figure(tag='current', figure=fig, global_step=epoch)

    def log_loss_plot(self, time_point, epoch):
        """
        adds plot of loss of the trained model over time to tensorboard
        :param time_point: [B, T, 2]
        :return:
        """
        if hasattr(self.data_loader.train.dataset, 'loss'):
            t_eval = (time_point[0, :, 0] + time_point[0, :, 1]).cpu() / 2
            loss = self.data_loader.train.dataset.loss
            plt.plot(t_eval.cpu().numpy(), torch.log(loss[:-1]).cpu().numpy(),
                     label=f'loss')
            plt.legend()
            plt.xlabel('time')
            plt.ylabel('log loss')
            fig = plt.gcf()
            self.writer.add_figure(tag='loss', figure=fig, global_step=epoch)



    def create_model(self, config):
        config['model']['args']['max_time'] = self.data_loader.max_time
        self.model = get_model(config, data_shape=self.data_loader.data_shape)