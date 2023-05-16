import os
import pickle
from abc import ABC, abstractmethod
import math

import torch
import torch as t
import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm

from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline, ScoreSdeVeScheduler, UNet2DModel

from diffusion.utils.helpers import *
from diffusion.utils.metrics import inception_score, fid_score


class MultivariateOUProcess:
    def __init__(self, mean_0, var_0, A, B, T, num_steps=1000, diagonal=False, device='cpu'):
        self.device = torch.device(device)
        assert A.shape == B.shape == (*mean_0.shape, *mean_0.shape) == var_0.shape
        self.rng = np.random.default_rng()
        self.dt = T / num_steps

        self.T = T
        self.num_steps = num_steps
        self.dim = A.shape[0]

        self.time = np.linspace(0, self.T, self.num_steps, endpoint=True)
        self.diagonal = diagonal

        print('computing mean and variance...')
        if self.dim > 1:
            self.mean_0 = mean_0
            self.var_0 = var_0
            self.A = A
            self.B = B

            lamda, S = np.linalg.eig(A)  # [d], [d, d]
            S_ct = S.copy()
            S = S.conjugate().T

            # exp(-At) @ mu_0
            diags = np.exp(-np.outer(self.time, lamda))  # [t, d]
            term_exp = batched_diag(diags)  # [t, d, d]

            term1 = np.einsum('ij,tjk->tik', S_ct, term_exp)  # [t, d, d]
            term2 = np.einsum('tij,jk->tik', term1, S)
            self.mean = np.einsum('tij,j->ti', term2, self.mean_0)

            # pairwise eigenvalue sums
            lamda_sums = np.add.outer(lamda, lamda)

            # matrix G
            SBBS = S @ B @ B.T @ S_ct
            G = SBBS / (lamda_sums) * (1 - np.exp(-np.einsum('ij,t->tij', lamda_sums, self.time)))
            #G = (B @ B.T) / (lamda_sums) * (1 - np.exp(-np.einsum('ij,t->tij', lamda_sums, self.time)))

            # variance
            diags = np.exp(-2 * np.outer(self.time, lamda))  # [t, d]
            term_exp = batched_diag(diags)  # [t, d, d]
            term1 = term_exp + G  # [t, d, d]
            term2 = np.einsum('ij,tjk->tik', S_ct, term1)  # [t, d, d]
            self.var = np.einsum('tij,jk->tik', term2, S)
        else:
            self.mean_0 = float(mean_0)
            self.var_0 = float(var_0)
            self.A = float(A)
            self.B = float(B)
            D = B**2 / 2
            self.mean = self.mean_0 * np.exp(-A * self.time)
            self.var = (self.var_0 - D / A) * np.exp(-2 * A * self.time) + D / A

    def get_trajectory_batch(self, batch_size=100):
        x0 = self.rng.multivariate_normal(mean=self.mean_0,
                                              cov=self.var_0,
                                              size=batch_size)
        x0 = t.from_numpy(x0).to(self.device)

        normal_samples = t.randn((self.num_steps-1, batch_size, self.dim), device=self.device)
        A = t.from_numpy(self.A).unsqueeze(0).expand(batch_size, -1, -1).to(self.device).float()
        B = t.from_numpy(self.B).unsqueeze(0).expand(batch_size, -1, -1).to(self.device).float()
        trajectory = [x0]
        xt = x0.clone().unsqueeze(-1).float()

        for i in range(normal_samples.size(0)):
            shift = -t.bmm(A, xt) * self.dt
            noise = t.bmm(B, normal_samples[i].unsqueeze(-1)) * np.sqrt(self.dt)
            xt += shift + noise
            trajectory.append(xt.clone().squeeze())

        return torch.stack(trajectory, dim=0)


    def get_trajectory(self):
        if self.dim > 1:
            x0 = self.rng.multivariate_normal(mean=self.mean_0,
                                              cov=self.var_0,
                                              size=1).T.squeeze()

            normal_samples = self.rng.multivariate_normal(mean=np.zeros(self.dim),
                                                          cov=np.eye(self.dim),
                                                          size=self.num_steps - 1)

            trajectory = [x0]
            xt = x0.copy()
            for i in range(len(normal_samples)):
                xt += -self.A @ xt * self.dt + self.B @ normal_samples[i] * np.sqrt(self.dt)
                trajectory.append(xt.copy())

            return trajectory
        else:
            # returns x0, ... xT
            x0 = self.rng.normal(loc=self.mean_0,
                                scale=np.sqrt(self.var_0),
                                size=1)
            normal_samples = self.rng.normal(loc=0,
                                scale=self.B * np.sqrt(self.dt),
                                size=self.num_steps - 1)

            # x0 sampled from initial condition
            # adjust variance of the normal samples

            trajectory = [x0]
            xt = x0.copy()
            for i in range(len(normal_samples)):
                xt += -self.A * xt * self.dt + normal_samples[i]
                trajectory.append(xt.copy())

            return np.array(trajectory)

    def get_exact_epr(self):
        A = self.A
        B = self.B
        mean = self.mean
        var = self.var
        if self.dim > 1:
            D = B @ B.T / 2
            inv_D = np.linalg.inv(D)
            inv_var = np.linalg.inv(var)
            ADA = (A.T @ inv_D @ A)[None, :, :]
            D = D[None, :, :]
            mean_T = mean[:, None, :]
            mean = mean[:, :, None]
            epr = np.squeeze(mean_T @ ADA @ mean) + np.trace(ADA @ var + inv_var @ D, axis1=1, axis2=2) - 2 * np.trace(A)

        else:
            D = B**2 / 2
            epr = A ** 2 / D * (mean ** 2 + var) + D / var - 2 * A
            epr = epr.squeeze()

        return epr

    def get_cum_ep(self):
        epr = self.get_exact_epr()
        cumulative_epr = [simpson(epr[:t], dx=self.dt) for t in range(1, len(epr))]
        return cumulative_epr

    def make_dataset(self, save_dir, ensemble_size=10000, batch_size=1000):
        #  create dir if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trajectory_list = []
        print('sampling trajectories...')
        if self.dim > 1:
            for _ in tqdm(range(ensemble_size // batch_size)):
                trajectory = self.get_trajectory_batch(batch_size)
                trajectory_list.append(trajectory.cpu())
            trajectories = torch.cat(trajectory_list, dim=1)
            trajectories = t.transpose(trajectories, 0, 1).cpu().numpy()


        else:
            for _ in tqdm(range(ensemble_size)):
                trajectory = self.get_trajectory()
                trajectory_list.append(trajectory)

            trajectories = np.stack(trajectory_list, axis=0)
        time_points = np.expand_dims(self.time, 0).repeat(ensemble_size, axis=0)
        print('computing exact epr...')
        exact_epr = self.get_exact_epr()

        np.save(os.path.join(save_dir, 'trajectories.npy'), trajectories)
        np.save(os.path.join(save_dir, 'time_points.npy'), time_points)
        np.save(os.path.join(save_dir, 'exact_epr.npy'), exact_epr)
        print('saved trajecories of shape', trajectories.shape)
        print('saved time points of shape', time_points.shape)
        print('saved exact epr of shape', exact_epr.shape)

        with open(os.path.join(save_dir, 'meta.pkl'), 'wb') as file:
            pickle.dump({'mean_0': self.mean_0,
                         'var_0': self.var_0,
                         'A': self.A,
                         'B': self.B,
                         'T': self.T}, file)



class DiffusionModel(ABC):
    def __init__(self, data_shape, **kwargs):
        self.data_shape = data_shape
        self.device = torch.device(kwargs.get('device', 'cpu'))
        self.batch_size = kwargs.get('batch_size', 100)
        self.num_steps = kwargs.get('num_steps', 1000)
        self.ensemble_size = kwargs.get('num_samples', 1000)

    @abstractmethod
    def forward(self, rng=None):
        pass

    def make_dataset(self, save_dir):
        #  create dir if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        rng = torch.Generator(self.device)
        print('sampling trajectories...')
        trajectory_list = []
        for _ in tqdm(range(self.ensemble_size // self.batch_size)):
            trajectory = self.forward(rng=rng)
            trajectory_list.append(trajectory.cpu())
        trajectories = torch.cat(trajectory_list, dim=0)
        trajectories = trajectories.flatten(start_dim=2).numpy()

        time_points = np.expand_dims(np.linspace(0, 1, self.num_steps), 0).repeat(self.ensemble_size, axis=0)

        np.save(os.path.join(save_dir, 'trajectories.npy'), trajectories)
        np.save(os.path.join(save_dir, 'time_points.npy'), time_points)
        print('saved trajecories of shape', trajectories.shape)
        print('saved time points of shape', time_points.shape)


    def metrics(self, images_true):
        rng = torch.Generator(self.device)
        num_images = images_true.size(0)
        num_batches = math.ceil(images_true.size(0) / self.batch_size)

        images_pred_list = []
        for _ in tqdm(range(num_batches)):
            trajectory = self.forward(rng=rng)
            images_pred_list.append(trajectory[:, -1])
        images_pred = torch.cat(images_pred_list, dim=0)[:num_images]
        images_pred = (images_pred / 2 + 0.5).clamp(0, 1)

        inception_score_val = inception_score(images_pred, batch_size=self.batch_size, device=self.device)
        fid_score_val = fid_score(images_pred, images_true, batch_size=self.batch_size, device=self.device)

        return {'IS': inception_score_val, 'FID': fid_score_val}

class DDPM(DiffusionModel):

    def __init__(self, data_shape, **kwargs):
        super().__init__(data_shape, **kwargs)
        pipeline = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
        self.unet = pipeline.unet.to(self.device)
        self.scheduler = pipeline.scheduler

    def forward(self, rng=None):
        """
        :param clean_images: batch of clean images of shape [batch_size, 3, 32, 32]
        :return:
        """
        rng = torch.Generator(self.device) if rng is None else rng
        self.scheduler.set_timesteps(num_inference_steps=self.num_steps, device=self.device)
        image_shape = (self.batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = torch.randn(image_shape, generator=rng, device=self.device)

        image_list = []
        for t in tqdm(self.scheduler.timesteps, leave=False):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=rng).prev_sample

            image_list.append(image.clone())

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        return torch.stack(image_list, dim=1)

class DDIM(DiffusionModel):

    def __init__(self, data_shape, **kwargs):
        super().__init__(data_shape, **kwargs)
        pipeline = DDIMPipeline.from_pretrained('google/ddpm-cifar10-32')
        self.unet = pipeline.unet.to(self.device)
        self.scheduler = pipeline.scheduler

    def forward(self, rng=None):
        """
        :param clean_images: batch of clean images of shape [batch_size, 3, 32, 32]
        :return:
        """
        rng = torch.Generator(self.device) if rng is None else rng
        self.scheduler.set_timesteps(num_inference_steps=self.num_steps)
        image_shape = (self.batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = torch.randn(image_shape, generator=rng, device=self.device)

        image_list = []
        for t in tqdm(self.scheduler.timesteps, leave=False):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=rng).prev_sample

            image_list.append(image)

        # image = (image / 2 + 0.5).clamp(0, 1)
        # image = image.cpu().permute(0, 2, 3, 1).numpy()
        return torch.stack(image_list, dim=1)


class ScoreSDE(DiffusionModel):

    def __init__(self, data_shape, **kwargs):
        super().__init__(data_shape, **kwargs)
        #pipeline = ScoreSdeVePipeline.from_pretrained('fusing/cifar10-ncsnpp-ve')
        self.unet = UNet2DModel.from_pretrained('google/ddpm-cifar10-32').to(self.device)
        self.scheduler = ScoreSdeVeScheduler.from_pretrained('google/ncsnpp-church-256')
    def forward(self, rng=None):
        rng = torch.Generator(self.device) if rng is None else rng
        self.scheduler.set_timesteps(num_inference_steps=self.num_steps)
        image_shape = (self.batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = torch.randn(image_shape, generator=rng, device=self.device)

        self.scheduler.set_timesteps(self.num_steps)
        self.scheduler.set_sigmas(self.num_steps)

        image_list = []
        for i, t in tqdm(enumerate(self.scheduler.timesteps), leave=False):
            sigma_t = self.scheduler.sigmas[i] * torch.ones(self.batch_size, device=self.device)

            # correction step
            for _ in range(self.scheduler.config.correct_steps):
                model_output = self.unet(image, sigma_t).sample
                image = self.scheduler.step_correct(model_output, image, generator=rng).prev_sample

            # prediction step
            model_output = self.unet(image, sigma_t).sample
            output = self.scheduler.step_pred(model_output, t, image, generator=rng)

            image = output.prev_sample
            image_list.append(image)

        return torch.stack(image_list, dim=1)
