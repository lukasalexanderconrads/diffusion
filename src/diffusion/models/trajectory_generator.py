import os
import pickle

import torch
import numpy as np
from scipy.integrate import simpson
from tqdm import tqdm

from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

from diffusion.utils import *


class MultivariateOUProcess:
    def __init__(self, mean_0, var_0, A, B, T, num_steps=1000, diagonal=False):
        assert A.shape == B.shape == (*mean_0.shape, *mean_0.shape) == var_0.shape
        self.rng = np.random.default_rng()
        self.dt = T / num_steps

        self.T = T
        self.num_steps = num_steps
        self.dim = A.shape[0]

        self.time = np.linspace(0, self.T, self.num_steps, endpoint=True)
        self.diagonal = diagonal


        if self.dim > 1:
            self.mean_0 = mean_0
            self.var_0 = var_0
            self.A = A
            self.B = B
            # if self.diagonal and False:
            #     self.mean = mean_0[None, :] * np.exp(-2 * np.outer(self.time, np.diag(A)))
            #     diags = np.exp(-2 * np.outer(self.time, np.diag(A)))  # [t, d]
            #     term_exp = np.apply_along_axis(np.diag, axis=1, arr=diags)  # [t, d, d]
            #     A_inv = np.diag(1 / np.diag(A))
            #     self.var = (var_0 - B**2 / 2 * A_inv) * term_exp + B**2 / 2 * A_inv
            #
            # else:
            # Eigenvalue Decomposition
            lamda, S = np.linalg.eig(A)  # [d], [d, d]
            S_ct = S.copy()
            S = S.conjugate().T

            # exp(-At) @ mu_0
            diags = np.exp(-np.outer(self.time, lamda))  # [t, d]
            term_exp = np.apply_along_axis(np.diag, axis=1, arr=diags)  # [t, d, d]

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
            term_exp = np.apply_along_axis(np.diag, axis=1, arr=diags)  # [t, d, d]
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

            if self.diagonal or True:
                ### THIS WORKS
                epr = 0
                for i in range(self.dim):
                    d = B[i, i] ** 2 / 2
                    a = A[i, i]
                    mean = self.mean[:, i]
                    var = self.var[:, i, i]
                    epr += a**2 / d * (mean**2 + var) + d / var - 2 * a
                return epr
            else:
                D = B @ B.T / 2
                inv_D = np.linalg.inv(D)
                inv_var = np.linalg.inv(var)
                ADA = A.T @ inv_D @ A
                term1 = np.einsum('ti,ti->t', mean @ ADA, mean)
                epr = term1 + np.trace(ADA[None, :, :] @ var, axis1=1, axis2=2) \
                      - 2 * np.trace(A) + np.trace(ADA[None, :, :] @ inv_var, axis1=1, axis2=2)
        else:
            D = B**2 / 2
            epr = A ** 2 / D * (mean ** 2 + var) + D / var - 2 * A
            epr = epr.squeeze()

        return epr

    def get_cum_ep(self):
        epr = self.get_exact_epr()
        cumulative_epr = [simpson(epr[:t], dx=self.dt) for t in range(1, len(epr))]
        return cumulative_epr

    def make_dataset(self, save_dir, ensemble_size=10000):
        #  create dir if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        trajectory_list = []
        for _ in tqdm(range(ensemble_size)):
            trajectory = self.get_trajectory()
            trajectory_list.append(trajectory)

        trajectories = np.stack(trajectory_list, axis=0)
        time_points = np.expand_dims(self.time, 0).repeat(ensemble_size, axis=0)
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




class DDPM():
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, num_steps, device='cpu'):
        self.device = device
        pipeline = DDPMPipeline.from_pretrained('google/ddpm-cifar10-32')
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.time_points = torch.arange(0, len(self.scheduler))

    def forward(self, clean_images):
        """
        :param clean_images: batch of clean images of shape [batch_size, 3, 32, 32]
        :return:
        """
        print(clean_images.size())
        batch_size = clean_images.size(0)
        noise = torch.randn((batch_size, len(self.scheduler),  3, 32, 32)).to(self.device)
        time_points = self.time_points.unsqueeze(0).repeat(batch_size, 1)
        noisy_images = self.scheduler.add_noise(clean_images, noise, time_points)
        print(noisy_images.size())



