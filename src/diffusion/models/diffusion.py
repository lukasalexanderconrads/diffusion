import torch
from torch import nn
from diffusion.models.base import BaseModel
from diffusers import UNet2DModel

class DDPM(BaseModel):
    def __init__(self, data_shape, **kwargs):
        """
        :param data_shape: tuple (n_channels, width, height)
        """
        super().__init__(**kwargs)
        self.unet = UNet2DModel(in_channels=data_shape[0])


    def forward(self, input):
        """
        :param input: batch of images to diffuse and reconstruct [batch_size, n_channels, width, height]
        :return: (forward_trajectory: first latent to approximate gaussian noise [n_steps, batch_size, n_channels, width, height],
                  backward_trajectory: first reverse latent to reconstruction [n_steps, batch_size, n_channels, width, height])
        """
        forward_trajectory = self.forward_process(input)
        backward_trajectory = self.backward_process()
        return forward_trajectory, backward_trajectory

    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer):
        original = minibatch['data']

        forward_trajectory, backward_trajectory = self(original)

        loss_stats = self.loss(original, forward_trajectory, backward_trajectory)

        optimizer.zero_grad()
        loss_stats['loss'].backward()
        optimizer.step()

        return loss_stats

    def forward_process(self, latent):
        for _ in range(self.n_steps):
            latent = self.unet(latent)

    def loss(self, original, forward_trajectory, backward_trajectory, rec_weight=1, prior_weight=1):
        """
        :param original: original image data [batch_size, n_channels, width, height]
        :param forward_trajectory: [n_steps, batch_size, n_channels, width, height]
        :param backward_trajectory: [n_steps, batch_size, n_channels, width, height]
        :return:
        """
        reconstruction = backward_trajectory[-1]
        reconstruction_loss = self.get_reconstruction_loss(original, reconstruction)

        prior_loss = self.get_prior_loss(forward_trajectory[-1])

        diffusion_loss = self.get_diffusion_loss(forward_trajectory[:-1], backward_trajectory[:-1])

        loss = diffusion_loss + rec_weight * reconstruction_loss + prior_weight * prior_loss

        return {'reconstruction_loss': reconstruction_loss,
                'prior_loss': prior_loss,
                'diffusion_loss': diffusion_loss,
                'loss': loss}






