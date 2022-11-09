import torch
from torch import nn
from scipy.integrate import quad, simpson

from diffusion.models.base import BaseModel
from diffusion.utils import create_mlp
from matplotlib import pyplot as plt


class NEEP(BaseModel):
    """
    Neural estimator for entropy production
    """

    def __init__(self, data_shape, **kwargs):
        super(NEEP, self).__init__(**kwargs)

        self.estimator_type = kwargs.get('estimator_type')

        #self.breathing_parabola_model = kwargs.get('breathing_parabola_model', False)

        layer_dims = kwargs.get('layer_dims')
        layer_dims = [data_shape] + layer_dims

        self.mlp = create_mlp(layer_dims).to(self.device)

        self.max_time = kwargs.get('max_time')

        self.out_dim = layer_dims[-1]
        self.t_centers = nn.Parameter(torch.arange(self.out_dim, device=self.device).unsqueeze(0).unsqueeze(1) \
                                      * self.max_time / (self.out_dim - 1))  # [1, 1, out_dim]
        self.bias = nn.Parameter(torch.ones_like(self.t_centers, device=self.device) \
                                 * self.max_time / (self.out_dim - 1))  # [1, 1, out_dim]

        # if self.breathing_parabola_model:
        #     p = (10.0, 1.0)
        #     total_ep_ = quad(get_exact_ep_rate, 0, 1, args=(p,))
        #     self.true_total_ep = total_ep_[0]
        # else:
        #     self.true_total_ep = None

        # always [B,T, D] -> [T, D]
        self.estimators = {
            "neep"    : lambda j : torch.mean(j - torch.exp(-j) + 1.0, dim=0),          # Eq. 10
            "simple"  : lambda j : 2.0 * torch.mean(j, dim=0) - 0.5 * torch.var(j, dim=0),  # Eq. 9
            "tur"     : lambda j : 2.0 * torch.mean(j, dim=0)**2 / torch.var(j, dim=0),      # Eq. 8
            "var"     : lambda j : 0.5 * torch.var(j, dim=0)                            # Eq. 14
        }

    def forward(self, input) -> torch.Tensor:
        """
        returns the neural estimator for the generalized current
        [B, T, 2, D], [B, T, 2] -> [B, T, D]
        """
        x, t = input  # [B, T, 2, D] [B, T, 2]

        x_in = torch.sum(x, dim=2) / 2                        # [B, T, D]
        t_in = torch.sum(t, dim=2) / 2                        # [B, T]
        t_in = t_in.unsqueeze(-1).repeat(1, 1, self.out_dim)  # [B, T, out_dim]

        d_vec = self.mlp(x_in)  # [B, T, out_dim]

        # equation 22
        t_gaussian = torch.exp(-torch.pow((t_in - self.t_centers) / self.bias, 2))  # [B, T, out_dim]
        d = torch.sum(d_vec * t_gaussian, dim=2)  # [B, T]

        # generalized current
        j = d.unsqueeze(-1) * (x[:, :, 1] - x[:, :, 0])  # [B, T, D]

        return j  # [B, T, D]

    def get_entropy_production_rate(self, j, t):
        """
        estimates entropy production rate given probability current j and time points t
        :param j: probability current for each trajectory, time step and dimension [B, T, D]
        :param t: time points that j was evaluated at [B, T, 2]
        :return: entropy_production_rate for each time point [T]
        """
        dt = (t[:, :, 1] - t[:, :, 0])[0]    # [T]
        entropy_production_rate = torch.max(self.estimators[self.estimator_type](j), dim=-1).values / dt     # [B, T, D] -> [T]
        return entropy_production_rate

    def loss(self, entropy_production_rate) -> dict:
        """
        calculates loss as negative average of the ep rate over all time points
        https://arxiv.org/abs/2010.03852, equation 13
        :param entropy_production_rate: estimated ep rate for each time points [T]
        :return: loss [1]
        """
        loss = -torch.mean(entropy_production_rate)
        return {'loss': loss}

    def metric(self, entropy_production_rate, t) -> dict:
        """
        calculates total entropy production as integral of ep rate
        :param entropy_production_rate: average entropy production rate at each time point [T]
        :param t: time points [B, T, 2]
        :return: dict('total_entropy_production')
        """

        midpoints = (t[0, :, 1] + t[0, :, 0]) / 2
        total_entropy_production = torch.tensor(simpson(entropy_production_rate.cpu().numpy(),
                                                            midpoints.cpu().numpy(), even="avg"))

        # if self.true_total_ep is not None:
        #     stats[f"ratio_to_true_total_ep"] = stats[f"total_entropy_production"] / self.true_total_ep

        return {'total_entropy_production': total_entropy_production}


    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        x = minibatch['data'].to(self.device)          # [B,T,2,D]
        t = minibatch['time_point'].to(self.device)     # [B,T,2]

        optimizer.zero_grad()

        j = self.forward((x, t))  # [B,T,2,D], [B,T,2] -> [B,T]

        entropy_production_rate = self.get_entropy_production_rate(j, t)

        loss_stats = self.loss(entropy_production_rate)

        loss_stats['loss'].backward()
        optimizer.step()

        return loss_stats

    def valid_step(self, minibatch: dict) -> dict:
        x = minibatch['data'].to(self.device)  # [B, T, 2, D]
        t = minibatch['time_point'].to(self.device)  # [B, T, 2]

        j = self.forward((x, t))  # [B, T]

        entropy_production_rate = self.get_entropy_production_rate(j, t)

        loss_stats = self.loss(entropy_production_rate)

        metric_stats = self.metric(entropy_production_rate, t)

        return loss_stats | metric_stats \
               | {'current': j,
                  'time_point': t}


def _build_layers(activation_fn, input_dim: int, layer_normalization: bool, layers: list, out_activation, output_dim: int) -> nn.Sequential:
    layer_sizes = [input_dim] + list(map(int, layers))
    layers = nn.Sequential()
    for i in range(len(layer_sizes) - 1):
        layers.add_module(f'layer {i}', nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if layer_normalization:
            layers.add_module(f'layer norm {i}', nn.LayerNorm(layer_sizes[i + 1]))
        layers.add_module(f'activation {i}', activation_fn())
    layers.add_module('output layer', nn.Linear(layer_sizes[-1], output_dim))
    return layers