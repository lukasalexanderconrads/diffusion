import torch
from torch import nn
from scipy.integrate import quad, simpson

from diffusion.models.base import BaseModel
from diffusion.utils.helpers import create_mlp


class NEEP(BaseModel):
    """
    Neural estimator for entropy production
    """

    def __init__(self, data_dim, **kwargs):
        super(NEEP, self).__init__(**kwargs)

        self.estimator_type = kwargs.get('estimator_type')

        layer_dims = kwargs.get('layer_dims').copy()


        self.max_time = kwargs.get('max_time')

        self.kernel = kwargs.get('kernel', 'gaussian')
        self.data_dim = data_dim
        if self.kernel == 'gaussian':
            self.out_dim = layer_dims[-1]
            layer_dims[-1] *= data_dim
            layer_dims = [data_dim] + layer_dims

            self.t_centers = nn.Parameter(torch.arange(self.out_dim, device=self.device).unsqueeze(0).unsqueeze(1) \
                                          * self.max_time / (self.out_dim - 1))  # [1, 1, out_dim]
            self.bias = nn.Parameter(torch.ones_like(self.t_centers, device=self.device) \
                                 * self.max_time / (self.out_dim - 1))  # [1, 1, out_dim]
        elif self.kernel == 'exponential':
            self.out_dim = 4
            layer_dims = [data_dim] + layer_dims + [4 * data_dim]
        elif self.kernel == 'global_exponential':
            self.out_dim = layer_dims[-1]
            layer_dims[-1] *= data_dim
            layer_dims = [data_dim] + layer_dims
            self.exp_params = nn.Parameter(torch.randn((4, data_dim), device=self.device))

        self.dropout = kwargs.get('dropout', .0)
        self.mlp = create_mlp(layer_dims, dropout=self.dropout).to(self.device)

        print(self.mlp)

        # exponential curve that is weight for loss: can be None or [a, b] where weight(t) = a * exp(-t * b) + 1
        self.loss_weight_exponential_parameters = kwargs.get('loss_weight_exponential_parameters', None)

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

        d_vec = self.mlp(x_in)  # [B, T, out_dim * D]
        d_vec = d_vec.view(x.size(0), x.size(1), self.out_dim, self.data_dim) # [B, T, out_dim, D]

        if self.kernel == 'gaussian':
            t_in = t_in.unsqueeze(-1).repeat(1, 1, self.out_dim)  # [B, T, out_dim]
            # equation 22
            t_gaussian = torch.exp(-torch.pow((t_in - self.t_centers) / self.bias, 2))[:, :, :, None]  # [B, T, out_dim, 1]
            d = torch.sum(d_vec * t_gaussian, dim=2)  # [B, T, D]
        if self.kernel == 'exponential':
            t_in = t_in.unsqueeze(-1)
            d = d_vec[:, :, 0, :] + d_vec[:, :, 1, :] * torch.exp(d_vec[:, :, 2, :] * t_in + d_vec[:, :, 3, :])
        if self.kernel == 'global_exponential':
            t_in = t_in.unsqueeze(-1)
            d = self.exp_params[0] + self.exp_params[1] * torch.exp(self.exp_params[2] * t_in) + self.exp_params[3]
            t_indices = torch.floor(t_in * (self.out_dim - 1) / self.max_time).unsqueeze(-1).expand(-1, -1, -1, self.data_dim).long()

            d = d * torch.gather(d_vec, dim=2, index=t_indices).squeeze()

        # generalized current
        j = torch.sum(d * (x[:, :, 1] - x[:, :, 0]).double(), dim=-1)  # [B, T]

        return j  # [B, T, D]

    def get_entropy_production_rate(self, j, t):
        """
        estimates entropy production rate given probability current j and time points t
        :param j: probability current for each trajectory, time step and dimension [B, T, D]
        :param t: time points that j was evaluated at [B, T, 2]
        :return: entropy_production_rate for each time point [T]
        """
        dt = (t[:, :, 1] - t[:, :, 0])[0]    # [T]
        entropy_production_rate = self.estimators[self.estimator_type](j) / dt     # [B, T, D] -> [T]
        return entropy_production_rate

    def loss(self, entropy_production_rate) -> dict:
        """
        calculates loss as negative average of the ep rate over all time points
        https://arxiv.org/abs/2010.03852, equation 13
        :param entropy_production_rate: estimated ep rate for each time points [T]
        :return: loss [1]
        """
        if self.loss_weight_exponential_parameters is not None:
            loss = -torch.mean(self.loss_weight * entropy_production_rate)
        else:
            loss = -torch.mean(entropy_production_rate)
        return {'loss': loss}

    def metric(self, entropy_production_rate, t) -> dict:
        """
        calculates total entropy production as integral of ep rate
        :param entropy_production_rate: average entropy production rate at each time point [T]
        :param t: time points [B, T, 2]
        :return: dict('total_entropy_production')
        """

        #midpoints = (t[0, :, 1] + t[0, :, 0]) / 2
        #total_entropy_production = torch.tensor(simpson(entropy_production_rate.cpu().numpy(),
        #                                                     midpoints.cpu().numpy(), even="avg"))
        # total_entropy_production = simpson_integrate(entropy_production_rate, t[0, :, 0], return_only_last=True)
        # total_entropy_production = simpson_integrate(entropy_production_rate, t)

        # if self.true_total_ep is not None:
        #     stats[f"ratio_to_true_total_ep"] = stats[f"total_entropy_production"] / self.true_total_ep

        total_entropy_production = torch.sum(entropy_production_rate * (t[0, :, 1] - t[0, :, 0]))


        return {'total_entropy_production': total_entropy_production}


    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        x = minibatch['data'].to(self.device)          # [B,T,2,D]
        t = minibatch['time_point'].to(self.device)     # [B,T,2]
        self._set_loss_weight(t)

        optimizer.zero_grad()

        j = self.forward((x, t))  # [B,T,2,D], [B,T,2] -> [B,T]
        entropy_production_rate = self.get_entropy_production_rate(j, t)
        loss_stats = self.loss(entropy_production_rate)
        metric_stats = self.metric(entropy_production_rate, t)

        loss_stats['loss'].backward()
        optimizer.step()

        return loss_stats | metric_stats

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

    def _set_loss_weight(self, t):
        if not hasattr(self, 'loss_weight') and self.loss_weight_exponential_parameters is not None:
            a, b = self.loss_weight_exponential_parameters
            self.loss_weight = a * torch.exp(- t[0, :, 0] * b) + 1

