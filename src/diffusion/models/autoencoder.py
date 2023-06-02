import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from diffusion.models.base import BaseModel
from diffusion.utils.metrics import mutual_information_data_rep
from diffusion.utils.helpers import create_mlp


class BaseAutoencoder(BaseModel):

    def __init__(self, data_dim, **kwargs):
        super(BaseAutoencoder, self).__init__(**kwargs)

        self.data_dim = data_dim
        self.latent_dim = kwargs.get('latent_dim')
        assert self.latent_dim is not None, 'latent_dim not provided to PPCA model'

        # Ground-truth covariance of Gaussian model (to compute the MI)
        self.ground_truth_cov = torch.tensor(kwargs['max_likelihood_sol']['ground_truth_data_cov_matrix'])

        # Maximum likelihood solution (for comparison)
        self.mean_max_lik = torch.tensor(kwargs['max_likelihood_sol']['mean_ml'])
        self.cov_max_lik = torch.tensor(kwargs['max_likelihood_sol']['covariance_matrix_ml'])

        # Mutual information with max. likelihood solution (for comparison)
        posterior_ml_cov = torch.tensor(kwargs['max_likelihood_sol']['encoder_covariance_ml'])
        posterior_ml_proj = torch.tensor(kwargs['max_likelihood_sol']['encoder_proj_ml'])
        self.mi_max_likelihood = mutual_information_data_rep(self.ground_truth_cov, posterior_ml_cov, posterior_ml_proj)

class PPCA(BaseAutoencoder):
    """
    Probabilistic PCA
    """
    def __init__(self, data_dim, **kwargs):
        super(PPCA, self).__init__(data_dim, **kwargs)
        # Linear map from latent to data space
        self.linear_map = torch.nn.Parameter(torch.rand(self.data_dim, self.latent_dim, device=self.device),
                                             requires_grad=True)
        torch.nn.init.normal_(self.linear_map, mean=0.0, std=0.01)

        # Decoder mean
        self.mean = torch.nn.Parameter(torch.zeros(self.data_dim, device=self.device), requires_grad=True)
        torch.nn.init.normal_(self.mean, mean=0.0, std=0.01)

        # Decoder std deviation
        self.sigma_square = torch.nn.Parameter(torch.full((1,), 0.1, device=self.device), requires_grad=True)

        self.sample = False


    def forward(self, input: torch.Tensor, sample=False):
        """
        input: [B, D]
        returns
        (i) the covariance of the exact posterior
        (ii) encoder projection  (Eq. 12.42 Bishop)
        (iii) samples from the exact posterior (optional)
        """
        m1 = torch.matmul(torch.t(self.linear_map), self.linear_map) \
             + torch.abs(self.sigma_square) * torch.eye(self.latent_dim, device=self.device)  # [latent_dim, latent_dim]
        m1 = torch.inverse(m1)

        enc_projection = torch.matmul(m1, torch.t(self.linear_map))  # encoder projection [data_dim, latent_dim]
        enc_cov = torch.abs(self.sigma_square) * m1

        if sample:
            # sample from posterior distribution
            mean_z = torch.matmul(input - self.mean.unsqueeze(0), torch.t(enc_projection))
            pz = MultivariateNormal(loc=mean_z,
                                    covariance_matrix=enc_cov)
            z = pz.sample()
        else:
            z = None

        return enc_cov, enc_projection, z

    def loss(self, x_target: torch.Tensor) -> dict:

        data_dim = x_target.shape[-1]
        ppca = MultivariateNormal(loc=self.mean,
                                  covariance_matrix=(torch.matmul(self.linear_map, self.linear_map.T)
                                                     + torch.abs(self.sigma_square) * torch.eye(data_dim, device=self.device)))
        loss = -torch.mean(ppca.log_prob(x_target))
        # # Exact maximum likelihood distribution (for comparison)
        # max_lik_distr = MultivariateNormal(loc=torch.tensor(self.mean_max_lik.to(self.device)),
        #                                    covariance_matrix=torch.tensor(self.cov_max_lik.to(self.device)))
        # max_likelihood_sol = - torch.mean(max_lik_distr.log_prob(x_target))
        #
        # stats['max_likelihood_sol'] = max_likelihood_sol
        # stats['d_loss_wrt_ml'] = torch.abs(torch.mean(loss) - max_likelihood_sol)

        return {'loss': loss}

    def metric(self, cov_z: torch.Tensor, enc_proj: torch.Tensor):
        """
        Computes the mutual information between data and representation
        """
        if self.ground_truth_cov is not None:
            mi = mutual_information_data_rep(self.ground_truth_cov.to(self.device),
                                             cov_z,
                                             enc_proj)
        else:
            mi = torch.tensor(0.0)

        mi_diff = torch.abs(mi - self.mi_max_likelihood)

        return {'mi': mi, 'mi_ml': self.mi_max_likelihood, 'mi_diff': mi_diff}


    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        data = minibatch['data'].to(self.device).float()

        # optimizer initialization
        optimizer.zero_grad()

        # forward pass
        cov_z, enc_proj, _ = self.forward(data, self.sample)

        # backprop + update
        loss_stats = self.loss(data)
        loss_stats['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()

        metric_stats = self.metric(cov_z, enc_proj)

        return loss_stats | metric_stats

    def valid_step(self, minibatch: torch.Tensor) -> dict:
        data = minibatch['data'].to(self.device).float()

        # Evaluate model
        cov_z, enc_proj, _ = self.forward(data, self.sample)
        loss_stats = self.loss(data)

        metric_stats = self.metric(cov_z, enc_proj)

        return loss_stats | metric_stats

class AE(BaseAutoencoder):
    """
    Variational Autoencoder
    """
    def __init__(self, data_dim, **kwargs):
        super(AE, self).__init__(data_dim, **kwargs)

        ### encoder
        encoder_layer_dims = [self.data_dim] + kwargs.get('encoder_layer_dims', []) + [self.latent_dim]
        self.encoder = create_mlp(encoder_layer_dims)

        ### decoder
        decoder_layer_dims = [self.latent_dim] + kwargs.get('decoder_layer_dims', []) + [self.data_dim]
        self.decoder = create_mlp(decoder_layer_dims)


    def forward(self, input: torch.Tensor):
        """
        input: [B, D]
        returns
        (i) reconstructed input
        (ii) encoder mean
        (iii) encoder variance
        (iv) latent sample
        """
        z = self.encoder(input)
        reconstruction = self.decoder(z)

        return reconstruction, z


    def loss(self, reconstruction, target) -> dict:

        loss = torch.nn.functional.mse_loss(reconstruction, target, reduction='mean')

        return {'loss': loss}

    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        data = minibatch['data'].to(self.device).float()

        # optimizer initialization
        optimizer.zero_grad()

        # forward pass
        reconstruction, _ = self.forward(data)

        # backprop + update
        loss_stats = self.loss(reconstruction, data)
        loss_stats['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()

        return loss_stats

    def valid_step(self, minibatch: torch.Tensor) -> dict:
        data = minibatch['data'].to(self.device).float()

        # forward pass
        reconstruction, _ = self.forward(data)
        loss_stats = self.loss(reconstruction, data)


        return loss_stats

class VAE(BaseAutoencoder):
    """
    Variational Autoencoder
    """
    def __init__(self, data_dim, **kwargs):
        super(VAE, self).__init__(data_dim, **kwargs)

        ### encoder
        encoder_layer_dims = [self.data_dim] + kwargs.get('encoder_layer_dims', []) + [2 * self.latent_dim]
        self.encoder = create_mlp(encoder_layer_dims)

        ### decoder
        decoder_layer_dims = [self.latent_dim] + kwargs.get('decoder_layer_dims', []) + [self.data_dim]
        self.decoder = create_mlp(decoder_layer_dims)


    def forward(self, input: torch.Tensor, sample=False):
        """
        input: [B, D]
        returns
        (i) reconstructed input
        (ii) encoder mean
        (iii) encoder variance
        (iv) latent sample
        """
        encoder_out = self.encoder(input)
        mean = encoder_out[:self.latent_dim]
        variance = encoder_out[self.latent_dim:]

        noise = None




    def loss(self, x_target: torch.Tensor) -> dict:

        data_dim = x_target.shape[-1]
        ppca = MultivariateNormal(loc=self.mean,
                                  covariance_matrix=(torch.matmul(self.linear_map, self.linear_map.T)
                                                     + torch.abs(self.sigma_square) * torch.eye(data_dim, device=self.device)))
        loss = -torch.mean(ppca.log_prob(x_target))
        # # Exact maximum likelihood distribution (for comparison)
        # max_lik_distr = MultivariateNormal(loc=torch.tensor(self.mean_max_lik.to(self.device)),
        #                                    covariance_matrix=torch.tensor(self.cov_max_lik.to(self.device)))
        # max_likelihood_sol = - torch.mean(max_lik_distr.log_prob(x_target))
        #
        # stats['max_likelihood_sol'] = max_likelihood_sol
        # stats['d_loss_wrt_ml'] = torch.abs(torch.mean(loss) - max_likelihood_sol)

        return {'loss': loss}

    def metric(self, cov_z: torch.Tensor, enc_proj: torch.Tensor):
        """
        Computes the mutual information between data and representation
        """
        if self.ground_truth_cov is not None:
            mi = mutual_information_data_rep(self.ground_truth_cov.to(self.device),
                                             cov_z,
                                             enc_proj)
        else:
            mi = torch.tensor(0.0)

        mi_diff = torch.abs(mi - self.mi_max_likelihood)

        return {'mi': mi, 'mi_ml': self.mi_max_likelihood, 'mi_diff': mi_diff}


    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        data = minibatch['data'].to(self.device).float()

        # optimizer initialization
        optimizer.zero_grad()

        # forward pass
        cov_z, enc_proj, _ = self.forward(data, self.sample)

        # backprop + update
        loss_stats = self.loss(data)
        loss_stats['loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()

        metric_stats = self.metric(cov_z, enc_proj)

        return loss_stats | metric_stats

    def valid_step(self, minibatch: torch.Tensor) -> dict:
        data = minibatch['data'].to(self.device).float()

        # Evaluate model
        cov_z, enc_proj, _ = self.forward(data, self.sample)
        loss_stats = self.loss(data)

        metric_stats = self.metric(cov_z, enc_proj)

        return loss_stats | metric_stats
