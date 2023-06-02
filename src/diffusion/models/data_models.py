from typing import List, Union

import numpy as np
import torch
from tqdm import tqdm

class LatentGaussianMixture:
    k: int
    mu: List[list]
    cov: List[list]
    pi: List[float]  # mixing proportion
    d_z: int
    d_x: int
    M: list
    B: list

    def __init__(self, k: int, d_z: int, d_x: int, pi: List[float],
                 mu: List[list], cov: List[list],
                 M: Union[list, np.ndarray],
                 B: Union[list, np.ndarray],
                 cov_0: Union[list, np.ndarray]):
        self.k = k
        self.d_z = d_z
        self.d_x = d_x
        self.mu = mu
        self.cov = cov
        self.pi = pi
        self.M = M
        self.B = B
        self.cov_0 = cov_0

    def sample(self, n: int, batch_size: int = None):
        if batch_size is None or n < batch_size:
            batch_size = n

        p_bar = tqdm(total=n, unit="data points",
                     desc="Generating data points from Latent Gaussian Mixture")
        total_ = 0
        while total_ < n:
            z = np.random.randn(batch_size, self.k, self.d_z)
            z = np.einsum('nmj,mjd->nmd', z, np.asarray(self.cov)) + np.asarray(self.mu)
            pi = np.random.multinomial(1, self.pi, size=batch_size)
            z = np.sum(pi[:, :, np.newaxis] * z, axis=1)
            mu_0 = np.matmul(z, self.M) + self.B

            x = np.random.randn(batch_size, self.d_x)
            x = np.einsum('nj,ji->ni', x, self.cov_0) + mu_0

            yield x, z
            total_ += batch_size
            p_bar.update(batch_size)

    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, val):
        assert val > 1
        self.__k = val

    @property
    def pi(self):
        return self.__pi

    @pi.setter
    def pi(self, val):
        assert len(val) == self.k and np.sum(val) == 1.0
        self.__pi = val

    @property
    def d_z(self):
        return self.__d_z

    @d_z.setter
    def d_z(self, val):
        assert 1 < val < 4
        self.__d_z = val

    @property
    def d_x(self):
        return self.__d_x

    @d_x.setter
    def d_x(self, val):
        assert val > 2
        self.__d_x = val

    @property
    def mu(self):
        return self.__mu

    @mu.setter
    def mu(self, val):
        assert len(val) == self.k
        assert (all(len(com) == self.__d_z for com in val))
        self.__mu = val

    @property
    def cov(self):
        return self.__cov

    @cov.setter
    def cov(self, val):
        assert len(val) == self.k
        assert (all(len(com) == self.__d_z for com in val))
        self.__cov = val

    @property
    def M(self):
        return self.__M

    @M.setter
    def M(self, val):
        if isinstance(val, list):
            val = np.asarray(val)
        assert val.shape == (self.d_z, self.d_x)
        self.__M = val

    @property
    def B(self):
        return self.__B

    @B.setter
    def B(self, val):
        if isinstance(val, list):
            val = np.asarray(val)
        assert val.shape == (1, self.d_x)
        self.__B = val

    @property
    def cov_0(self):
        return self.__cov_0

    @cov_0.setter
    def cov_0(self, val):
        if isinstance(val, list):
            val = np.asarray(val)
        assert val.shape == (self.d_x, self.d_x)
        self.__cov_0 = torch.from_numpy(val).type(torch.float32)


class LatentLinearGaussian:
    """
    Samples
    z ~ Normal(mean_z, cov_z)
    x ~ Normal(M.z + b, 1)
    where
    M = T(sigma, rho) . R_x(phi) . R_y(theta)
    with R_y, R_x rotations and T the shear mapping
    T =     1     sigma  rho
            0       1     0
            0       0     1
       1+sigma*rho  0   sigma
            0       1     0
          rho       0     1
    """
    def __init__(self, theta: int, phi: int, sigma: int, rho: int,
                 mean_z: list, std_dev_z: list, std_dev_x: int = 0.1,
                 use_random_lin_map: bool = False, use_hidden_variable_model: bool = True,
                 data_dim: int = 6):
        """

        mean_z, std_dev_z: gaussian distr. param in latent space
        std_dev_x: gaussian distr. param in data space. Here mean_x is set to zero

        (*) If use_hidden_variable_model is TRUE and use_random_lin_map is FALSE
        then M = T(sigma, rho) . R_x(phi) . R_y(theta) is used, where:
        theta:  rotation about 'y' axis
        phi:    rotation about 'x' axis
        sigma:  parameter for shear mapping 1
        rho:    parameter for shear mapping 2
        """
        if use_hidden_variable_model:
            # (1) linear map to data space:
            if use_random_lin_map:
                latent_dim = len(mean_z)
                self.lin_map = np.random.randn(data_dim, latent_dim)
            else:
                latent_dim, data_dim = 3, 6
                if len(mean_z) != latent_dim or len(std_dev_z) != latent_dim:
                    raise ValueError('If use_random_lin_map is False, the latent dim is fixed to 3 '
                                     '(i.e. len(mean_z), len(std_dev_z) should equal 3)')
                cos_phi = np.cos(phi)
                cos_theta = np.cos(theta)
                sin_phi = np.sin(phi)
                sin_theta = np.sin(theta)
                r_x = np.array([[1., 0., 0.],
                                [0., cos_phi, -sin_phi],
                                [0., sin_phi, cos_phi]])
                r_y = np.array([[cos_theta, 0., sin_theta],
                                [0., 1., 0.],
                                [-sin_theta, 0., cos_theta]])
                t =np.array([[1., sigma, rho],
                             [0., 1., 0.],
                             [0., 0., 1.],
                             [1. + sigma*rho, 0., sigma],
                             [0., 1., 0.],
                             [rho, 0., 1.]])
                self.lin_map = np.matmul(t, np.matmul(r_x, r_y))  # [6, 3]

            # (2) Gaussian parameters for latent distribution
            self.mean_z = np.array(mean_z)[np.newaxis, :]
            self.std_dev_z = np.array(std_dev_z)[np.newaxis, :]

            # (3) Gaussian parameters for data distribution
            self.mean_x = np.zeros((data_dim,))[np.newaxis, :]
            self.std_dev_x = std_dev_x * np.ones((data_dim,))[np.newaxis, :]

            # (4) compute covariance matrix of marginal (Gaussian) density
            cov_z = np.zeros((latent_dim, latent_dim))
            np.fill_diagonal(cov_z, self.std_dev_z[0] * self.std_dev_z[0])
            self.data_cov_matrix =  (std_dev_x * std_dev_x) * np.eye(data_dim) + \
                                    np.matmul(self.lin_map, np.matmul(cov_z, self.lin_map.T))
        else:
            latent_dim = None
            self.lin_map = np.random.randn(data_dim, data_dim)
            self.data_cov_matrix = np.matmul(self.lin_map, self.lin_map.T)

        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.use_hidden_variable_model = use_hidden_variable_model

    def sample(self, n: int, batch_size: int = None):
        if batch_size is None or n < batch_size:
            batch_size = n
        p_bar = tqdm(total=n, unit="data points",
                     desc="Generating data points from Linear Gaussian Model")
        total_ = 0
        while total_ < n:
            if self.use_hidden_variable_model:
                # sample latent variable
                z = np.random.randn(batch_size, self.latent_dim)
                z = self.mean_z + z * self.std_dev_z
                # sample data
                x = np.random.randn(batch_size, self.data_dim)
                x = np.matmul(z, self.lin_map.T) + self.mean_x + x * self.std_dev_x
            else:
                x = np.random.randn(batch_size, self.data_dim)
                x = np.matmul(x, self.lin_map.T)
                z = np.zeros((batch_size,2))
            yield x, z
            total_ += batch_size
            p_bar.update(batch_size)


