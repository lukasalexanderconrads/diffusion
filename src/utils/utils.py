import torch
from torch import nn
import yaml
from importlib import import_module
from collections import defaultdict
import numpy as np
import scipy as sp


def read_yaml(path):
    """
    read .yaml file and return as dictionary
    :param path: path to .yaml file
    :return: parsed file as dictionary
    """
    with open(path, 'r') as file:
        try:
            parsed_yaml = yaml.load(file, yaml.Loader)
        except yaml.YAMLError as exc:
            print(exc)
    file.close()

    return parsed_yaml

def create_instance(module_name, class_name, kwargs, *args):
    """
    create instance of a class
    :param module_name: str, module the class is in
    :param class_name: str, name of the class
    :param kwargs:
    :return: class instance
    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)
    return instance


class MetricAccumulator:
    def __init__(self):
        self.concat_keys = []
        self.reset()

    def update(self, metrics):
        for key, value in metrics.items():
            if key in self.concat_keys:
                self.metrics[key] = [] if self.metrics[key] == 0 else self.metrics[key]
                self.metrics[key] += [value]
            else:
                self.metrics[key] += value
        self.counter += 1

    def get_average(self):
        for key in self.metrics.keys():
            if not key in self.concat_keys:
                self.metrics[key] /= self.counter
        return self.metrics

    def reset(self):
        self.metrics = defaultdict(lambda: 0)
        self.counter = 0

    def exclude_keys_from_average(self, keys):
        """
        exclude some keys from averaging, collect them in a list instead
        :param keys: list of str
        """
        self.concat_keys += keys


def create_mlp(layer_dims, activation_fn=nn.ReLU(), out_activation=False, dropout=.0, out_dropout=False, layer_normalization=True, out_layernorm=False):
    n_layers = len(layer_dims)
    layers = []
    for layer_idx in range(n_layers - 1):
        # weight and bias
        layers.append(torch.nn.Linear(layer_dims[layer_idx], layer_dims[layer_idx + 1]))
        # layer notmalization
        if layer_normalization and (layer_idx != n_layers - 2 or out_layernorm):
            layers.append(nn.LayerNorm(layer_dims[layer_idx + 1]))
        # activation function
        if layer_idx != n_layers - 2 or out_activation:
            layers.append(activation_fn)
        # dropout
        if dropout > 0 and (layer_idx != n_layers - 2 or out_dropout):
            layers.append(torch.nn.Dropout(dropout))

    return torch.nn.Sequential(*layers)

def get_trainer(config):
    module_name = config['trainer']['module']
    class_name = config['trainer']['name']
    args = config['trainer']['args']
    trainer = create_instance(module_name, class_name, args, config)
    return trainer


def get_model(config, data_shape):
    module_name = config['model']['module']
    class_name = config['model']['name']
    args = config['model']['args']
    args = args | {'device': config['device']} if args is not None else {'device': config['device']}
    model = create_instance(module_name, class_name, args, data_shape)
    return model

def get_data_loader(config):
    module_name = config['loader']['module']
    class_name = config['loader']['name']
    args = config['loader']['args']
    loader = create_instance(module_name, class_name, args)
    return loader

def reset_parameters(model):
    model.apply(lambda w: w.reset_parameters() if hasattr(w, 'reset_parameters') else None)

def create_rbf(in_dim, rbf_dim, out_dim):
    rbf_layer = RBF(in_dim, rbf_dim)
    linear_layer = nn.Linear(rbf_dim, out_dim)
    return nn.Sequential(rbf_layer, linear_layer)

class RBF(nn.Module):
    def __init__(self, in_dim, out_dim, basis_function='exp'):
        super().__init__()
        self.center = nn.Parameter(torch.randn((1, in_dim, out_dim)))  # [1, in_dim, out_dim]
        self.variance = nn.Parameter(torch.exp(torch.randn((1, out_dim))))            # [1, out_dim]

        if basis_function == 'exp':
            self.activation = lambda x: torch.exp(-x)
        elif basis_function == 'inverse':
            self.activation = lambda x: 1 / torch.sqrt(x)

    def forward(self, x):
        """
        :param x: input [batch_size, in_dim]
        :return: output [batch_size, out_dim]
        """
        x = x.unsqueeze(-1)                                 # [batch_size, in_dim, 1]
        squared_entries = (x - self.center)**2              # [batch_size, in_dim, out_dim]
        squared_distances = torch.sum(squared_entries, dim=1)   # [batch_size, out_dim]
        out = self.activation(squared_distances / self.variance)
        return out


# https://stackoverflow.com/questions/18215163/cumulative-simpson-integration-with-scipy
def cumsimp(func,a,b,num):
    #Integrate func from a to b using num intervals.

    num*=2
    a=float(a)
    b=float(b)
    h=(b-a)/num

    output=4*func(a+h*np.arange(1,num,2))
    tmp=func(a+h*np.arange(2,num-1,2))
    output[1:]+=tmp
    output[:-1]+=tmp
    output[0]+=func(a)
    output[-1]+=func(b)
    return np.cumsum(output*h/3)

def integ1(x):
    return x

def integ2(x):
    return x**2

def integ0(x):
    return np.ones(np.asarray(x).shape)*5

def get_random_hermitian(dim, allow_singular=True, unitary=False, rng=None):
    """
    :param dim: size of the matrix
    :param allow_singular: if matrix is allowed to be singular
    :return: hermitian matrix of shape [dim, dim]
    """
    rng = np.random.default_rng() if rng is None else rng

    if unitary:
        A = rng.random((dim, dim))
        Q, _ = np.linalg.qr(A)
        return Q @ Q.T
    while True:
        A_root = rng.random((dim, dim))
        A = A_root @ A_root.T

        if allow_singular or not np.isclose(np.linalg.det(A), 0):
            return A

def get_random_matrix(dim, allow_singular=True, rng=None):
    """
    :param dim: size of the matrix
    :param allow_singular: if matrix is allowed to be singular
    :return: hermitian matrix of shape [dim, dim]
    """
    rng = np.random.default_rng() if rng is None else rng
    while True:
        A = rng.random((dim, dim))

        if allow_singular or not np.isclose(np.linalg.det(A), 0):
            return A

def get_random_diagonal_matrix(dim, allow_singular=True, rng=None):
    """
    :param dim: size of the matrix
    :param allow_singular: if matrix is allowed to be singular
    :return: hermitian matrix of shape [dim, dim]
    """
    rng = np.random.default_rng() if rng is None else rng
    while True:
        A = np.diag(rng.random((dim,)))

        if allow_singular or not np.isclose(np.linalg.det(A), 0):
            return A

def batched_diag(A):
    """
    if A is 2d: first dimension is batch dimension, second is diagonal.
        returns a 3d array which is a batch of diagonal matrices
    if A is 3d: first dimension is batch dimension, second and third are matrix dimensions
        returns a 2d array which is a batch of diagonals of the input matrices
    :return:
    """
    if A.ndim == 2:
        return np.apply_along_axis(np.diag, axis=1, arr=A)
    elif A.ndim == 3:
        return np.diagonal(A, axis1=1, axis2=2)
    else:
        raise Exception('Expected 2 or 3 dimensions, but got', A.ndim)


# def simpson_integrate(y, x, return_only_last=False):
#     """
#     :param y: function values of the function to integrate, tensor of shape [num_points]
#     :param x: points where the function was evaluated, tensor of shape [num_points]
#     :return:
#     """
#     if return_only_last:
#         return (x[-1] - x[0]) / len(y)
#     else:
#         normalizer = (x[1:] - x[0]) / torch.arange(1, len(y), device=y.device)
#         cumsum = torch.cumsum(y[:-1], dim=0)
#         cumsum[0] = 0
#         integral = normalizer * (y[0] / 2 + cumsum + y[1:])
#     return integral

def simpson_integrate(y, x, return_only_last=False):
    """
    y is a 1d-tensor of function values
    x is a 1d-tensor of the points where y was sampled
    returns: a 1d-tensor with length 1-len(y) that contains the value of the integral [x[0], x[n]] in entry n
    """
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")
    n = len(x)
    if n % 2 == 0:
        raise ValueError("n must be an odd integer.")

    h = (x[-1] - x[0]) / (n - 1)
    if return_only_last:
        integral = ((h/3) * (y[:-2:2] + 4*y[1:-1:2] + y[2::2])).sum()
    else:
        indices = torch.arange(1, n, 2)
        integral = torch.zeros_like(y)
        integral[indices] = (h / 3) * (y[indices - 1:-1:2] + 4 * y[indices::2] + y[indices + 1::2])
        integral = integral.cumsum(dim=0)
    return integral


if __name__ == '__main__':

    A = np.random.rand(100, 3, 3)
    print(batched_diag(A).shape)