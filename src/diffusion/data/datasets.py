import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision
import os
import re
import numpy as np
from math import floor, ceil
from sklearn.datasets import make_classification

import json

class ImageDataset(Dataset):
    def __init__(self, device='cpu'):
        super(ImageDataset, self).__init__()

        data, labels = self._get_data()

        self.data = data.to(device)
        self.labels = labels.to(device)

        self.data_shape = self.data.size()[1:]
        self.n_classes = len(torch.unique(self.labels))

    @staticmethod
    def _get_data():
        raise NotImplementedError('_get_data() is not implemented')

    def __getitem__(self, item):
        return {'data': self.data[item],
                'label': self.labels[item]}

    def __len__(self):
        return self.data.size(0)
class MNISTDataset(ImageDataset):
    def __init__(self, device, set='train', **kwargs):
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        super(MNISTDataset, self).__init__(device)

    def _get_data(self):
        if self.set == 'train':
            dataset = torchvision.datasets.MNIST(self.path, train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
        else:
            dataset = torchvision.datasets.MNIST(self.path, train=False, download=True,
                                                 transform=torchvision.transforms.ToTensor())
            set_len = len(dataset) // 2
        data = dataset.data
        labels = dataset.targets
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.set == 'test':
            data = data[:set_len]
            labels = labels[:set_len]
        elif self.set == 'valid':
            data = data[set_len:]
            labels = labels[set_len:]


        return data, labels
class CIFAR10Dataset(ImageDataset):
    def __init__(self, device, set='train', **kwargs):
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        super(CIFAR10Dataset, self).__init__(device)

    def _get_data(self):
        if self.set == 'train':
            dataset = torchvision.datasets.CIFAR10(self.path, train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
        else:
            dataset = torchvision.datasets.CIFAR10(self.path, train=False, download=True,
                                                 transform=torchvision.transforms.ToTensor())
            set_len = len(dataset) // 2
        data = dataset.data
        labels = dataset.targets
        data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.set == 'test':
            data = data[:set_len]
            labels = labels[:set_len]
        elif self.set == 'valid':
            data = data[set_len:]
            labels = labels[set_len:]

        return data, labels
class TrajectoryDataset(Dataset):
    """
    Data set for trajectories. Loads
    trajectories from trajectories.npy saved as shape [n_trajects, traject_length, data_dim]
    time points from and time_points.npy saved as shape [n_trajects, traject_length]
    and returns them as pairs of subsequent time points
    data: trajectory pairs of shape [batch_size, traject_len, 2, data dim]
    time_points: time points of shape [batch_size, traject_len, 2]
    """
    def __init__(self, set='train', **kwargs):
        super(TrajectoryDataset, self).__init__()
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        self.train_fraction = kwargs.get('train_fraction', .8)

        data, time_points = self._get_data()

        self.data = data.float()                 # [batch_size, traject_length, 2, data_dim]
        self.time_points = time_points.float()   # [batch_size, traject_length, 2]

        self.data_shape = self.data.size(-1)
        self.max_time = float(torch.max(time_points))

    def _get_data(self):
        path_to_data = os.path.join(self.path, 'trajectories.npy')
        path_to_time_points = os.path.join(self.path, 'time_points.npy')
        path_to_exact_epr = os.path.join(self.path, 'exact_epr.npy')

        data = torch.from_numpy(np.load(path_to_data))  # [n_trajects, traject_length, data_dim]
        time_points = torch.from_numpy(np.load(path_to_time_points))    # [n_trajects, traject_length]
        if os.path.exists(path_to_exact_epr):
            self.exact_epr = torch.from_numpy(np.load(path_to_exact_epr))[:-1]

        # shuffle data and split according to set
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(data.size(0))
        data = data[shuffled_indices]
        n_train_samples = floor(self.train_fraction * data.size(0))
        n_valid_samples = ceil((1 - self.train_fraction) / 2 * data.size(0))
        if self.set == 'train':
            data = data[:n_train_samples]
            time_points = time_points[:n_train_samples]
        if self.set == 'valid':
            data = data[n_train_samples:(n_train_samples + n_valid_samples)]
            time_points = time_points[n_train_samples:(n_train_samples + n_valid_samples)]
        if self.set == 'test':
            data = data[(n_train_samples + n_valid_samples):]
            time_points = time_points[(n_train_samples + n_valid_samples):]

        n_trajects = data.shape[0]
        dim = data.shape[-1]

        data = data.repeat_interleave(2, dim=1)         # [n_trajects, 2 * traject_length, data_dim]
        data = data[:, 1:-1]                            # [n_trajects, (traject_length - 1) * 2, data_dim]
        data = data.reshape(n_trajects, -1, 2, dim)     # [n_trajects, traject_length - 1, 2, data_dim]

        time_points = time_points.repeat_interleave(2, dim=1)
        time_points = time_points[:, 1:-1]
        time_points = time_points.reshape(n_trajects, -1, 2)

        return data, time_points

    def __getitem__(self, item):
        return {'data': self.data[item],
                'time_point': self.time_points[item]}

    def __len__(self):
        return self.data.size(0)
class TrajectoryDatasetLazy(Dataset):
    """
    Data set for trajectories. Loads
    trajectories from trajectories.npy saved as shape [n_trajects, traject_length, data_dim]
    time points from and time_points.npy saved as shape [n_trajects, traject_length]
    and returns them as pairs of subsequent time points
    data: trajectory pairs of shape [batch_size, traject_len, 2, data dim]
    time_points: time points of shape [batch_size, traject_len, 2]
    """
    def __init__(self, set='train', **kwargs):
        super(TrajectoryDatasetLazy, self).__init__()
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        self.train_fraction = kwargs.get('train_fraction', .8)
        self.max_time_step = kwargs.get('max_time_step', None)


        self._prep_data()

        self.data_shape = self.data.shape[-1]
        self.max_time = float(np.max(self.time_points))

    def _prep_data(self):
        path_to_data = os.path.join(self.path, 'trajectories.dat')
        path_to_time_points = os.path.join(self.path, 'time_points.npy')
        path_to_exact_epr = os.path.join(self.path, 'exact_epr.npy')

        data_shape = (get_values_from_file_name(self.path, 'num_samples'),
                      get_values_from_file_name(self.path, 'num_steps'),
                      get_values_from_file_name(self.path, 'dim'))

        self.data = np.memmap(path_to_data, dtype='float32', mode='r', shape=data_shape)  # [n_trajects, traject_length, data_dim]
        self.time_points = np.load(path_to_time_points, mmap_mode='r')    # [n_trajects, traject_length]
        if self.max_time_step is not None:
            self.data = self.data[:, :self.max_time_step]
            self.time_points = self.time_points[:, :self.max_time_step]
        if os.path.exists(path_to_exact_epr):
            self.exact_epr = torch.from_numpy(np.load(path_to_exact_epr))[:-1]
            if self.max_time_step is not None:
                self.exact_epr = self.exact_epr[:self.max_time_step-1]
        # shuffle data and split according to set
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(self.data.shape[0])
        n_train_samples = floor(self.train_fraction * self.data.shape[0])
        n_valid_samples = ceil((1 - self.train_fraction) / 2 * self.data.shape[0])
        if self.set == 'train':
            self.indices = shuffled_indices[:n_train_samples]
        if self.set == 'valid':
            self.indices = shuffled_indices[n_train_samples:(n_train_samples + n_valid_samples)]
        if self.set == 'test':
            self.indices = shuffled_indices[(n_train_samples + n_valid_samples):]
    def __getitem__(self, item):
        data = torch.tensor(self.data[self.indices[item]].copy()).float()
        time_points = torch.tensor(self.time_points[self.indices[item]].copy()).float()
        dim = data.shape[-1]

        data = data.repeat_interleave(2, dim=0)  # [2 * traject_length, data_dim]
        data = data[1:-1]  # [(traject_length - 1) * 2, data_dim]
        data = data.reshape(-1, 2, dim)  # [traject_length - 1, 2, data_dim]

        time_points = time_points.repeat_interleave(2, dim=0)
        time_points = time_points[1:-1]
        time_points = time_points.reshape(-1, 2)

        return {'data': data,
                'time_point': time_points}

    def __len__(self):
        return self.indices.shape[0]
class TrajectoryDatasetAE(Dataset):
    def __init__(self, set='train', **kwargs):
        super(TrajectoryDatasetAE, self).__init__()
        self.n_time_steps = kwargs.get('n_time_steps', None)
        self.max_time_step = kwargs.get('max_time_step', None)
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        self.train_fraction = kwargs.get('train_fraction', .8)

        time_points = self._get_data()

        self.time_points = time_points.float()  # [batch_size, traject_length, 2]

        self.data_shape = self.data.shape[-1]
        self.max_time = float(torch.max(time_points))

    def _get_data(self):
        path_to_data = os.path.join(self.path, 'latent.dat')
        path_to_data_shape = os.path.join(self.path, 'data_shape.npy')
        path_to_lr = os.path.join(self.path, 'learn_rate.pt')
        path_to_mi = os.path.join(self.path, 'mi.pt')
        path_to_loss = os.path.join(self.path, 'loss.pt')
        path_to_bounds = os.path.join(self.path, 'bounds.pt')

        data_shape = tuple(np.load(path_to_data_shape))
        self.data = np.memmap(path_to_data, dtype='float32', mode='r', shape=data_shape)  # [n_trajects, traject_length, data_dim]
        lr = torch.load(path_to_lr, map_location="cpu")  # [total_series_length]
        lr[0] = 0
        seq_len = data_shape[1]
        if os.path.exists(path_to_mi):
            self.mi = torch.load(path_to_mi, map_location="cpu")[:seq_len-1]  # [total_series_length]
        if os.path.exists(path_to_loss):
            self.loss = torch.load(path_to_loss, map_location="cpu") [:seq_len-1] # [total_series_length]
        if os.path.exists(path_to_bounds):
            self.bounds = torch.load(path_to_bounds, map_location="cpu")  # [total_series_length]

        if self.max_time_step is not None:
            self.data = self.data[:, :self.max_time_step]
            lr = lr[:self.max_time_step]
            if hasattr(self, 'mi'):
                self.mi = self.mi[:self.max_time_step-1]
            if hasattr(self, 'loss'):
                self.loss = self.loss[:self.max_time_step-1]
            if hasattr(self, 'bounds'):
                self.bounds = self.bounds[:self.max_time_step-1]

        # shuffle data and split according to set
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(data_shape[0])
        n_train_samples = floor(self.train_fraction * data_shape[0])
        n_valid_samples = ceil((1 - self.train_fraction) / 2 * data_shape[0])
        if self.set == 'train':
            self.indices = shuffled_indices[:n_train_samples]
        if self.set == 'valid':
            self.indices = shuffled_indices[n_train_samples:(n_train_samples + n_valid_samples)]
        if self.set == 'test':
            self.indices = shuffled_indices[(n_train_samples + n_valid_samples):]

        n_series = len(self.indices)

        time_points = torch.cumsum(lr, 0)  # [total_series_length]
        time_points = time_points.repeat_interleave(2, dim=0)  # [total_series_length * 2]
        time_points = time_points[1:-1]  # [(total_series_length - 1) * 2]
        time_points = time_points.unsqueeze(0).repeat(n_series, 1)  # [n_series, (total_series_length - 1) * 2]
        time_points = time_points.reshape(n_series, -1, 2)  # [n_series, total_series_length - 1, 2]
        # if self.n_time_steps is not None:
        #     idx = torch.arange(0, seq_len - 1, step=int(seq_len / self.n_time_steps))
        #     time_points = time_points[:, idx]
        #     if os.path.exists(path_to_mi):
        #         self.mi = self.mi[idx]
        #     if os.path.exists(path_to_loss):
        #         self.loss = self.loss[idx]
        print('data_shape:', (len(self.indices), *data_shape[1:]))

        return time_points

    def __getitem__(self, item):
        data = torch.from_numpy(self.data[self.indices[item]].copy())
        data = data.repeat_interleave(2, dim=0)  # [n_series, total_series_length * 2, data_dim]
        data = data[1:-1]  # [n_series, (total_series_length - 1) * 2, data_dim]
        data = data.reshape(-1, 2, data.size(-1))  # [n_series, total_series_length - 1, 2, data_dim]
        return {'data': data,
                'time_point': self.time_points[item]}

    def __len__(self):
        return len(self.indices)

class DiffusionTrajectoryDataset(TrajectoryDataset):
    def __init__(self, set='train', **kwargs):
        super(TrajectoryDataset, self).__init__()
        self.set = set
        self.seed = kwargs.get('seed', 1)
        self.path = kwargs.get('path', './data')
        self.path_to_image_data = kwargs.get('path_to_image_data', '.')
        self.train_fraction = kwargs.get('train_fraction', .8)
        self.model = kwargs.get('model', 'DDPM')
        self.num_steps = kwargs.get('num_steps', 100)
        model_list = ['DDPM']
        assert self.model in model_list, f'model must be one of {model_list}'

        data, time_points = self._get_data()

        self.data = data.float()                 # [batch_size, traject_length, 2, data_dim]
        self.time_points = time_points.float()   # [batch_size, traject_length, 2]

        self.data_shape = self.data.size(-1)
        self.max_time = torch.max(time_points)

    def _get_data(self):
        image_dataset = torchvision.datasets.CIFAR10(root=self.path_to_image_data, download=True, train=True)
        images = torch.tensor(image_dataset.data).permute(0, 3, 1, 2)

        model = eval(self.model)(self.num_steps)
        model.forward(images)
        exit()

        image_data = None

class LatentDataset:
    def __init__(self, set='train', **kwargs):
        self.set = set
        self.path = kwargs.get('path')

        self.data, self.latent = self._get_data()

        self.data_shape = self.data.size(-1)
        self.latent_shape = self.latent.size(-1)

        path_to_ml_sol = os.path.join(self.path, 'max_likelihood_sol.json')
        if os.path.exists(path_to_ml_sol):
            with open(path_to_ml_sol) as file:
                self.max_likelihood_sol = json.load(file)

    def _get_data(self):
        path_to_data = os.path.join(self.path, f'{self.set}.csv')
        path_to_latent = os.path.join(self.path, f'{self.set}_latent.csv')
        data = torch.tensor(np.genfromtxt(path_to_data, delimiter=','))
        latent = torch.tensor(np.genfromtxt(path_to_latent, delimiter=','))
        return data, latent

    def __getitem__(self, item):
        return {'data': self.data[item],
                'latent': self.latent[item]}

    def __len__(self):
        return self.data.size(0)


class TeacherDataset:

    def __init__(self, set='train', **kwargs):
        self.set = set

        self.data_shape = kwargs.get('data_dim', 1)
        self.n_classes = 2
        self.n_samples = kwargs.get('n_samples', 5000)
        self.seed = kwargs.get('seed', 1)

        self.teacher = torch.tensor(kwargs.get('teacher', [1] * self.data_shape), dtype=torch.float32)

        self.data, self.label = self._get_data()

    def _get_data(self):
        rng = torch.Generator().manual_seed(self.seed)
        data = torch.randn((self.n_samples, self.data_shape), generator=rng)
        label = (torch.matmul(data, self.teacher.unsqueeze(-1)) >= 0).squeeze(-1)
        return data, label

    def __getitem__(self, item):
        return {'data': self.data[item],
                'label': self.label[item]}
    def __len__(self):
        return self.data.size(0)
class ClassificationDataset:

    def __init__(self, set='train', **kwargs):
        self.set = set

        self.data_shape = kwargs.get('data_dim', 2)
        self.n_classes = kwargs.get('n_classes', 2)
        self.n_samples = kwargs.get('n_samples', 5000)
        self.seed = kwargs.get('seed', 1)
        self.data, self.label = self._get_data()

    def _get_data(self):
        data, label = make_classification(n_samples=self.n_samples, n_features=self.data_shape, n_classes=self.n_classes,
                                           n_repeated=0, n_redundant=0, random_state=self.seed)

        data = torch.tensor(data)
        label = torch.tensor(label)

        return data, label

    def __getitem__(self, item):
        return {'data': self.data[item],
                'label': self.label[item]}
    def __len__(self):
        return self.data.size(0)


def get_values_from_file_name(string, variable):
    pattern = rf"{variable}_(\d+)"
    match = re.search(pattern, string)
    if match:
        value = int(match.group(1))
        return value
    else:
        raise Exception(f'{variable} not contained in {string}')

if __name__ == '__main__':
    dataset = TeacherDataset(n_samples=20)


    print(dataset.data[dataset.label == 0])
    print(dataset.data[dataset.label == 1])