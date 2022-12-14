import torch
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
from math import floor, ceil



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
        self.max_time = torch.max(time_points)

    def _get_data(self):
        path_to_data = os.path.join(self.path, 'trajectories.npy')
        path_to_time_points = os.path.join(self.path, 'time_points.npy')

        data = torch.from_numpy(np.load(path_to_data))  # [n_trajects, traject_length, data_dim]
        time_points = torch.from_numpy(np.load(path_to_time_points))    # [n_trajects, traject_length]

        # shuffle data and split according to set
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(data.size(0))
        data = data[shuffled_indices]
        n_train_samples = floor(self.train_fraction * data.size(0))
        n_valid_samples = ceil((1 - self.train_fraction) / 2 * data.size(0))
        if self.set == 'train':
            data = data[:n_train_samples]
        if self.set == 'valid':
            data = data[n_train_samples:(n_train_samples + n_valid_samples)]
        if self.set == 'test':
            data = data[(n_train_samples + n_valid_samples):]

        n_trajects = data.shape[0]
        dim = data.shape[-1]
        if time_points.shape[0] == 1:
            time_points = time_points.repeat(n_trajects, 1)

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


class TrajectoryDatasetAE(TrajectoryDataset):
    def __init__(self, set='train', **kwargs):
        self.n_time_steps = kwargs.get('n_time_steps', None)
        super(TrajectoryDatasetAE, self).__init__(set, **kwargs)

    def _get_data(self):
        path_to_data = os.path.join(self.path, 'latent.pt')
        path_to_lr = os.path.join(self.path, 'learn_rate.pt')
        path_to_mi = os.path.join(self.path, 'mi.pt')
        path_to_loss = os.path.join(self.path, 'loss.pt')
        path_to_bounds = os.path.join(self.path, 'bounds.pt')

        data = torch.load(path_to_data, map_location="cpu")  # [n_series, total_series_length, data_dim]
        lr = torch.load(path_to_lr, map_location="cpu")  # [total_series_length]
        self.mi = torch.load(path_to_mi, map_location="cpu")  # [total_series_length]
        self.loss = torch.load(path_to_loss, map_location="cpu")  # [total_series_length]
        self.bounds = torch.load(path_to_bounds, map_location="cpu")  # [total_series_length]

        # shuffle data and split according to set
        rng = np.random.default_rng(self.seed)
        shuffled_indices = rng.permutation(data.size(0))
        data = data[shuffled_indices]
        n_train_samples = floor(self.train_fraction * data.size(0))
        n_valid_samples = ceil((1 - self.train_fraction) / 2 * data.size(0))
        if self.set == 'train':
            data = data[:n_train_samples]
        if self.set == 'valid':
            data = data[n_train_samples:(n_train_samples + n_valid_samples)]
        if self.set == 'test':
            data = data[(n_train_samples + n_valid_samples):]

        n_series, seq_len, dim = data.shape

        if self.n_time_steps is not None:
            idx1 = torch.arange(0, seq_len - 1, step=int(seq_len / self.n_time_steps))
            idx2 = 1 + idx1
            idx, _ = torch.sort(torch.cat((idx1, idx2), dim=0))
            data = data[:, idx].reshape(n_series, -1, 2, dim)  # [n_series, n_time_steps+1, 2, data_dim]

            #time_points = torch.cumsum(lr, 0)  # [total_series_length]
            time_points = lr[idx].reshape(-1, 2)  # [n_time_steps+1, 2]
            time_points = time_points.unsqueeze(0).repeat(n_series, 1, 1)  # [n_series, n_time_steps+1, 2]
        else:
            data = data.repeat_interleave(2, dim=1)  # [n_series, total_series_length * 2, data_dim]
            data = data[:, 1:-1]  # [n_series, (total_series_length - 1) * 2, data_dim]
            data = data.reshape(n_series, -1, 2, dim)  # [n_series, total_series_length - 1, 2, data_dim]

            #time_points = torch.cumsum(lr, 0)  # [total_series_length]
            time_points = lr.repeat_interleave(2, dim=0)  # [total_series_length * 2]
            time_points = time_points[1:-1]  # [(total_series_length - 1) * 2]
            time_points = time_points.unsqueeze(0).repeat(n_series, 1)  # [n_series, (total_series_length - 1) * 2]
            time_points = time_points.reshape(n_series, -1, 2)  # [n_series, total_series_length - 1, 2]

        return data, time_points

    def __getitem__(self, item):
        return {'data': self.data[item],
                'time_point': self.time_points[item]}