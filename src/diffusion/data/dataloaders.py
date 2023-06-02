import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler
from diffusion.data.datasets import *


class DataLoaderMNIST:
    def __init__(self, device: torch.device, batch_size: int = 1, **kwargs):
        train_set = MNISTDataset(device, set='train', **kwargs)
        valid_set = MNISTDataset(device, set='valid', **kwargs)
        test_set = MNISTDataset(device, set='test', **kwargs)
        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        self.data_shape = self.train.dataset.data_shape
        self.n_classes = self.train.dataset.n_classes

class DataLoaderTrajectory:
    def __init__(self, batch_size: int = 1, **kwargs):
        standardize = kwargs.get('standardize', True)
        train_set = TrajectoryDataset(set='train', **kwargs)
        valid_set = TrajectoryDataset(set='valid', **kwargs)
        test_set = TrajectoryDataset(set='test', **kwargs)

        self.data_shape = train_set.data_shape
        self.max_time = train_set.max_time
        if standardize:
            train_set, valid_set, test_set = self.standardize(train_set, valid_set, test_set)

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    def standardize(self, train_set, valid_set, test_set):
        orig_shape = train_set.data.size()
        scaler = StandardScaler()
        train_set.data = torch.from_numpy(scaler.fit_transform(train_set.data.reshape(-1, self.data_shape)).reshape(orig_shape)).float()
        orig_shape = valid_set.data.size()
        valid_set.data = torch.from_numpy(scaler.transform(valid_set.data.reshape(-1, self.data_shape)).reshape(orig_shape)).float()
        orig_shape = test_set.data.size()
        test_set.data = torch.from_numpy(scaler.transform(test_set.data.reshape(-1, self.data_shape)).reshape(orig_shape)).float()
        return train_set, valid_set, test_set


class DataLoaderTrajectoryAE(DataLoaderTrajectory):
    def __init__(self, batch_size: int = 1, **kwargs):
        standardize = kwargs.get('standardize', True)
        train_set = TrajectoryDatasetAE(set='train', **kwargs)
        valid_set = TrajectoryDatasetAE(set='valid', **kwargs)
        test_set = TrajectoryDatasetAE(set='test', **kwargs)

        self.data_shape = train_set.data_shape
        self.max_time = train_set.max_time
        if standardize:
            train_set, valid_set, test_set = self.standardize(train_set, valid_set, test_set)

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)


class DataLoaderTrajectoryLazy:
    def __init__(self, batch_size: int = 1, **kwargs):
        train_set = TrajectoryDatasetLazy(set='train', **kwargs)
        valid_set = TrajectoryDatasetLazy(set='valid', **kwargs)
        test_set = TrajectoryDatasetLazy(set='test', **kwargs)

        self.data_shape = train_set.data_shape
        self.max_time = train_set.max_time

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

class DataLoaderLatent:
    def __init__(self, batch_size: int = 1, shuffle=True, **kwargs):
        train_set = LatentDataset(set='train', **kwargs)
        valid_set = LatentDataset(set='valid', **kwargs)
        test_set = LatentDataset(set='test', **kwargs)

        self.data_shape = train_set.data_shape
        self.latent_shape = train_set.latent_shape
        if hasattr(train_set, 'max_likelihood_sol'):
            self.max_likelihood_sol = train_set.max_likelihood_sol

        self.train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        self.valid = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        self.test = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
