import torch
from torch import nn


class LocallyConnected(nn.Linear):
    def __init__(self, in_features, out_features, n_units, bias=True, device=None):
        """
        Layer with n_units linear blocks next to each other without connections between them
        :param in_features: number of inputs per unit
        :param out_features: number of outputs per unit
        :param n_units: number of units next to each other
        """
        super().__init__(in_features * n_units, out_features * n_units, bias=bias, device=device)
        I = torch.eye(n_units)
        ones = torch.ones((out_features, in_features))
        self.mask = torch.kron(I, ones)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.mask = self.mask.to(self.weight.device)
        return nn.functional.linear(input, self.weight * self.mask, self.bias)


class LocallyConnectedDifferentOutputs(nn.Linear):
    def __init__(self, in_features, out_feature_list, bias=True, device=None):
        """
        Layer with len(out_feature_list) linear blocks next to each other without connections between them
        :param in_features: number of inputs per unit
        :param out_feature_list: list of number of outputs for each unit
        """
        super().__init__(in_features * len(out_feature_list), sum(out_feature_list), bias=bias, device=device)
        ones_list = torch.stack([torch.ones((out_features, in_features)) for out_features in out_feature_list], dim=0)
        self.mask = torch.diag_embed(ones_list)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.mask = self.mask.to(self.weight.device)
        return nn.functional.linear(input, self.weight * self.mask, self.bias)