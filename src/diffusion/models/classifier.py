import torch
from diffusion.models.base import BaseModel
import math


class HebbianNeuron(BaseModel):
    def __init__(self, data_dim, **kwargs):
        super(HebbianNeuron, self).__init__(**kwargs)

        self.weight = torch.nn.Parameter(torch.randn((1, data_dim), device=self.device))

        self.learning_force_multiplier = 1


    def forward(self, input):
        activation = self.weight * input
        return activation

    def metrics(self, label, activation):

        prediction = activation > 0
        accuracy = torch.mean(torch.tensor(label == prediction, dtype=torch.float32))

        return {'accuracy': accuracy, 'weight': self.weight.data}



    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        data = minibatch['data'].to(self.device).float()
        label = minibatch['label'].to(self.device).long()

        activation = self.forward(data)

        stats = self.metrics(label, activation)

        label[label == 0] = -1
        lr = optimizer.param_groups[0]['lr']
        learning_force = self.learning_force_multiplier * data * label * lr
        thermal_noise = torch.randn(size=(1,), device=self.device) * math.sqrt(2)
        self.weight.data = self.weight.data + (- self.weight.data + learning_force) + thermal_noise

        return stats

    def valid_step(self, minibatch: torch.Tensor) -> dict:
        data = minibatch['data'].to(self.device).float()
        label = minibatch['label'].to(self.device).long()

        activation = self.forward(data)

        stats = self.metrics(label, activation)

        return stats

class Neuron(BaseModel):
    def __init__(self, data_dim, **kwargs):
        super(Neuron, self).__init__(**kwargs)

        self.data_dim = data_dim
        self.n_classes = kwargs.get('n_classes')


        out_dim = 1 if self.n_classes == 2 else self.n_classes

        self.neuron = torch.nn.Linear(data_dim, out_dim, bias=False).to(self.device)


    def forward(self, input):
        output = self.neuron(input)
        if self.n_classes == 2:
            prediction = torch.cat((output, 1-output), dim=-1)
        else:
            prediction = torch.nn.functional.softmax(output, dim=-1).squeeze()
        return prediction

    def loss(self, input, target):
        loss = torch.nn.functional.cross_entropy(input, target)
        return {'loss': loss}

    def train_step(self, minibatch: dict, optimizer: torch.optim.Optimizer) -> dict:
        data = minibatch['data'].to(self.device).float()
        label = minibatch['label'].to(self.device).long()

        # optimizer initialization
        optimizer.zero_grad()

        # forward pass
        prediction = self.forward(data)

        # backprop + update
        loss_stats = self.loss(prediction, label)
        loss_stats['loss'].backward()
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
        optimizer.step()

        return loss_stats

    def valid_step(self, minibatch: torch.Tensor) -> dict:
        data = minibatch['data'].to(self.device).float()
        label = minibatch['label'].to(self.device).long()

        # forward pass
        prediction = self.forward(data)

        loss_stats = self.loss(prediction, label)

        return loss_stats


