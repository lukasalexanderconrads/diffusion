from torch import nn

class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()

        self.device = kwargs.get('device')

    def forward(self, input):
        pass


