from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2, 2, (1, 1), bias=False)

    def forward(self, x):
        y = self.conv(x)

        return y
