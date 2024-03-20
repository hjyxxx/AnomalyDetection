from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, conv_layer):
        super(GCNLayer, self).__init__()
        self.conv_layer = conv_layer

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return:
        """

        y = self.conv_layer(x)

        return y