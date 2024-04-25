import torch
from torch import nn
import torch.nn.functional as F

from models.components.embedding import FreEmbedding2


class FreMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreMLP, self).__init__()

        self.real_linear = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.imag_linear = nn.Linear(in_features=in_channels, out_features=out_channels)

        self.real_bias = nn.Parameter(torch.randn(out_channels))
        self.imag_bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        :param x: (B, V, L, C)
        :return:
        """
        real = F.relu(
            self.real_linear(x.real) - self.imag_linear(x.imag) + self.real_bias
        )

        imag = F.relu(
            self.real_linear(x.imag) + self.imag_linear(x.real) + self.imag_bias
        )

        y = torch.stack([real, imag], dim=-1)
        y = F.softshrink(y, lambd=0.01)
        y = torch.view_as_complex(y)

        y = torch.fft.irfft(y, n=12, dim=2, norm="ortho")

        return y

if __name__ == '__main__':
    x = torch.randn(128, 2, 12, 17)

    emb_model = FreEmbedding2(in_features=2, embedding_channels=128)
    encoder_model = FreMLP(in_channels=128, out_channels=256)

    y = emb_model(x)
    y = encoder_model(y)