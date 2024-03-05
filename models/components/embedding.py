import math

import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, embedding_channels, max_len=5000):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, embedding_channels).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_channels, 2).float()
                    * -(math.log(10000.0) / embedding_channels)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return:
        """
        return self.pe[:, :x.size(1)]


class ConvEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super(ConvEmbedding, self).__init__()

        self.emb_layer = nn.Sequential(
            nn.Conv1d(in_channels, embedding_channels, 1, bias=False),
            nn.Conv1d(embedding_channels, embedding_channels, 1, bias=False),
        )

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return: (B, L, embedding_channels)
        """

        x_emb = self.emb_layer(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x_emb
