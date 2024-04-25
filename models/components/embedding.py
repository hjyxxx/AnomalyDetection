import math

import numpy as np
import pywt
import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, embedding_layers, dropout=0.1):
        super(Embedding, self).__init__()
        self.embedding_layers = nn.ModuleList(embedding_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return: (N, S, embedding_channels)
        """
        for embedding_layer in self.embedding_layers:
            x = embedding_layer(x)

        return self.dropout(x)


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
        :return: (B, L, C)
        """
        return x + self.pe[:, :x.size(1)]


class LinearEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels):
        super(LinearEmbedding, self).__init__()

        self.emb_layer = nn.Linear(in_features=in_channels, out_features=embedding_channels)

    def forward(self, x):
        """

        :param x: (B, C, T, V)
        :return: (B, V*T, embedding_channels)
        """
        b, c, t, v = x.shape

        # (B, C, T, V)-->(B, V, T, C)
        x = x.permute(0, 3, 2, 1).contiguous()

        x = x.reshape(b, v*t, -1)

        x_emb = self.emb_layer(x)

        return x_emb


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


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels, patch_len, patch_stride, patch_padding):
        super(PatchEmbedding, self).__init__()

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.padding_patch_layer = nn.ReplicationPad2d((0, 0, 0, patch_padding))

        self.linear = nn.Linear(in_features=patch_len * in_channels, out_features=embedding_channels, bias=False)

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return: (B, V*patch, embedding_channels)
        """

        # (B, C, T, V)-->(B, C, T+padding, V)
        x = self.padding_patch_layer(x)

        # (B, C, T+padding, V)-->(B, C, patch, V, patch_len)
        x = x.unfold(dimension=-2, size=self.patch_len, step=self.patch_stride)

        # (B, C, patch, V, patch_len)-->(B, V, patch, patch_len, C)
        x = x.permute(0, 3, 2, 4, 1).contiguous()

        # (B, V, patch, patch_len, C)-->(B, V, patch, patch_len*C)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)

        # (B, V, patch, patch_len, C)-->(B, V, patch, embedding_channels)
        x = self.linear(x)

        # (B, V, patch, embedding_channels)-->(B, V*patch, embedding_channels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels, patch_len, patch_stride, patch_padding):
        super(TimeEmbedding, self).__init__()

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.padding_patch_layer = nn.ReplicationPad2d((0, 0, 0, patch_padding))

        self.linear = nn.Linear(in_features=patch_len * in_channels, out_features=embedding_channels, bias=False)

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return: (B, V*patch, embedding_channels)
        """

        # (B, C, T, V)-->(B, C, T+padding, V)
        x = self.padding_patch_layer(x)

        # (B, C, T+padding, V)-->(B, C, patch, V, patch_len)
        x = x.unfold(dimension=-2, size=self.patch_len, step=self.patch_stride)

        # (B, C, patch, V, patch_len)-->(B, V, patch, patch_len, C)
        x = x.permute(0, 3, 2, 4, 1).contiguous()

        # (B, V, patch, patch_len, C)-->(B, V, patch, patch_len*C)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)

        # (B, V, patch, patch_len, C)-->(B, V, patch, embedding_channels)
        x = self.linear(x)

        # (B, V, patch, embedding_channels)-->(B, V*patch, embedding_channels)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x


class FreEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels, patch_len, patch_stride, patch_padding):
        super(FreEmbedding, self).__init__()

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.padding_patch_layer = nn.ReplicationPad2d((0, 0, 0, patch_padding))

        self.linear = nn.Linear(in_features=in_channels * (int(patch_len / 2) + 1) * 2,
                                out_features=embedding_channels, bias=False)

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        # (B, C, T, V)-->(B, C, T+padding, V)
        x = self.padding_patch_layer(x)
        # (B, C, T+padding, V)-->(B, C, patch, V, patch_len)
        x = x.unfold(dimension=-2, size=self.patch_len, step=self.patch_stride)

        # (B, C, patch, V, patch_len)->(B, V, patch, patch_len, C)
        x = x.permute(0, 3, 2, 4, 1).contiguous()

        x = torch.fft.rfft(x, dim=-2, norm='ortho')        # FFT on patch_len
        # (B, V, patch, patch_len/2, C)
        x_real = x.real
        # (B, V, patch, patch_len/2, C)
        x_imag = x.imag

        # (B, V, patch, patch_len/2, C)-->(B, V, patch, patch_len/2*C)
        x_real = x_real.reshape(x_real.shape[0], x_real.shape[1], x_real.shape[2], -1)
        x_imag = x_imag.reshape(x_imag.shape[0], x_imag.shape[1], x_imag.shape[2], -1)

        # (B, V, patch, patch_len/2*C)-->(B, V, patch, patch_len/2*C*2)
        x_emb = torch.cat([x_real, x_imag], dim=-1)

        x_emb = self.linear(x_emb)

        x_emb = x_emb.reshape(x_emb.shape[0], -1, x_emb.shape[-1])

        return x_emb


class WaveletEmbedding(nn.Module):
    def __init__(self, in_channels, embedding_channels, patch_len, patch_stride, patch_padding):
        super(WaveletEmbedding, self).__init__()

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.padding_patch_layer = nn.ReplicationPad2d((0, 0, 0, patch_padding))

        self.linear = nn.Linear(in_features=in_channels * 28,
                                out_features=embedding_channels, bias=False)

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        # (B, C, T, V)-->(B, C, T+padding, V)
        x = self.padding_patch_layer(x)

        # (B, C, T+padding, V)-->(B, C, patch, V, patch_len)
        x = x.unfold(dimension=-2, size=self.patch_len, step=self.patch_stride)

        b, c, p, v, _ = x.shape

        # (B, C, patch, V, patch_len)-->(B, V, patch, C, patch_len)-->(B*V*patch*C, patch_len)
        y = x.permute(0, 3, 2, 1, 4).reshape(-1, self.patch_len).cpu()

        wp = pywt.WaveletPacket(data=y.numpy(), wavelet='db4', mode='symmetric', maxlevel=2)

        y = np.concatenate([wp['aa'].data, wp['ad'].data, wp['dd'].data, wp['da'].data], axis=-1)

        y = torch.from_numpy(y).to(device=x.device)

        # (B*V*patch*C, patch_len)-->(B, V*patch, C*patch_len)
        y = y.reshape(b, v*p, -1)

        y = self.linear(y)

        return y


class FreEmbedding2(nn.Module):
    def __init__(self, in_features, embedding_channels):
        super(FreEmbedding2, self).__init__()

        self.tokenEmb = nn.Linear(in_features=in_features, out_features=embedding_channels, bias=False)

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        b, _, t, v = x.shape
        # (B, C, T, V)-->(B, V, T, C)
        x = x.permute(0, 3, 2, 1).contiguous()

        # (B, V, T, C)-->(B, V, T, embedding_channels)
        x = self.tokenEmb(x)

        # (B, V, T, embedding_channels)-->(B, V, T//2, embedding_channels)
        x = torch.fft.rfft(x, dim=2, norm="ortho")

        return x


if __name__ == '__main__':

    x = torch.randn(128, 2, 12, 17)

    # model = TFWEmbedding(in_channels=2, embedding_channels=128, patch_len=4, stride=2, padding=2)
    # model = PatchEmbedding(in_channels=2, embedding_channels=128, patch_len=4, stride=2, padding=2)

    # model = FreEmbedding2(in_features=2, embedding_channels=128)
    model = WaveletEmbedding(in_channels=2, embedding_channels=128, patch_len=4, patch_stride=1, patch_padding=0)
    y = model(x)











