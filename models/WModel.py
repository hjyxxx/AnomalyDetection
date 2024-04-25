import numpy as np
import torch
from torch import nn

from models.components.embedding import TimeEmbedding, FreEmbedding, WaveletEmbedding
from models.components.gcn import ChebyshevConv


class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        """
        :param x: (B, L, C), B: 批次大小, L: 节点数, C: 特征维度
        :return:
        """
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, gcn_layer, in_channels, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.gcn_layer = gcn_layer

        # self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=in_channels, kernel_size=1, bias=False)
        #
        # self.dropout = nn.Dropout(dropout)
        # self.activate = nn.ReLU()

    def forward(self, x):
        """

        :param x: (B, L, C), B: 批次大小, L: 节点数, C: 特征维度
        :return:
        """

        y = self.gcn_layer(x)

        # y = self.dropout(self.activate(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name

        e_layers = configs.e_layers

        fusion_layer = 3

        assert fusion_layer <= e_layers

        in_features = configs.in_features
        out_features = configs.out_features
        embedding_channels = configs.embedding_channels
        d_ff = configs.d_ff
        seg_len = configs.seg_len
        self.pose_num = configs.pose_num
        patch_len = configs.patch_len
        patch_stride = configs.patch_stride
        patch_padding = configs.patch_padding

        dropout = configs.dropout

        patch_num = int(np.floor((seg_len + patch_padding - patch_len) / patch_stride) + 1)
        adj = torch.ones((patch_num * self.pose_num, patch_num * self.pose_num))

        self.embedding_wave = WaveletEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                               patch_len=patch_len, patch_stride=patch_stride, patch_padding=patch_padding)

        self.encoder_wave = Encoder(
            encoder_layers=[
                EncoderLayer(
                    gcn_layer=ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3,
                                            adj=adj, dropout=dropout),
                    in_channels=embedding_channels,
                    d_ff=d_ff,
                    dropout=dropout
                ) for _ in range(fusion_layer)
            ]
        )

        # Decoder Head
        if self.task_name == 'rec':
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * seg_len)
            )
        elif self.task_name == 'pre':
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * self.for_len)
            )

        elif self.task_name == 'pr':
            self.head_rec = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * seg_len)
            )
            self.head_pre = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * self.for_len)
            )
        elif self.task_name == 'ws':
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * seg_len)
            )

    def emb_wave(self, x):
        """
        小波域嵌入
        :param x: (B, C, T, V)
        :return:
        """
        emb = self.embedding_wave(x)

        return emb

    def forward_wave(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        # 小波域嵌入
        emb = self.emb_wave(x)
        enc_out = self.encoder_wave(emb)

        return enc_out

    def decode(self, x):
        """
        :param x: (B, N, embedding_channels)
        :return:
        """

        # (B, N, embedding_channels)-->(B, V, patch_num, embedding_channels)
        x = x.reshape(x.shape[0], self.pose_num, -1, x.shape[-1])

        if self.task_name == 'rec':
            # (B, V, patch_num, embedding_channels)-->(B, V, seg_len*C)
            y = self.head(x)
            return y
        elif self.task_name == 'pre':
            # (B, V, patch_num, embedding_channels)-->(B, V, for_len*C)
            y = self.head(x)
            return y
        elif self.task_name == 'pr':
            # (B, V, patch_num, embedding_channels)-->(B, V, seg_len*C)
            y_rec = self.head_rec(x)
            # (B, V, patch_num, embedding_channels)-->(B, V, for_len*C)
            y_pre = self.head_pre(x)
            return y_rec, y_pre
        elif self.task_name == 'ws':
            # (B, V, patch_num, embedding_channels)-->(B, V, seg_len*C)
            y = self.head(x)
            return y

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """

        b, _, t, v = x.shape

        wave = self.forward_wave(x)

        # (B, N, C)
        res = wave

        if self.task_name == 'rec':
            y = self.decode(res)
            y = y.reshape(b, v, t, -1).permute(0, 3, 2, 1).contiguous()

            return y