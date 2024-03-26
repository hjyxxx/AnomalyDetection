import numpy as np
import torch
from torch import nn

from models.components.embedding import Embedding, ConvEmbedding, LinearEmbedding, PatchEmbedding, TFWEmbedding
from models.components.gcn import ChebyshevConv, GraphAttentionConv
from models.components.graph import Graph


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

        seg_len = configs.seg_len
        self.for_len = configs.for_len

        in_features = configs.in_features
        out_features = configs.out_features
        embedding_channels = configs.embedding_channels
        d_ff = configs.d_ff
        dropout = configs.dropout

        fusion = configs.fusion

        self.node_num = configs.node_num

        # graph = Graph(layout='openpose', strategy='pure', seg_len=seg_len)
        #
        # adj = torch.from_numpy(graph.A)

        patch_len = configs.patch_len
        stride = configs.stride
        padding = configs.padding

        patch_num = int(np.floor((seg_len + padding - patch_len) / stride) + 1)

        adj = torch.ones((patch_num * self.node_num, patch_num * self.node_num))

        # self.embedding = PatchEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
        #                                 patch_len=patch_len, stride=stride, padding=padding)

        self.embedding = TFWEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                      patch_len=patch_len, stride=stride, padding=padding, mode=fusion)

        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    gcn_layer=ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3, adj=adj,
                                            dropout=dropout),
                    in_channels=embedding_channels,
                    d_ff=d_ff,
                    dropout=dropout
                ) for _ in range(e_layers)
            ]
        )

        # self.encoder = Encoder(
        #     encoder_layers=[
        #         EncoderLayer(
        #             gcn_layer=GraphAttentionConv(adj=adj, in_channels=embedding_channels,
        #                                          out_channels=embedding_channels, n_heads=1, is_concat=True,
        #                                          dropout=dropout),
        #             in_channels=embedding_channels,
        #             d_ff=d_ff,
        #             dropout=dropout
        #         ) for _ in range(e_layers)
        #     ]
        # )

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

    def emb(self, x):
        """
        :param x: (B, C, T, V)
        :return: (B, V*patch, embedding_channels)
        """
        x_emb = self.embedding(x)
        return x_emb

    def encode(self, x):
        """
        :param x: (B, N, embedding_channels)
        :return: (B, N, embedding_channels)
        """
        y = self.encoder(x)
        return y

    def decode(self, x):
        """
        :param x: (B, N, embedding_channels)
        :return:
        """

        # (B, N, embedding_channels)-->(B, V, patch_num, embedding_channels)
        x = x.reshape(x.shape[0], self.node_num, -1, x.shape[-1])

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

    def reconstruct(self, x):
        """

        :param x:
        :return:
        """
        b, c, t, v = x.shape

        x_emb = self.emb(x)

        x_enc = self.encode(x_emb)

        x_dec = self.decode(x_enc)

        y = x_dec.reshape(b, v, t, -1).permute(0, 3, 2, 1).contiguous()

        return y

    def prediction(self, x):
        """

        :param x:
        :return:
        """
        b, c, t, v = x.shape

        x_emb = self.emb(x)

        x_enc = self.encode(x_emb)

        x_dec = self.decode(x_enc)

        y = x_dec.reshape(b, v, self.for_len, -1).permute(0, 3, 2, 1).contiguous()

        return y

    def rec_and_pre(self, x):
        """
        :param x:
        :return:
        """
        b, c, t, v = x.shape

        x_emb = self.emb(x)

        x_enc = self.encode(x_emb)

        y_rec, y_pre = self.decode(x_enc)

        y_rec = y_rec.reshape(b, v, t, -1).permute(0, 3, 2, 1).contiguous()
        y_pre = y_pre.reshape(b, v, self.for_len, -1).permute(0, 3, 2, 1).contiguous()

        return y_rec, y_pre

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        if self.task_name == 'rec':
            y = self.reconstruct(x)

            return y

        elif self.task_name == 'pre':
            y = self.prediction(x)

            return y

        elif self.task_name == 'pr':
            y_rec, y_pre = self.rec_and_pre(x)
            return y_rec, y_pre

        elif self.task_name == 'ws':
            y = self.reconstruct(x)
            return y

        else:
            raise ValueError("Do Not Exists This Value: {}".format(self.task_name))