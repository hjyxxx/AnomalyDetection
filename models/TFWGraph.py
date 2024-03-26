import numpy as np
import torch
from torch import nn

from models.components.embedding import TFWEmbedding, PatchEmbedding, TimeEmbedding, FreEmbedding
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


class FusionLayer(nn.Module):
    def __init__(self, adj, embedding_channels, num_latents=4, dropout=0.1):
        super(FusionLayer, self).__init__()

        self.time_gcn = ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3, adj=adj,
                                      dropout=dropout)

        self.fre_gcn = ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3, adj=adj,
                                     dropout=dropout)

        self.latents = nn.Parameter(torch.empty(1, num_latents, embedding_channels).normal_(std=0.02))
        self.scale_t = nn.Parameter(torch.zeros(1))
        self.scale_f = nn.Parameter(torch.zeros(1))

    def attention(self, q, k, v):
        """
        :param q: (B, N, C)
        :param k: (B, S, C)
        :param v: (B, S, D)
        :return:
        """
        b, n, c = q.shape

        # (B, N, C)(B, S, C)-->(B, N, S)
        attn = q.matmul(k.transpose(-2, -1)) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        # (B, N, S)(B, S, D)-->(B, N, D)
        y = attn.matmul(v).reshape(b, n, c)
        return y

    def fusion(self, time, fre):
        """
        :param time:
        :param fre:
        :return:
        """
        b, n, c = time.shape
        concat = torch.concat([time, fre], dim=1)

        fused_latents = self.attention(q=self.latents.expand(b, -1, -1), k=concat, v=concat)

        time = time + self.scale_t * self.attention(q=time, k=fused_latents, v=fused_latents)
        fre = fre + self.scale_f * self.attention(q=fre, k=fused_latents, v=fused_latents)

        return time, fre

    def forward(self, time, fre):
        """
        :param time: (B, N, C)
        :param fre: (B, N, C)
        :return:
        """

        time, fre = self.fusion(time, fre)

        time = time + self.time_gcn(time)
        fre = fre + self.fre_gcn(fre)

        return time, fre


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        e_layers = configs.e_layers

        fusion_layer = 2

        assert fusion_layer <= e_layers

        share_weights = False

        seg_len = configs.seg_len
        self.for_len = configs.for_len

        in_features = configs.in_features
        out_features = configs.out_features
        embedding_channels = configs.embedding_channels
        d_ff = configs.d_ff
        dropout = configs.dropout

        patch_len = configs.patch_len
        stride = configs.stride
        padding = configs.padding

        fusion = configs.fusion

        self.node_num = configs.node_num

        self.time_embedding = TimeEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                            patch_len=patch_len, stride=stride, padding=padding)

        self.fre_embedding = FreEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                          patch_len=patch_len, stride=stride, padding=padding)

        patch_num = int(np.floor((seg_len + padding - patch_len) / stride) + 1)

        adj = torch.ones((patch_num * self.node_num, patch_num * self.node_num))

        self.time_encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    gcn_layer=ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3, adj=adj,
                                            dropout=dropout),
                    in_channels=embedding_channels,
                    d_ff=d_ff,
                    dropout=dropout
                ) for _ in range(fusion_layer)
            ]
        )

        self.fre_encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    gcn_layer=ChebyshevConv(in_channels=embedding_channels, out_channels=embedding_channels, K=3, adj=adj,
                                            dropout=dropout),
                    in_channels=embedding_channels,
                    d_ff=d_ff,
                    dropout=dropout
                ) for _ in range(fusion_layer)
            ]
        )

        layers = []
        for i in range(e_layers - fusion_layer):
            layers.append(FusionLayer(adj, embedding_channels, num_latents=4, dropout=dropout))

        self.fusion_layers = nn.Sequential(*layers)

        self.decoder = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(in_features=embedding_channels * patch_num, out_features=out_features * self.for_len)
            )

    def forward_fre(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        emb = self.fre_embedding(x)
        y = self.fre_encoder(emb)

        return y

    def forward_time(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        emb =self.time_embedding(x)
        y = self.time_encoder(emb)

        return y

    def forward_fusion(self, time, fre):
        """
        :param time:
        :param fre:
        :return:
        """
        for layer in self.fusion_layers:
            time, fre = layer(time, fre)

        return time, fre

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """

        b, c, t, v = x.shape

        fre = self.forward_fre(x)
        time = self.forward_time(x)

        time, fre = self.forward_fusion(time, fre)

        res = (time + fre) * 0.5

        res = res.reshape(res.shape[0], self.node_num, -1, res.shape[-1])

        res = self.decoder(res)

        res = res.reshape(b, v, self.for_len, -1).permute(0, 3, 2, 1).contiguous()

        return res

