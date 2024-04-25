import numpy as np
import torch
from torch import nn

from models.components.embedding import TimeEmbedding, FreEmbedding, WaveletEmbedding
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


class FusionLayer(nn.Module):
    def __init__(self, embedding_channels, num_latents, time_forward_layer, fre_forward_layer, wave_forward_layer):
        super(FusionLayer, self).__init__()

        self.time_forward_layer = time_forward_layer
        self.fre_forward_layer = fre_forward_layer
        self.wave_forward_layer = wave_forward_layer

        self.latents = nn.Parameter(torch.empty(1, num_latents, embedding_channels).normal_(std=0.02))
        self.scale_t = nn.Parameter(torch.zeros(1))
        self.scale_f = nn.Parameter(torch.zeros(1))
        self.scale_w = nn.Parameter(torch.zeros(1))

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

    def fusion(self, time, fre, wave):
        """
        :param time: (B, L1, C)
        :param fre: (B, L2, C)
        :param wave: (B, L3, C)
        :return:
        """

        b, _, _ = time.shape

        # (B, L1+L2+L3, C)
        concat = torch.concat([time, fre, wave], dim=1)

        fusion_latents = self.attention(q=self.latents.expand(b, -1, -1), k=concat, v=concat)

        time = time + self.scale_t * self.attention(q=time, k=fusion_latents, v=fusion_latents)
        fre = fre + self.scale_f * self.attention(q=fre, k=fusion_latents, v=fusion_latents)
        wave = wave + self.scale_w * self.attention(q=wave, k=fusion_latents, v=fusion_latents)

        return time, fre, wave

    def forward(self, time, fre, wave):
        """
        :param time:
        :param fre:
        :param wave:
        :return:
        """
        time, fre, wave = self.fusion(time, fre, wave)

        time = time + self.time_forward_layer(time)
        fre = fre + self.fre_forward_layer(fre)
        wave = wave + self.wave_forward_layer(wave)

        return time, fre, wave


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
        self.for_len = configs.for_len
        self.pose_num = configs.pose_num
        patch_len = configs.patch_len
        patch_stride = configs.patch_stride
        patch_padding = configs.patch_padding

        dropout = configs.dropout

        patch_num = int(np.floor((seg_len + patch_padding - patch_len) / patch_stride) + 1)

        graph = Graph(layout=configs.layout, strategy=configs.strategy, pose_num=self.pose_num, seg_len=patch_num)

        adj = graph.A

        adj = torch.tensor(adj, requires_grad=False)

        # adj = torch.ones((patch_num * self.pose_num, patch_num * self.pose_num))

        self.embedding_time = TimeEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                            patch_len=patch_len, patch_stride=patch_stride, patch_padding=patch_padding)

        self.embedding_fre = FreEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                          patch_len=patch_len, patch_stride=patch_stride, patch_padding=patch_padding)

        self.embedding_wave = WaveletEmbedding(in_channels=in_features, embedding_channels=embedding_channels,
                                               patch_len=patch_len, patch_stride=patch_stride, patch_padding=patch_padding)

        self.encoder_time = Encoder(
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

        self.encoder_fre = Encoder(
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

        fusion_layers = []
        for i in range(e_layers - fusion_layer):
            fusion_layers.append(
                FusionLayer(
                    embedding_channels=embedding_channels,
                    num_latents=16,
                    time_forward_layer=ChebyshevConv(
                        in_channels=embedding_channels,
                        out_channels=embedding_channels,
                        K=3,
                        adj=adj,
                        dropout=dropout),
                    fre_forward_layer=ChebyshevConv(
                        in_channels=embedding_channels,
                        out_channels=embedding_channels,
                        K=3,
                        adj=adj,
                        dropout=dropout),
                    wave_forward_layer=ChebyshevConv(
                        in_channels=embedding_channels,
                        out_channels=embedding_channels,
                        K=3,
                        adj=adj,
                        dropout=dropout)
                )
            )

        self.fusion_layers = nn.Sequential(*fusion_layers)

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


    def emb_time(self, x):
        """
        时域嵌入
        :param x: (B, C, T, V)
        :return:
        """
        # (B, C, T, V)-->(B, N, embedding_channels)
        emb = self.embedding_time(x)

        return emb

    def emb_fre(self, x):
        """
        频域嵌入
        :param x: (B, C, T, V)
        :return:
        """
        emb = self.embedding_fre(x)

        return emb

    def emb_wave(self, x):
        """
        小波域嵌入
        :param x: (B, C, T, V)
        :return:
        """
        emb = self.embedding_wave(x)

        return emb

    def forward_time(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        # 时域嵌入
        emb = self.emb_time(x)

        # 时域特征
        enc_out = self.encoder_time(emb)

        return enc_out

    def forward_fre(self, x):
        """

        :param x: (B, C, T, V)
        :return:
        """
        # 频域嵌入
        emb = self.emb_fre(x)
        enc_out = self.encoder_fre(emb)

        return enc_out

    def forward_wave(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        # 小波域嵌入
        emb = self.emb_wave(x)
        enc_out = self.encoder_wave(emb)

        return enc_out

    def forward_fusion(self, time, fre, wave):

        for layer in self.fusion_layers:
            time, fre, wave = layer(time, fre, wave)
        return time, fre, wave

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

        time = self.forward_time(x)
        fre = self.forward_fre(x)
        wave = self.forward_wave(x)

        time, fre, wave = self.forward_fusion(time, fre, wave)

        # (B, N, C)
        res = (time + fre + wave) * 0.5

        if self.task_name == 'rec':
            y = self.decode(res)
            y = y.reshape(b, v, t, -1).permute(0, 3, 2, 1).contiguous()

            return y

        if self.task_name == 'pre':
            y = self.decode(res)
            y = y.reshape(b, v, self.for_len, -1).permute(0, 3, 2, 1).contiguous()

            return y