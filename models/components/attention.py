from math import sqrt

import torch
from torch import nn

from models.components.masking import TriangularCausalMask


class DynamicAttention(nn.Module):
    def __init__(self, in_channels, out_channels, mask_flag=False, dropout=0.1):
        """
        动态注意力
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param mask_flag: 掩码标志
        :param dropout: 丢失率
        """
        super(DynamicAttention, self).__init__()

        self.mask_flag = mask_flag

        self.linear_l = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.linear_r = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        self.a = nn.Parameter(torch.FloatTensor(out_channels, 1))

        self.activate = nn.LeakyReLU()

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        """
        h_0 || h_0
        h_1 || h_0
        h_2 || h_0
            ...
        h_0 || h_1
            ...
        :param queries: (B, L, C), B: 批次大小, L: 计算注意力的节点数, C: 节点特征
        :param keys:
        :param values:
        :param mask:
        :return:
        """
        b, l, c = queries.shape

        # Wh_i, (B, L, C)(C, C')-->(B, L, C')
        g_l = torch.einsum('blc,cd->bld', queries, self.linear_l)

        # Wh_j, (B, L, C)(C, C')-->(B, L, C')
        g_r = torch.einsum('blc,cd->bld', queries, self.linear_r)

        # Wh_i, (B, L, C')-->(B, L*L, C')
        g_l_repeat = g_l.repeat(1, l, 1)

        # Wh_j, (B, L, C')-->(B, L*L, C')
        g_r_repeat_interleave = g_r.repeat_interleave(l, dim=-2)

        # Wh_i || Wh_j
        g_sum = g_l_repeat + g_r_repeat_interleave

        # (B, L*L, C')-->(B, L, L, C')
        g_sum = g_sum.view(b, l, l, -1)

        g_sum = self.activate(g_sum)

        # (B, L, L, C')(C', 1)-->(B, L, L, 1)
        e = torch.einsum('bxyd,de->bxye', g_sum, self.a)

        # (B, L, L)
        e = e.squeeze(-1)

        if self.mask_flag:
            if mask is None:
                # 下三角矩阵
                mask = torch.tril(torch.ones(l, l), diagonal=0).to(device=queries.device)

            e = e.masked_fill(mask == 0, float('-inf'))

        scores = self.dropout(self.softmax(e))

        # (B, L, L)(B, L, C')-->(B, L, C')
        y = torch.matmul(scores, g_r)

        return y, scores


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, dropout=0.1):
        super(FullAttention, self).__init__()

        self.mask_flag = mask_flag
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries: (B, L, H, C), B: 批次大小, L: 计算注意力的节点, H: 多头头数, C: d_keys
        :param keys: (B, S, H, C), B: 批次大小, S: 计算注意力的节点, H: 多头头数, C: d_keys
        :param values: (B, S, H, D), B: 批次大小, S: 计算注意力的节点, H: 多头头数, D: d_values
        :param mask:
        :return: (B, L, H, D)
                 (B, L, S)
        """
        B, L, H, C = queries.shape
        _, S, _, D = keys.shape

        scale = 1. / sqrt(C)

        # (B, L, H, C)-->(B, H, L, C)
        queries = queries.transpose(-2, -3).contiguous()

        # (B, S, H, C)-->(B, H, S, C)
        keys = keys.transpose(-2, -3).contiguous()

        # (B, S, H, D)-->(B, H, S, D)
        values = values.transpose(-2, -3).contiguous()

        # (B, H, L, C)(B, H, C, S)-->(B, H, L, S)
        scores = torch.matmul(queries, keys.transpose(-1, -2))

        if self.mask_flag:
            if mask is None:
                mask = TriangularCausalMask(L, S, device=queries.device)

            scores.masked_fill_(mask.mask, float('-inf'))

        # (B, H, L, S)
        A = self.dropout(self.softmax(scores * scale))

        # (B, H, L, S)(B, H, S, D)-->(B, H, L, D)
        V = torch.matmul(A, values)

        # (B, H, L, D)-->(B, L, H, D)
        V = V.permute(0, 2, 1, 3).contiguous()

        return V, A


if __name__ == '__main__':

    x = torch.randn(size=(128, 17, 8, 256))

    model = FullAttention(mask_flag=True)

    y, attn = model(x, x, x)
    print(y.shape)
    print(attn.shape)