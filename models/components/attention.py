from math import sqrt

import torch
from torch import nn

from models.components.masking import TriangularCausalMask


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