import torch
from torch import nn

from models.components.attention import FullAttention


class AttentionLayer(nn.Module):
    def __init__(self, attention, in_channels, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (in_channels // n_heads)
        d_values = d_values or (in_channels // n_heads)

        self.n_heads = n_heads

        self.inner_attention = attention

        self.query_projection = nn.Linear(in_channels, n_heads * d_keys)
        self.key_projection = nn.Linear(in_channels, n_heads * d_keys)
        self.value_projection = nn.Linear(in_channels, n_heads * d_values)

        self.out_projection = nn.Linear(n_heads * d_values, in_channels)

    def forward(self, queries, keys, values, attn_mask=None):
        """
        :param queries: (B, L, C), B: 样本个数, L: 样本长度, C: 特征维度
        :param keys: (B, S, C), B: 样本个数, S: 样本长度, C: 特征维度
        :param values: (B, S, C), B: 样本个数, S: 样本长度, C: 特征维度
        :param attn_mask:
        :return: (B, L, C)
                 (B, H, L, S)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        # (B, L, C)-->(B, L, H, d_keys)
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        # (B, S, C)-->(B, S, H, d_keys)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        # (B, S, C)-->(B, S, H, d_values)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)

        # (B, L, H, D), (B, H, L, S)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)

        # (B, L, HD)
        out = out.view(B, L, -1)

        # (B, L, C)
        out = self.out_projection(out)

        return out, attn


class GCNLayer(nn.Module):
    def __init__(self, conv_layer, ):
        super(GCNLayer, self).__init__()
        self.conv_layer = conv_layer

    def forward(self, x):
        """
        :param x: (B, L, C)
        :return:
        """

        y = self.conv_layer(x)

        return y


if __name__ == '__main__':

    x = torch.randn(size=(128, 17, 128))

    model = AttentionLayer(attention=FullAttention(mask_flag=False), in_channels=128, n_heads=8)

    y = model(x, x, x)