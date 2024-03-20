import torch
from torch import nn

from model_utils.utils import rescaled_L


class GraphAttentionConv(nn.Module):
    def __init__(self, adj, in_channels, out_channels, n_heads, is_concat=True, dropout=0.1):
        super(GraphAttentionConv, self).__init__()

        adj = adj.unsqueeze(-1)
        self.register_buffer('adj', adj)

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_channels % n_heads == 0
            self.n_hidden = out_channels // n_heads

        else:
            self.n_hidden = out_channels

        self.linear = nn.Linear(in_features=in_channels, out_features=self.n_hidden * self.n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)

        self.activate = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: (B, N, C), B: 批次大小, N: 节点个数, C: 特征维度
        :return:
        """

        b, n_nodes = x.shape[0], x.shape[1]

        # (B, N, C)-->(B, N, heads, n_hidden)
        g = self.linear(x).view(b, n_nodes, self.n_heads, self.n_hidden)

        # (B, N, heads, n_hidden)-->(B, N*N, heads, n_hidden)
        g_repeat = g.repeat(1, n_nodes, 1, 1)

        # (B, N, heads, n_hidden)-->(B, N*N, heads, n_hidden)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)

        # (B, N*N, heads, n_hidden)-->(B, N*N, heads, 2*n_hidden)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)

        # (B, N*N, heads, 2*n_hidden)-->(B, N, N, heads, 2 * n_hidden)
        g_concat = g_concat.view(b, n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        # (B, N, N, heads, 2 * n_hidden)-->(B, N, N, heads, 1)
        e = self.activate(self.attn(g_concat))

        # (B, N, N, heads, 1)-->(B, N, N, heads)
        e = e.squeeze(-1)

        assert self.adj.shape[0] == 1 or self.adj.shape[0] == n_nodes
        assert self.adj.shape[1] == 1 or self.adj.shape[1] == n_nodes
        assert self.adj.shape[2] == 1 or self.adj.shape[2] == self.n_heads

        e = e.masked_fill(self.adj == 0, float('-inf'))

        a = self.softmax(e)

        a = self.dropout(a)

        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)

        if self.is_concat:
            return attn_res.reshape(b, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=-2)


class ChebyshevConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, adj, dropout=0.1):
        super(ChebyshevConv, self).__init__()

        L = torch.diag(torch.sum(adj, dim=-1)) - adj
        L_tilde = rescaled_L(L)

        L_tilde = L_tilde.float()

        self.register_buffer('adj', adj)
        self.register_buffer('L_tilde', L_tilde)

        self.K = K
        self.out_channels = out_channels

        self.theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])

        self.dropout = nn.Dropout(dropout)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        """
        :param x: (B, N, C), B: 批次大小, N: 节点个数, C: 特征维度
        :return:
        """
        b, n, c = x.shape

        T0 = torch.eye(n, n).to(device=x.device)

        out = torch.zeros(b, n, self.out_channels).to(device=x.device)

        # (N, N)(B, N, C)(C, C')-->(B, N, C')
        out = out + torch.matmul(T0, x).matmul(self.theta[0])

        if self.K > 1:
            T1 = self.L_tilde

            out = out + torch.matmul(T1, x).matmul(self.theta[1])

        for k in range(2, self.K):
            T2 = 2 * self.L_tilde.matmul(T1) - T0
            out = out + torch.matmul(T2, x).matmul(self.theta[k])
            T0, T1 = T1, T2

        out = self.dropout(out)

        return self.activate(out)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super(GraphConv, self).__init__()

    def forward(self, x):
        """

        :param x:
        :return:
        """


if __name__ == '__main__':

    torch.random.manual_seed(2024)

    adj = torch.ones(size=(180, 180))

    model = ChebyshevConv(2, 64, 3, adj)

    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)

    for name, param in model.named_parameters():
        print(name, param.data)

    x = torch.randn(size=(128, 180, 2))

    y = model(x)
    print(y)
