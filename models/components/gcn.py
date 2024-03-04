import torch
from torch import nn

from model_utils.utils import rescaled_L


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
        :param x: (B, L, C), B: 批次大小, L: 样本个数, C: 特征维度
        :return:
        """
        b, l, c = x.shape

        T0 = torch.eye(l, l).to(device=x.device)

        out = torch.zeros(b, l, self.out_channels).to(device=x.device)

        # (L, L)(B, L, C)(C, C')-->(B, L, C')
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

if __name__ == '__main__':

    adj = torch.ones(size=(180, 180))

    model = ChebyshevConv(2, 64, 3, adj)

    x = torch.randn(size=(128, 180, 2))

    y = model(x)
