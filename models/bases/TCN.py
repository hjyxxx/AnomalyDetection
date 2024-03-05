from torch import nn

from models.components.embedding import ConvEmbedding


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        :param x: (B, C, T)
        :return:
        """

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: (B, C, T)
        :return:
        """

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: (B, C, T)
        :return:
        """
        return self.network(x)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        e_layers = configs.e_layers

        in_features = configs.in_features
        out_features = configs.out_features
        embedding_channels = configs.embedding_channels
        d_ff = configs.d_ff
        dropout = configs.dropout

        node_num = configs.node_num

        self.embedding = ConvEmbedding(in_channels=in_features * node_num, embedding_channels=embedding_channels * node_num)

        num_inputs = embedding_channels * node_num
        num_channels = [embedding_channels * node_num, d_ff * node_num, d_ff * node_num, embedding_channels * node_num]
        num_outputs = out_features * node_num

        self.encoder = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, dropout=dropout)

        self.decoder = nn.Sequential(
            nn.Linear(num_inputs, num_outputs),
            # nn.LeakyReLU(),
            # nn.Linear(num_outputs, num_outputs)
        )

    def emb(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """
        y = self.embedding(x)
        return y

    def encode(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """
        y = self.encoder(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y

    def decode(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """
        y = self.decoder(x)

        return y

    def forward(self, x):
        """
        :param x: (B, C, T, V)
        :return:
        """
        b, c, t, v = x.shape
        # (B, C, T, V)-->(B, T, VC)
        new_x = x.permute(0, 2, 3, 1).contiguous().reshape(b, t, -1)

        x_emb = self.emb(new_x)

        x_enc = self.encode(x_emb)

        x_dec = self.decode(x_enc)

        y = x_dec.reshape(b, t, v, -1).permute(0, 3, 1, 2).contiguous()

        return y

