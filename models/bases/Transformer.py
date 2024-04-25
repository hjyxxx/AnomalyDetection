from torch import nn

from models.components.attention import FullAttention
from models.components.embedding import ConvEmbedding, PositionEmbedding
from models.layers.attention_layer import AttentionLayer


class Encoder(nn.Module):
    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, x):
        """
        :param x: ()
        :return:
        """
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, in_channels, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention_layer = attention_layer

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=in_channels, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.activate = nn.ReLU()

    def forward(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """
        new_x, attn = self.attention_layer(x, x, x)

        x = x + self.dropout(new_x)

        y = x

        y = self.dropout(self.activate(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return x + y


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        e_layers = configs.e_layers

        in_features = configs.in_features
        out_features = configs.out_features
        embedding_channels = configs.embedding_channels
        d_ff = configs.d_ff
        dropout = configs.dropout

        pose_num = configs.pose_num

        self.conv_embedding = ConvEmbedding(in_channels=in_features * pose_num, embedding_channels=embedding_channels * pose_num)
        self.position_embedding = PositionEmbedding(embedding_channels=embedding_channels * pose_num)

        self.encoder = Encoder(
            encoder_layers=[
                EncoderLayer(
                    attention_layer=AttentionLayer(
                        attention=FullAttention(),
                        in_channels=embedding_channels * pose_num,
                        n_heads=8
                    ),
                    in_channels=embedding_channels * pose_num,
                    d_ff=d_ff * pose_num,
                    dropout=dropout
                ) for _ in range(e_layers)
            ]
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_channels * pose_num, out_features=out_features * pose_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=out_features * pose_num, out_features=out_features * pose_num)
        )

    def emb(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """

        x = self.conv_embedding(x)
        y = self.position_embedding(x)

        return y

    def encode(self, x):
        """
        :param x: (B, T, VC)
        :return:
        """
        y = self.encoder(x)

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
        :param x:
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


