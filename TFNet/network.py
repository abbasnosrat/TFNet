import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(conv_block, self).__init__()
        expansion = 4
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels *
                                                             expansion, stride=1, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels * expansion),
        )
        self.identity_downsample = identity_downsample
        self.relu = nn.ReLU()

    def forward(self, X):
        identity = X
        X = self.block(X)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        X += identity
        return self.relu(X)


class Resnet(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], input_channels=1, num_outputs=3):
        """
        One dimensional ResNet architecture.
        :param layers: number of blocks of each layer.
        :param input_channels: number of input channels.
        :param num_outputs: number of outputs
        """
        super(Resnet, self).__init__()

        self.in_channels = 64
        self.initial_block = nn.Sequential(nn.BatchNorm1d(1),
                                           nn.Conv1d(
                                               in_channels=input_channels, out_channels=64, kernel_size=7, stride=2,
                                               padding=3),
                                           nn.BatchNorm1d(64), nn.ReLU(),
                                           nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.layers = nn.ModuleList()
        a = 1
        for i in range(len(layers)):
            self.layers.append(self.make_layer(
                layers[i], out_channels=64 * (2 ** (i)), stride=a))
            a = 2
        self.dropout = nn.Dropout(0.4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64 * (2 ** (len(layers) + 1)), num_outputs)

    def make_layer(self, num_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv1d(in_channels=self.in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels * 4))
        layers.append(conv_block(self.in_channels, out_channels,
                                 identity_downsample, stride))
        self.in_channels = out_channels * 4
        for i in range(num_blocks - 1):
            layers.append(conv_block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.initial_block(X)
        for i in range(len(self.layers)):
            X = self.layers[i](X)
        X = self.dropout(X)
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = self.fc(X)
        return X
