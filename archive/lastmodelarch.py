import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size):
        super(CRNN, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1,
                                          kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Recurrent layers
        self.rnn = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)

        # Linear layer
        self.linear = nn.Linear(hidden_size, input_size)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # upsample 488 to 489
        self.upsample2 = nn.Upsample(size=(129, 489), mode='nearest')

        # Activation functions
        self.relu = nn.ReLU()

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(32)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        # Encoder
        x = self.conv1(x)
        # x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        # x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        # x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Recurrent layer
        # print(x.shape)
        # batch, channel, feature = 16 freqs bins * 61 time steps
        # batch = 2, channels = 128, features = 976
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)

        # Linear layer
        x = self.linear(x)
        x = self.relu(x)

        x = x.view(x.size(0), 128, 16, 61)
        # Decoder
        x = self.deconv1(x)
        # x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.deconv2(x)
        # x = self.batch_norm5(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.deconv3(x)
        x = self.relu(x)
        x = self.upsample2(x)

        return x


# %%
