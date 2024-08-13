# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Define the model

# Evaluation metrics will be : signal-to-distortion ratio improvement SDRi and signal-to-distortion ratio improvement.
# activate torch cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class SeparationModel(nn.Module):
    def __init__(self):
        super(SeparationModel, self).__init__()
        #
        # Define the model architecture
        #     It's a resunet model
        #     The input will be the mixture stft the encoder decoder will process the magnitude spectrogram
        #     3 encoder blocks 1 bottleneck blocks and 3 decoder blocks.

        #     In each encoder block, the spectrogram is downsampled into a bottleneck feature using 4 residual convolutional blocks,
        #     while each decoder block utilizes 4 residual deconvolutional blocks to upsample the feature and obtain the separation components.

        #     A skip connection is established between each encoder block and the corresponding decoder block, operating at the same downsampling/upsampling rate.

        #     The residual block consists of 2 CNN layers, 2 batch normalization layers, and 2 Leaky-ReLU activation layers.

        #     Furthermore, we introduce an additional residual shortcut connecting the input and output of each residual block.

        #     The ResUNet model inputs the complex spectrogram X and outputs the magnitude mask |M| and the phase residual ∠M.
        #     |M| controls how much the magnitude of |X| should be scaled, and the angle ∠M controls how much the angle of ∠X should be rotated.
        #     The separated complex spectrogram can be obtained by multiplying the STFT of the mixture and the predicted magnitude mask |M| and phase residual ∠X:
        #     Y = |M| ⊙ |X|exp(j(∠X + ∠M)), where ⊙ denotes Hadamard product.
        #

        self.encoder1 = nn.Sequential(
            # 1 channels for the magnitude and phase
            nn.Conv2d(1, 16, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # to get the mask
        )

    def forward(self, x):

        # print(f'input shape: {x.shape}')

        # add the channel dimension torch.size([16, 1, 129, 489])
        x = x.unsqueeze(1)

        # we want  torch.size([16, 2, 129, 489]) first channel is the magnitude and the second is the phase
        x = torch.cat((torch.abs(x), torch.angle(x)), dim=1)

        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Bottleneck
        b = self.bottleneck(x3)

        # Decoder
        d1 = self.decoder1(b)
        d2 = self.decoder2(d1)
        d3 = self.decoder3(d2)

        # Output
        out = self.output(d3)

        # resize the output to match the input
        out = F.interpolate(out, size=(129, 489),
                            mode='bilinear', align_corners=False)
        # print(f'out shape after interpolate: {out.shape}')
        return out


# calculate the nb of parameters
model = SeparationModel().to('cuda')
print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

# now using torchsummary
summary(model, (129, 489))
