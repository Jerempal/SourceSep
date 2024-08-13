import torchsummary as summary
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%


class ResUNetSeparate(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(ResUNetSeparate, self).__init__()
        self.output = nn.Conv2d(base_channels, 3, kernel_size=1, padding=0)

    def forward(self, x):
        output = self.output(x)
        mag_mask = torch.sigmoid(output[:, 0, :, :])
        real_mask = torch.tanh(output[:, 1, :, :])
        imag_mask = torch.tanh(output[:, 2, :, :])

        output_masks_dict = {
            'mag_mask': mag_mask,
            'real_mask': real_mask,
            'imag_mask': imag_mask
        }

        return output_masks_dict


class ResUNetCombined(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(ResUNetCombined, self).__init__()
        self.mag_mask_layer = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.real_mask_layer = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )
        self.imag_mask_layer = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        mag_mask = self.mag_mask_layer(x)
        real_mask = self.real_mask_layer(x)
        imag_mask = self.imag_mask_layer(x)

        output_masks_dict = {
            'mag_mask': mag_mask,
            'real_mask': real_mask,
            'imag_mask': imag_mask
        }

        return output_masks_dict


# Model initialization
model_separate = ResUNetSeparate(1, 64).to('cuda')
model_combined = ResUNetCombined(1, 64).to('cuda')

# %%

summary.summary(model_separate, (64, 129, 489), batch_size=16, device='cuda')
# %%
summary.summary(model_combined, (64, 129, 489), batch_size=16, device='cuda')
# %%
