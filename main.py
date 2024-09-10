# %%
from model.model import SepModel
from data.config import *
from data.dataset import PreComputedMixtureDataset
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from train_test import train_model, test_model, plot_losses_and_metrics
import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
#%%
# Load metadata
metadata = pd.read_csv(os.path.join(
    DATASET_MIX_AUDIO_PATH, "metadata.csv"))

dataset = PreComputedMixtureDataset(metadata_file=metadata)

# load data
train_indices = np.load('train_indices_new_last.npy')
val_indices = np.load('val_indices_new_last.npy')
test_indices = np.load('test_indices_new_last.npy')

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# load laoder again
train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=32)
val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=32)
test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
model = SepModel(in_c=1, out_c=32).to(device)
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
# optimizer = AdamW(model.parameters(), lr=1e-3)
optimizer = AdamW(model.parameters(), lr=1e-3, amsgrad=True, fused=True)

# Train model
train_losses, val_losses, val_sdr, val_si_sdr = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=100,
    device='cuda',
    checkpoint_dir="checkpoint/"
)

# plot losses and metrics
plot_losses_and_metrics(
    train_losses=train_losses,
    val_losses=val_losses,
    val_sdr=val_sdr,
    val_si_sdr=val_si_sdr
)
# on charge le meilleur model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SepModel(in_c=1, out_c=32).to(device)
model.load_state_dict(torch.load('checkpoint\\best_model.pth',
                      map_location=device, weights_only=True))

model.eval()
# Test model
test_model(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device='cuda'
)

#%%

# load checkpoint with optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SepModel(in_c=1, out_c=32).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, amsgrad=True, fused=True)
from data.utils import load_checkpoint
model, optimizer, epoch, train_loss, val_loss = load_checkpoint(model, optimizer, checkpoint_dir="checkpoint", filename="checkpoint_last_epoch_8.pth")

# see lr
for param_group in optimizer.param_groups:
    print(param_group['lr'])
#%%
# do this for epoch 0 to 16
lr = []
for epoch in range(0, 17):
    model, optimizer, epoch, train_loss, val_loss = load_checkpoint(model, optimizer, checkpoint_dir="checkpoint", filename=f"checkpoint_last_epoch_{epoch}.pth")
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
        
# plot lr
import matplotlib.pyplot as plt
plt.plot(lr)
plt.show()


    
# %%

