#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:10:20 2025

@author: johannes
"""
import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from gaussian_model import Gaussians
from gaussian_renderer import BatchedGaussianRenderer
from data_loader import TimeAngleImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
dataset_path = 'am_all_bin3_l.npy'
num_gaussians = 256
num_epochs = 1
learning_rate = 1e-3

# Dataset and dataloader
dataset = TimeAngleImageDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # no batch support in renderer

# Initialize model and renderer
gaussians = Gaussians(dataset_path, num_gaussians).to(device)
renderer = BatchedGaussianRenderer(gaussians).to(device)

optimizer = optim.Adam(gaussians.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

for epoch in range(num_epochs):
    renderer.train()  # just to be explicit
    total_loss = 0.0

    for batch in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = batch['input'].to(device)  # (1, 2): time and angle
        target = batch['target'].to(device) # (1, H, W)

        # Extract scalars for t and angle
        t = inputs[0, 0].float()
        angle = inputs[0, 1].float()

        optimizer.zero_grad()
        output = renderer(t, angle)  # (H, W)

        loss = loss_fn(output, target.squeeze(0))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
