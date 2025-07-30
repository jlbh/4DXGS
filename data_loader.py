#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:04:12 2025

@author: johannes
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class TimeAngleImageDataset(Dataset):
    def __init__(self, npy_path):
        
        # Memory-map mode
        self.data = np.load(npy_path, mmap_mode='r') # shape: (199, 399, 70, 528)
        self.time_steps = self.data.shape[0]
        self.angle_steps = self.data.shape[1]

        # Create all (time_idx, angle_idx) combinations
        self.indices = [
            (t_idx, a_idx)
            for t_idx in range(self.time_steps)
            for a_idx in range(self.angle_steps)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx, a_idx = self.indices[idx]

        # Normalize time to [0, 1], angle to [0, 2π]
        time = t_idx / (self.time_steps - 1)
        angle = (a_idx / (self.angle_steps - 1)) * (2 * np.pi)

        # Convert inputs to tensor
        input_tensor = torch.tensor([time, angle], dtype=torch.float32)

        # Get the image target and convert to tensor
        image = self.data[t_idx, a_idx]  # shape: (70, 528)
        target_tensor = torch.tensor(image, dtype=torch.float32)

        return {
            'input': input_tensor,
            'target': target_tensor
        }