#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:26:15 2025

@author: johannes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Gaussians(nn.Module):
    def __init__(self, dataset_path, num_gaussians, zoom=0.5):
        super().__init__()
        from data_loader import TimeAngleImageDataset

        dataset = TimeAngleImageDataset(dataset_path)
        data_shape = dataset.data.shape  # (T, A, H, W)
        self.image_height = data_shape[2]
        self.image_width = data_shape[3]
        self.zoom = zoom
        self.num_gaussians = num_gaussians

        max_time_entry = dataset.__getitem__((dataset.time_steps - 1) * dataset.angle_steps)
        max_time = max_time_entry['input'][0].item()

        max_spatial_dim = max(self.image_height, self.image_width)
        bounding_box = torch.tensor([
            self.image_width / max_spatial_dim,
            self.image_height / max_spatial_dim,
            self.image_width / max_spatial_dim,
            max_time
        ], dtype=torch.float32)

        low = torch.zeros(4)
        high = bounding_box
        initial_pos = torch.rand(num_gaussians, 4) * (high - low) + low

        self.means = nn.Parameter(initial_pos)  # (N, 4)
        self.raw_scales = nn.Parameter(torch.empty(num_gaussians, 4).uniform_(0.1, 0.3).log())  # (N, 4)
        rotors = torch.empty(num_gaussians, 8).uniform_(0.5, 1.0)
        rotors = rotors / rotors.norm(dim=1, keepdim=True)
        self.rotors = nn.Parameter(rotors)  # (N, 8)

    def _left_quat_matrix(self, q):
        a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([
            torch.stack([ a, -b, -c, -d], dim=1),
            torch.stack([ b,  a, -d,  c], dim=1),
            torch.stack([ c,  d,  a, -b], dim=1),
            torch.stack([ d, -c,  b,  a], dim=1),
        ], dim=1)  # (N, 4, 4)

    def _right_quat_matrix_conj(self, q):
        e, f, g, h = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.stack([
            torch.stack([ e,  f,  g,  h], dim=1),
            torch.stack([-f,  e, -h,  g], dim=1),
            torch.stack([-g,  h,  e, -f], dim=1),
            torch.stack([-h, -g,  f,  e], dim=1),
        ], dim=1)  # (N, 4, 4)
