#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 14:09:02 2025

@author: johannes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchedGaussianRenderer(nn.Module):
    def __init__(self, gaussians, sigma_mult=4, zoom=0.5):
        super().__init__()
        self.gaussians = gaussians
        self.sigma_mult = sigma_mult
        self.zoom = zoom
        self.image_height = gaussians.image_height
        self.image_width = gaussians.image_width

        y_coords = torch.arange(self.image_height)
        x_coords = torch.arange(self.image_width)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.register_buffer('pixel_grid', torch.stack([xx, yy], dim=-1).float())  # (H, W, 2)

        # Precompute and register static terms
        scale_mat = torch.tensor([
            [(self.image_width-1)/2 * self.zoom, 0, 0],
            [0, (self.image_height-1)/2 * self.zoom, 0],
        ], dtype=torch.float32)
        offset = torch.tensor([
            (self.image_width-1)/2,
            (self.image_height-1)/2
        ], dtype=torch.float32)
        self.register_buffer('scale_mat', scale_mat)
        self.register_buffer('offset', offset)

    def forward(self, t, angle):
        device = self.pixel_grid.device
        H, W = self.image_height, self.image_width
        N = self.gaussians.num_gaussians
    
        means = self.gaussians.means  # (N,4)
        raw_scales = self.gaussians.raw_scales
        rotors = self.gaussians.rotors
    
        scale = torch.exp(raw_scales)
    
        q1 = F.normalize(rotors[:, [0,4,5,6]], dim=1)
        q2 = F.normalize(rotors[:, [7,1,2,3]] * torch.tensor([1,-1,-1,-1], device=device), dim=1)
    
        L = self.gaussians._left_quat_matrix(q1)
        Rm = self.gaussians._right_quat_matrix_conj(q2)
        R4 = torch.bmm(L, Rm)
    
        S = torch.diag_embed(scale)
        cov4D = R4 @ S @ S.transpose(1,2) @ R4.transpose(1,2)
    
        U = cov4D[:, :3, :3]
        V = cov4D[:, :3, 3:]
        W_mat = cov4D[:, 3, 3].clamp(min=1e-6)
        lambda_t = 1.0 / W_mat
    
        mu_xyz = means[:, :3]
        mu_t = means[:, 3]
        time_diff = t - mu_t
        mu3D = mu_xyz + (V.squeeze(-1) * time_diff.unsqueeze(-1)) / W_mat.unsqueeze(-1)
    
        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)
        view = torch.tensor([
            [cos_a, 0, sin_a],
            [0,     1, 0    ],
            [-sin_a,0, cos_a],
        ], device=device)
    
        mu3D_view = (view @ mu3D.T).T
        cov3D = U - (V @ V.transpose(1,2)) / W_mat.view(-1,1,1)
        cov3D_view = torch.bmm(torch.bmm(view.expand(N, -1, -1), cov3D), view.expand(N, -1, -1).transpose(1,2))
    
        scale_mat = torch.tensor([
            [(W-1)/2 * self.zoom, 0, 0],
            [0, (H-1)/2 * self.zoom, 0],
        ], device=device)
        offset = torch.tensor([(W-1)/2, (H-1)/2], device=device)
    
        mu2D = (scale_mat @ mu3D_view.T).T + offset  # (N,2)
        cov2D = torch.bmm(torch.bmm(scale_mat.expand(N, -1, -1), cov3D_view), scale_mat.transpose(0,1).expand(N, -1, -1))
    
        weights = torch.exp(-0.5 * lambda_t * time_diff**2)
    
        # Flatten pixel grid: (H*W, 2)
        grid = self.pixel_grid.view(-1, 2)  # (P, 2), where P = H*W

        # Expand mu2D and cov2D for broadcasting
        mu2D = mu2D.unsqueeze(1)  # (N, 1, 2)
        grid = grid.unsqueeze(0)  # (1, P, 2)
        delta = grid - mu2D       # (N, P, 2)

        # Invert all covariance matrices at once
        cov2D = cov2D + torch.eye(2, device=device).unsqueeze(0) * 1e-6  # (N, 2, 2)
        inv_cov2D = torch.linalg.inv(cov2D)  # (N, 2, 2)

        # Compute Mahalanobis distance for all pixels and all Gaussians
        # einsum: (N, P, 2), (N, 2, 2), (N, P, 2) -> (N, P)
        expo = -0.5 * torch.einsum('npi, nij, npj -> np', delta, inv_cov2D, delta)

        # Compute Gaussian weights and sum over Gaussians
        patch = weights.view(N, 1) * torch.exp(expo)  # (N, P)
        image_flat = patch.sum(dim=0)  # (P,)

        image = image_flat.view(H, W)

        max_val = image.max().clamp(min=1e-6)
        image = image / max_val

        return image
