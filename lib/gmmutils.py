#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/16/2023 6:16 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : gmm_utils.py
# @Software: PyCharm
import torch

from lib.utils import gmm_params, square_distance


def knn(src, tgt, k, normalize=False):
    '''
    Find K-nearest neighbor when ref==ref_xyz and query==src_xyz
    Return index of knn, [B, N, k]
    '''
    dist = square_distance(src, tgt, normalize)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x.transpose(-1, -2), x.transpose(-1, -2), k=k)
        else:
            idx = knn(x[:, 6:].transpose(-1, -2), x[:, 6:].transpose(-1, -2), k=k)  # idx = knn(src_xyz[:, :3], k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2 * num_dims, num_points, k)


def og_params(pts, gamma, o_score=None, feature=None):
    if o_score is not None:
        # score [B, N]
        gamma_ex = (1.0 - o_score)
        # score [B, N, 1]
        gamma_ex = gamma_ex.unsqueeze(-1)
        # score [B, N, J]
        score = torch.cat([torch.einsum('bnk,bn->bnk', gamma, o_score), gamma_ex], dim=-1)
    else:
        score = gamma
    # mu: B x J x 3
    pi, mu = gmm_params(score, pts)
    if feature is not None:
        fea_mu = gmm_params(score, feature)[1]
        return pi, mu, fea_mu
    return pi, mu