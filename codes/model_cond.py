
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import tqdm
import logging
import math
from os import path
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader

import time


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).cuda()


def normalize_to_neg_one_to_one(emb):
    return emb * 2 - 1


def exists(x):   ############
    return x is not None

class Residual(nn.Module):  ##########
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module): ##################
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):   #################
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# class SinusoidalPosEmb(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#
#     def forward(self, x):
#         half_dim = (self.dim // 2) + 1
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim) * -emb).cuda()
#         emb = x[:, None].cuda() * emb[None, :]
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb[:, :self.dim]
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.emb_cache = None

    def forward(self, x):
        if self.emb_cache is None:
            half_dim = (self.dim // 2) + 1
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim) * -emb).cuda()
            self.emb_cache = emb

        emb = x[:, None].cuda() * self.emb_cache[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb[:, :self.dim]

class Block(nn.Module):   #ddpm
    def __init__(self, in_ft, out_ft) -> None:
        super(Block, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.SiLU(),
            nn.Linear(out_ft, out_ft)
        )
        # self.norm = RMSNorm(out_ft)
        self.norm = nn.LayerNorm(out_ft)  #
        # self.norm = nn.BatchNorm1d(out_ft)  #
        self.act = nn.SiLU()
        # self.dropout = nn.Dropout(0.1)

        # self.lin = nn.Linear(in_ft, out_ft)
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_ft, out_ft * 2)
        )

    def forward(self, h, t):

        h = self.norm(h)
        h = self.proj(h)
        t = self.time(t)

        scale, shift = t.chunk(2, dim=1)
        h = (scale + 1) * h + shift

        h = self.act(h)

        return h

class Encoder(nn.Module):
    def __init__(self, in_ft, out_ft,  y=None) -> None:
        super(Encoder, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(out_ft, out_ft),
        #     nn.SiLU(),
        #     nn.Linear(out_ft, out_ft)
        # )

        self.l1 = Block(in_ft, out_ft)
        self.l2 = Block(out_ft, out_ft)

        self.\
            res_conv = nn.Sequential(
            nn.Linear(out_ft, out_ft),
            nn.SiLU(),
            nn.Linear(out_ft, out_ft))
        t = SinusoidalPosEmb(out_ft)
        self.time_mlp = nn.Sequential(
                t,
                nn.Linear(out_ft, out_ft),
                nn.SiLU(),
                nn.Linear(out_ft, out_ft)
         )

    def forward(self, h, t, y):
        t = self.time_mlp(t)
        if y is not None:
            t += y
        h = self.l1(h, t)
        h = self.l2(h, t)

        return h + self.res_conv(h)



class Diffusion_Cond(nn.Module):
    def __init__(self, in_feat, out_feat, args, y) -> None:
        super(Diffusion_Cond, self).__init__()


        self.encoder = Encoder(in_feat, out_feat, y)

        self.timesteps = args.timesteps  # 200 #50

        # define beta schedule

        self.betas = linear_beta_schedule(timesteps=self.timesteps)
        # self.betas = cosine_beta_schedule(timesteps=self.timesteps)


        # define alphas
        self.alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # forward diffusion (using the nice property)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, labels, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)


        predicted_noise = self.encoder(x_noisy, t, labels)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    def p_sample(self, model, x, t, labels, t_index, cfg_scale=0):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        predicted_noise = model(x, t, labels)

        if cfg_scale > 0:
            uncond_predicted_noise = model(x, t, None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    def p_sample_loop(self, model, shape, y):
        # device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        emb = torch.randn(shape).cuda()
        embs = []

        # for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
        for i in reversed(range(0, self.timesteps)):
            emb = self.p_sample(model, emb, torch.full((b,), i, dtype=torch.long), y, i)
            embs.append(emb)

        embs = [F.normalize(e, dim=-1) for e in embs]
        embs = embs[::-1]
        steps = list(range(1, 50, 2))[:20]

        out = [embs[step] for step in steps]
        return out

    @torch.no_grad()
    def sample(self, shape, y):
        return self.p_sample_loop(self.encoder, shape, y)

    def forward(self, input, labels):
        t = torch.randint(0, self.timesteps, (input.shape[0],)).long().cuda()
        return self.p_losses(input, t, labels)