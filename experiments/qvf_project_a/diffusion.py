"""Classical DDPM on normalized image codes, with cosine noise and v prediction.

The posterior coefficients are those of DDPM. Velocity parameterization avoids
division by a tiny signal coefficient at the end of the cosine schedule.
References: arxiv.org/abs/2006.11239, 2102.09672, 2202.00512.
"""

import math

import torch
from torch import nn


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, width=256, layers=4, time_dim=64, timesteps=100):
        super().__init__()
        if width < 1 or layers < 1 or time_dim < 2 or time_dim % 2 or timesteps < 2:
            raise ValueError("Invalid denoiser dimensions or timesteps.")
        self.timesteps = timesteps
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(time_dim // 2) / (time_dim // 2)))
        self.time = nn.Sequential(nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.input = nn.Linear(latent_dim, width)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
            for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, latent_dim))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, noisy_code, timesteps):
        phases = timesteps.float()[:, None] * (1000.0 / self.timesteps) * self.frequencies[None]
        time = self.time(torch.cat((phases.cos(), phases.sin()), dim=-1))
        hidden = self.input(noisy_code)
        for block in self.blocks:
            hidden = hidden + block(hidden + time)
        return self.output(hidden)


class LatentDDPM(nn.Module):
    def __init__(self, timesteps=100):
        super().__init__()
        if timesteps < 2:
            raise ValueError("DDPM requires at least two timesteps.")
        self.timesteps = timesteps
        times = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        abar = alphas.cumprod(dim=0)
        previous = torch.cat((torch.ones(1, dtype=torch.float64), abar[:-1]))
        buffers = {
            "betas": betas,
            "signal": abar.sqrt(),
            "noise": (1 - abar).sqrt(),
            "posterior_variance": betas * (1 - previous) / (1 - abar),
            "posterior_clean": betas * previous.sqrt() / (1 - abar),
            "posterior_noisy": (1 - previous) * alphas.sqrt() / (1 - abar),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.float())

    def q_sample(self, clean, timesteps, noise):
        return self.signal[timesteps, None] * clean + self.noise[timesteps, None] * noise

    def target(self, clean, timesteps, noise):
        return self.signal[timesteps, None] * noise - self.noise[timesteps, None] * clean

    def clean_from_velocity(self, noisy, timesteps, velocity):
        return self.signal[timesteps, None] * noisy - self.noise[timesteps, None] * velocity

    def reverse_step(self, noisy, timesteps, velocity, noise):
        clean = self.clean_from_velocity(noisy, timesteps, velocity)
        mean = self.posterior_clean[timesteps, None] * clean + self.posterior_noisy[timesteps, None] * noisy
        return mean + (timesteps > 0)[:, None] * self.posterior_variance[timesteps, None].sqrt() * noise

    @torch.no_grad()
    def sample(self, denoiser, count, latent_dim, generator):
        denoiser.eval()
        device = self.signal.device
        code = torch.randn((count, latent_dim), device=device, generator=generator)
        for step in reversed(range(self.timesteps)):
            timesteps = torch.full((count,), step, device=device, dtype=torch.long)
            velocity = denoiser(code, timesteps)
            noise = torch.randn(code.shape, device=device, generator=generator) if step else torch.zeros_like(code)
            code = self.reverse_step(code, timesteps, velocity, noise)
        if not torch.isfinite(code).all():
            raise FloatingPointError("Non-finite generated latent codes")
        return code
