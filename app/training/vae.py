"""ChartVAE — 谱面稠密张量 (3,T) 的卷积 VAE (B 路 / MuG-Diffusion 风格)。

设计要点(针对此前"后验塌缩"的修复):
  - β 默认极小(1e-4), 对标 MuG 的 kl_weight≈1e-6, 防止 KL 把潜变量 z 压没。
  - free_bits=0.5: 每个潜变量元素的 KL 有下限, 解码器不能丢弃 z。
  - 编码器 4 级下采样 (T -> T/16), 解码器用 Upsample+Conv 避免长度错位。
输入/输出: (B, 3, T), 值域大致 [0,1] (onset/twirl 热图, dir 0/1)。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH = 3
BASE = 32
LEVELS = 4          # 下采样级数 -> 潜空间帧率 T/16
Z_CH = 16


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c = nn.Conv1d(cin, cout, 7, stride=2, padding=3)
        self.n = nn.GroupNorm(8, cout)
        self.a = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.n(self.a(self.c(x)))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.u = nn.Upsample(scale_factor=2, mode="nearest")
        self.c = nn.Conv1d(cin, cout, 7, padding=3)
        groups = 8 if cout % 8 == 0 else 1
        self.n = nn.GroupNorm(groups, cout)
        self.a = nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.n(self.a(self.c(self.u(x))))


class ChartVAE(nn.Module):
    def __init__(self, in_ch=IN_CH, base=BASE, levels=LEVELS, z_ch=Z_CH):
        super().__init__()
        self.levels = levels
        chs = [in_ch] + [base * (2 ** i) for i in range(levels)]   # 3,32,64,128,256
        self.enc = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(levels)])
        self.mu = nn.Conv1d(chs[-1], z_ch, 3, padding=1)
        self.logvar = nn.Conv1d(chs[-1], z_ch, 3, padding=1)
        self.zup = nn.Conv1d(z_ch, chs[-1], 3, padding=1)
        # 解码器镜像编码器, 末层回到 in_ch(3): 256 ->128 ->64 ->32 ->3
        dec_chs = [chs[-1]] + [chs[levels - 1 - i] for i in range(levels)]
        self.dec = nn.ModuleList([Up(dec_chs[i], dec_chs[i + 1]) for i in range(levels)])

    def encode(self, x):
        h = x
        for b in self.enc:
            h = b(h)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        h = self.zup(z)
        for b in self.dec:
            h = b(h)
        return h

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv

    @staticmethod
    def loss(recon, x, mu, logvar, beta=1e-4, free_bits=0.5):
        recon_l = F.smooth_l1_loss(recon, x)
        # 每元素 KL (nats)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if free_bits and free_bits > 0:
            kl = torch.clamp(kl, min=free_bits)   # 下限, 防塌缩
        return recon_l + beta * kl.mean(), recon_l.detach(), kl.mean().detach()
