"""DDPM 在 VAE 潜空间去噪 (B 路 / MuG-Diffusion 风格)。

条件 = 局部 onset 包络(1ch) + MelCondEncoder(梅尔谱 128ch -> 8ch) 拼接 = 9ch,
在 UNet 每个分辨率 concat 进潜变量。含 Classifier-Free Guidance。

关键修复 (v2): UNet **必须有时间步嵌入**。上一版 forward 收了 t 却没用,
导致 DDPM 采样在中段(ti≈300)激活爆炸成 NaN -> 风格层(Twirl)全 0。
现用正弦时间嵌入注入每一层 ResBlock 的条件中, 并加采样期 clamp 防溢出。
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128
COND_CH = 9          # 8 (mel) + 1 (onset env)
TIME_DIM = 32        # 时间步嵌入维度(注入条件)
P_UNCOND = 0.15
TIMESTEPS = 1000


class TimeEmbed(nn.Module):
    """正弦位置嵌入 + 小 MLP -> (B, TIME_DIM)。"""
    def __init__(self, dim=TIME_DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -math.log(10000.0) / (half - 1), half,
                                         device=t.device))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)      # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)                                     # (B, dim)


class MelCondEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, out_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(16, out_ch, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, mel):
        return self.net(mel)        # (B,8,T/16)


class ResBlock(nn.Module):
    def __init__(self, ch, cond_ch=COND_CH + TIME_DIM):
        super().__init__()
        self.c1 = nn.Conv1d(ch + cond_ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch + cond_ch, ch, 3, padding=1)
        self.act = nn.ReLU()
    def forward(self, x, cond):
        if cond.shape[-1] != x.shape[-1]:
            cond = F.interpolate(cond, size=x.shape[-1], mode="nearest")
        xc = torch.cat([x, cond], dim=1)
        h = self.act(self.c1(xc))
        hc = torch.cat([h, cond], dim=1)
        return x + self.c2(hc)


class UNet1D(nn.Module):
    def __init__(self, z_ch=16, cond_ch=COND_CH, time_dim=TIME_DIM):
        super().__init__()
        self.time_emb = TimeEmbed(time_dim)
        self.cond_ch = cond_ch + time_dim
        self.inp = nn.Conv1d(z_ch, 64, 3, padding=1)
        self.down = nn.ModuleList([
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.Conv1d(256, 512, 3, stride=2, padding=1),
            nn.Conv1d(512, 512, 3, stride=2, padding=1),
        ])
        self.mid = ResBlock(512, self.cond_ch)
        self.up = nn.ModuleList([
            nn.ConvTranspose1d(512 + 512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(256 + 512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(128 + 256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(64 + 128, 64, 3, stride=2, padding=1, output_padding=1),
        ])
        self.rb_down = nn.ModuleList([ResBlock(128, self.cond_ch), ResBlock(256, self.cond_ch),
                                      ResBlock(512, self.cond_ch), ResBlock(512, self.cond_ch)])
        self.rb_up = nn.ModuleList([ResBlock(256, self.cond_ch), ResBlock(128, self.cond_ch),
                                    ResBlock(64, self.cond_ch), ResBlock(64, self.cond_ch)])
        self.out = nn.Conv1d(64, z_ch, 3, padding=1)
        self.act = nn.ReLU()

    def _aug_cond(self, cond, t, Tz):
        # cond: (B, COND_CH, Tz) -> 拼接时间嵌入 -> (B, COND_CH+TIME_DIM, Tz)
        te = self.time_emb(t)                       # (B, TIME_DIM)
        te = te.unsqueeze(-1).expand(-1, -1, Tz)    # (B, TIME_DIM, Tz)
        return torch.cat([cond, te], dim=1)

    def forward(self, z, t, cond):
        # z: (B, z_ch, Tz)  Tz = T/16 ; cond: (B, cond_ch, Tz)
        x = self.act(self.inp(z))            # (B,64,Tz)
        cond_aug = self._aug_cond(cond, t, x.shape[-1])
        skips = []
        for i, d in enumerate(self.down):
            x = self.act(d(x))
            x = self.rb_down[i](x, cond_aug)
            skips.append(x)
        x = self.mid(x, cond_aug)
        for i, u in enumerate(self.up):
            s = skips[-1 - i]
            x = torch.cat([x, s], dim=1)
            x = self.act(u(x))
            x = self.rb_up[i](x, cond_aug)
        return self.out(x)


class DDPM(nn.Module):
    def __init__(self, z_ch=16, cond_ch=COND_CH, timesteps=TIMESTEPS):
        super().__init__()
        self.net = UNet1D(z_ch, cond_ch)
        self.cond_enc = MelCondEncoder(out_ch=cond_ch - 1)   # 输出 8ch, +1 onset
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, 0))

    def make_cond(self, mel, onset_env):
        """mel:(B,128,T) onset_env:(B,1,T) -> cond(B,9,T/16)"""
        m = self.cond_enc(mel)               # (B,8,T/16)
        oe = F.interpolate(onset_env, size=m.shape[-1], mode="nearest")
        return torch.cat([m, oe], dim=1)     # (B,9,T/16)

    def q_sample(self, z0, t, noise):
        a = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        b = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        return a * z0 + b * noise

    def forward(self, z0, mel, onset_env, cond_drop=False):
        B = z0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        zt = self.q_sample(z0, t, noise)
        cond = self.make_cond(mel, onset_env)
        if cond_drop and (torch.rand(1).item() < P_UNCOND):
            cond = torch.zeros_like(cond)
        eps_pred = self.net(zt, t, cond)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, mel, onset_env, steps=50, guidance=2.5, device="cpu"):
        cond = self.make_cond(mel, onset_env).to(device)
        B = mel.shape[0]
        z = torch.randn(B, 16, mel.shape[-1] // 16, device=device)
        cond_u = torch.zeros_like(cond)
        seq = list(reversed(range(0, self.timesteps, max(1, self.timesteps // steps))))
        for i, ti in enumerate(seq):
            zc = torch.cat([z, z], 0)
            t = torch.full((zc.shape[0],), ti, device=device, dtype=torch.long)
            cc = torch.cat([cond, cond_u], 0)
            eps = self.net(zc, t, cc)
            eps = torch.clamp(eps, -10.0, 10.0)        # 防溢出
            eps_c, eps_u = eps.chunk(2, dim=0)
            eps = (1 + guidance) * eps_c - guidance * eps_u
            beta = self.betas[ti]
            a = self.alphas_cumprod[ti]
            noise = torch.randn_like(z) if ti > 0 else torch.zeros_like(z)
            z = (1 / a.sqrt()) * (z - (beta / (1 - a).sqrt()) * eps) + (beta ** 0.5) * noise
            z = torch.clamp(z, -8.0, 8.0)              # 潜变量限幅
        return z
