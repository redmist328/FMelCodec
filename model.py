from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union, List
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d, AvgPool1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) // 2)


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def load_checkpoint(path: str, device: torch.device) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    print(f"Loading '{path}'", flush=True)
    return torch.load(path, map_location=device)


def list_audio_files(input_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith((".wav", ".flac")):
                files.append(os.path.join(root, name))
    return sorted(files)


class GRN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor, cond_embedding_id=None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            raise NotImplementedError("AdaNorm is not used in this open-source inference bundle.")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + x


class FeaturePool:
    def __init__(self, pool_size: int, dim: int):
        self.pool_size = pool_size
        self.dim = dim
        self.features: list[torch.Tensor] = []

    def query(self, features: torch.Tensor) -> torch.Tensor:
        if len(self.features) < self.pool_size:
            self.features.append(features.detach().cpu())
            return features
        idx = torch.randint(0, len(self.features), (features.size(0),))
        pool = torch.stack([self.features[i.item()] for i in idx], dim=0).to(features.device)
        return pool[: features.size(0)]


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, online_clustered: bool = True):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.used_count = torch.zeros(self.codebook_size)
        self.reset_used_count = False
        self.batch_count = 1
        self.reset_batch_count = False
        self.avg_probs_all = 0.0
        self.register_buffer("embed_prob", torch.zeros(self.codebook_size))
        self.decay = 0.99
        self.anchor = "probrandom"
        self.pool = FeaturePool(self.codebook_size, self.codebook_dim)
        self.contras_loss = False
        self.balancing_loss = False
        self.online_clustered = online_clustered

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        b, d, t = latents.shape
        encodings = latents.permute(0, 2, 1).reshape(-1, d)
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        codewords = (-dist).max(1)[1]
        encodings_indices = torch.zeros(encodings.shape[0], self.codebook_size, device=latents.device)
        encodings_indices.scatter_(1, codewords.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings_indices, dim=0)
        batch_used_count = (avg_probs > 0).sum().item()
        batch_utilization_rate = batch_used_count / self.codebook_size
        if self.training:
            self.reset_used_count = True
        if (not self.training) and self.reset_used_count:
            self.used_count = torch.zeros(self.codebook_size)
            self.reset_used_count = False
        batch_used_indices = torch.nonzero(avg_probs > 0).squeeze()
        self.used_count = self.used_count.to(latents.device)
        if batch_used_indices.numel() > 0:
            self.used_count[batch_used_indices] += 1
        utilization_rate = (self.used_count > 0).sum().item() / self.codebook_size
        batch_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        if self.training:
            self.reset_batch_count = True
        if (not self.training) and self.reset_batch_count:
            self.batch_count = 1
            self.reset_batch_count = False
        self.avg_probs_all = (self.batch_count - 1) / self.batch_count * self.avg_probs_all + 1 / self.batch_count * avg_probs
        perplexity = torch.exp(-torch.sum(self.avg_probs_all * torch.log(self.avg_probs_all + 1e-10)))
        self.batch_count += 1
        codewords = codewords.reshape(b, t)
        z_q = self.decode_code(codewords)
        if self.online_clustered and self.training:
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)
            if self.anchor == "closest":
                sort_distance, indices = dist.sort(dim=0)
                random_feat = encodings.detach()[indices[-1, :]]
            elif self.anchor == "probrandom":
                norm_distance = F.softmax(dist.t(), dim=1)
                prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                random_feat = encodings.detach()[prob]
            else:
                random_feat = self.pool.query(encodings.detach())
            decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) / (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
            self.codebook.weight.data = self.codebook.weight.data * (1 - decay) + random_feat * decay
        if self.contras_loss:
            sort_distance, indices = dist.sort(dim=0)
            dis_pos = sort_distance[-max(1, int(sort_distance.size(0) / self.codebook_size)) :, :].mean(dim=0, keepdim=True)
            dis_neg = sort_distance[: int(sort_distance.size(0) * 1 / 2), :]
            dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
            contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
        else:
            contra_loss = torch.zeros(1, device=latents.device)
        if self.balancing_loss:
            balancing_loss = F.binary_cross_entropy(avg_probs, torch.full_like(avg_probs, 1.0 / self.codebook_size))
        else:
            balancing_loss = torch.zeros(1, device=latents.device)
        return z_q, codewords, (batch_perplexity, perplexity), (batch_utilization_rate, utilization_rate), contra_loss, balancing_loss

    def forward(self, z: torch.Tensor):
        z_e = self.in_proj(z)
        z_q, indices, perplexity, utilization_rates, contra_loss, balancing_loss = self.decode_latents(z_e)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        ssim_loss = torch.zeros(1, device=z.device)
        return (
            z_q,
            indices,
            rearrange_latents(z_e),
            commitment_loss,
            codebook_loss,
            perplexity,
            utilization_rates,
            contra_loss,
            balancing_loss,
            ssim_loss,
        )


def rearrange_latents(z_e: torch.Tensor) -> torch.Tensor:
    b, d, t = z_e.shape
    return z_e.permute(0, 2, 1).reshape(b * t, d)


class ResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim[i], online_clustered=True) for i in range(n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout
        self.ssim_loss = False

    def forward(self, z: torch.Tensor, n_quantizers: Optional[int] = None):
        z_q = torch.zeros_like(z)
        residual = z
        commitment_loss = torch.zeros(z.size(0), device=z.device)
        codebook_loss = torch.zeros(z.size(0), device=z.device)
        codebook_indices = []
        latents = []
        batch_perplexities = []
        perplexities = []
        batch_utilization_rates = []
        utilization_rates = []
        contra_loss = torch.zeros(1, device=z.device)
        balancing_loss = torch.zeros(1, device=z.device)
        ssim_loss = torch.zeros(1, device=z.device)

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training and self.quantizer_dropout > 0:
            n_quantizers = torch.ones((z.shape[0],), device=z.device) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],), device=z.device)
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break
            z_q_i, indices_i, z_e_i, commitment_loss_i, codebook_loss_i, perplexities_i, utilization_rates_i, contra_loss_i, balancing_loss_i, ssim_loss_i = quantizer(residual)
            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            contra_loss += (contra_loss_i * mask).mean()
            balancing_loss += (balancing_loss_i * mask).mean()
            batch_perplexities.append(perplexities_i[0])
            perplexities.append(perplexities_i[1])
            batch_utilization_rates.append(utilization_rates_i[0])
            utilization_rates.append(utilization_rates_i[1])
            codebook_indices.append(indices_i)
            latents.append(z_e_i)
            ssim_loss += (ssim_loss_i * mask).mean()
        codes = torch.stack(codebook_indices, dim=1) if codebook_indices else torch.empty(0)
        latents = torch.stack(latents, dim=1) if latents else torch.empty(0)
        return (
            z_q,
            codes,
            latents,
            commitment_loss,
            codebook_loss,
            (batch_perplexities, perplexities),
            (batch_utilization_rates, utilization_rates),
            contra_loss,
            balancing_loss,
            ssim_loss,
        )


class Encoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.input_channels = 80
        self.h = h
        self.dim = 256
        self.num_layers = 8
        self.intermediate_dim = 512
        self.embed_logamp = nn.Conv1d(self.input_channels, self.dim, kernel_size=7, padding=3)
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.convnext_logamp = nn.ModuleList(
            [ConvNeXtBlock(self.dim, self.intermediate_dim, layer_scale_init_value=1 / self.num_layers) for _ in range(self.num_layers)]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        self.out_logamp = nn.Linear(self.dim, h.AMP_Encoder_channel)
        self.AMP_Encoder_downsample_output_conv = weight_norm(
            Conv1d(
                h.AMP_Encoder_channel,
                h.AMP_Encoder_channel,
                h.AMP_Encoder_output_downconv_kernel_size,
                h.ratio,
                padding=get_padding(h.AMP_Encoder_output_downconv_kernel_size, 1),
            )
        )
        self.latent_output_conv = weight_norm(
            Conv1d(h.AMP_Encoder_channel, h.latent_dim, h.latent_output_conv_kernel_size, 1, padding=get_padding(h.latent_output_conv_kernel_size, 1))
        )
        self.AMP_Encoder_downsample_output_conv.apply(init_weights)
        self.latent_output_conv.apply(init_weights)
        self.quantizer = ResidualVectorQuantize(
            input_dim=h.latent_dim,
            codebook_dim=h.latent_dim,
            n_codebooks=1,
            codebook_size=1024,
            quantizer_dropout=False,
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, MDCT):
        x = self.embed_logamp(MDCT)
        x = self.norm_logamp(x.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext_logamp:
            x = block(x, cond_embedding_id=None)
        x = self.final_layer_norm_logamp(x.transpose(1, 2))
        x = self.out_logamp(x).transpose(1, 2)
        x = self.AMP_Encoder_downsample_output_conv(x)
        latent = self.latent_output_conv(x)
        latent, _, _, commitment_loss, codebook_loss, _, _, _, _, _ = self.quantizer(latent)
        return latent, commitment_loss, codebook_loss


class Decoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.dim = 256
        self.num_layers = 8
        self.intermediate_dim = 512
        self.latent_input_conv = weight_norm(
            Conv1d(h.latent_dim, h.AMP_Decoder_channel, h.latent_input_conv_kernel_size, 1, padding=get_padding(h.latent_input_conv_kernel_size, 1))
        )
        self.AMP_Decoder_upsample_input_conv = weight_norm(
            ConvTranspose1d(
                h.AMP_Decoder_channel,
                h.AMP_Decoder_channel,
                h.AMP_Decoder_input_upconv_kernel_size,
                h.ratio,
                padding=(h.AMP_Decoder_input_upconv_kernel_size - h.ratio) // 2,
            )
        )
        self.norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.convnext_logamp = nn.ModuleList(
            [ConvNeXtBlock(self.dim, self.intermediate_dim, layer_scale_init_value=1 / self.num_layers) for _ in range(self.num_layers)]
        )
        self.final_layer_norm_logamp = nn.LayerNorm(self.dim, eps=1e-6)
        self.apply(self._init_weights)
        self.out_logamp = nn.Linear(self.dim, h.AMP_Encoder_channel)
        self.PHA_Decoder_output_R_conv = weight_norm(
            Conv1d(h.PHA_Decoder_channel, 80, h.PHA_Decoder_output_R_conv_kernel_size, 1, padding=get_padding(h.PHA_Decoder_output_R_conv_kernel_size, 1))
        )
        self.PHA_Decoder_output_R_conv.apply(init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, latent):
        latent = self.latent_input_conv(latent)
        logamp = self.AMP_Decoder_upsample_input_conv(latent)
        logamp = self.out_logamp(logamp.transpose(1, 2))
        logamp = self.norm_logamp(logamp).transpose(1, 2)
        for block in self.convnext_logamp:
            logamp = block(logamp, cond_embedding_id=None)
        logamp = self.final_layer_norm_logamp(logamp.transpose(1, 2)).transpose(1, 2)
        mdct_coeff = self.PHA_Decoder_output_R_conv(logamp)
        mdct_coeff = mdct_coeff.permute(0, 2, 1)
        return mdct_coeff.permute(0, 2, 1)


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            ]
        )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2 ** i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class MelCodecHifiGanPipeline(nn.Module):
    def __init__(self, codec_encoder: Encoder, codec_decoder: Decoder, hifigan: Generator):
        super().__init__()
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.hifigan = hifigan

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        latent, _, _ = self.codec_encoder(mel)
        coarse = self.codec_decoder(latent)
        if coarse.size(-1) % 2 != 0:
            coarse = coarse[:, :, :-1]
        return self.hifigan(coarse)
