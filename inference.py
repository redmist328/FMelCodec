from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from typing import Optional

from config import PipelineConfig, load_config
from model import Decoder, Encoder, Generator, MelCodecHifiGanPipeline, load_checkpoint


MAX_WAV_VALUE = 32768.0


def list_audio_files(input_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, names in os.walk(input_dir):
        for name in names:
            if name.lower().endswith((".wav", ".flac")):
                files.append(os.path.join(root, name))
    return sorted(files)


def build_mel_basis(cfg: PipelineConfig, device: torch.device) -> torch.Tensor:
    mel = librosa.filters.mel(
        sr=cfg.codec.input_sr,
        n_fft=cfg.codec.n_fft,
        n_mels=cfg.codec.num_mels,
        fmin=cfg.codec.mel_fmin,
        fmax=cfg.codec.mel_fmax,
    )
    return torch.tensor(mel, dtype=torch.float32, device=device)


def mel_spectrogram(audio: torch.Tensor, cfg: PipelineConfig, mel_basis: torch.Tensor, hann_window: torch.Tensor) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    audio = audio.squeeze(1)
    spec = torch.stft(
        audio,
        n_fft=cfg.codec.n_fft,
        hop_length=cfg.codec.hop_size,
        win_length=cfg.codec.win_size,
        window=hann_window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    mag = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    mel = torch.matmul(mel_basis, mag)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def latest_step(cp_dir: str, prefix: str) -> int:
    steps = []
    if not Path(cp_dir).is_dir():
        raise FileNotFoundError(cp_dir)
    for name in os.listdir(cp_dir):
        if name.startswith(prefix + "_"):
            try:
                steps.append(int(name.split("_")[-1]))
            except ValueError:
                pass
    if not steps:
        raise RuntimeError(f"no checkpoints with prefix {prefix} in {cp_dir}")
    return max(steps)


def _import_cfm_modules(cfm_project_dir: str):
    cfm_project_dir = os.path.abspath(cfm_project_dir)
    if cfm_project_dir not in sys.path:
        sys.path.insert(0, cfm_project_dir)
    from flowmaatching import CFM

    return CFM


def load_codec_and_vocoder(cfg: PipelineConfig, device: torch.device, merged_checkpoint: Optional[str] = None):
    if merged_checkpoint:
        bundle = torch.load(merged_checkpoint, map_location=device)
        codec_cfg = PipelineConfig.from_dict(bundle["meta"]).codec if "meta" in bundle else cfg.codec
        voc_cfg = PipelineConfig.from_dict(bundle["meta"]).vocoder if "meta" in bundle else cfg.vocoder
        encoder = Encoder(codec_cfg).to(device)
        decoder = Decoder(codec_cfg).to(device)
        generator = Generator(voc_cfg).to(device)
        encoder.load_state_dict(bundle["codec"]["encoder"], strict=True)
        decoder.load_state_dict(bundle["codec"]["decoder"], strict=True)
        generator.load_state_dict(bundle["vocoder"]["generator"], strict=True)
        return encoder.eval(), decoder.eval(), generator.eval(), bundle

    codec_cp_dir = Path(cfg.codec.project_dir) / cfg.codec.checkpoint_dir
    vocoder_cp_dir = Path(cfg.vocoder.project_dir) / cfg.vocoder.checkpoint_dir
    enc_step = cfg.codec.encoder_step or latest_step(str(codec_cp_dir), "encoder")
    dec_step = cfg.codec.decoder_step or latest_step(str(codec_cp_dir), "decoder")
    gen_step = cfg.vocoder.generator_step or latest_step(str(vocoder_cp_dir), "g")
    enc_ckpt = codec_cp_dir / f"encoder_{enc_step:08d}"
    dec_ckpt = codec_cp_dir / f"decoder_{dec_step:08d}"
    gen_ckpt = vocoder_cp_dir / f"g_{gen_step:08d}"
    encoder = Encoder(cfg.codec).to(device)
    decoder = Decoder(cfg.codec).to(device)
    generator = Generator(cfg.vocoder).to(device)
    encoder.load_state_dict(load_checkpoint(str(enc_ckpt), device)["encoder"], strict=True)
    decoder.load_state_dict(load_checkpoint(str(dec_ckpt), device)["decoder"], strict=True)
    generator.load_state_dict(load_checkpoint(str(gen_ckpt), device)["generator"], strict=True)
    bundle = {
        "codec": {"encoder_step": enc_step, "decoder_step": dec_step, "encoder_ckpt": str(enc_ckpt), "decoder_ckpt": str(dec_ckpt)},
        "vocoder": {"generator_step": gen_step, "generator_ckpt": str(gen_ckpt)},
    }
    return encoder.eval(), decoder.eval(), generator.eval(), bundle


def load_cfm(cfg: PipelineConfig, device: torch.device, merged_checkpoint: Optional[str] = None):
    bundle = None
    if merged_checkpoint:
        bundle = torch.load(merged_checkpoint, map_location=device)
        if "meta" in bundle:
            cfm_cfg = PipelineConfig.from_dict(bundle["meta"]).cfm
        else:
            cfm_cfg = cfg.cfm
    else:
        cfm_cfg = cfg.cfm

    CFM = _import_cfm_modules(cfm_cfg.project_dir)
    cfm = CFM(
        in_channels=cfm_cfg.in_channels,
        out_channel=cfm_cfg.out_channels,
        cfm_params=cfm_cfg.cfm_params,
        decoder_params=cfm_cfg.decoder_params,
    ).to(device)

    if bundle is not None and "cfm" in bundle and "generator" in bundle["cfm"]:
        cfm.load_state_dict(bundle["cfm"]["generator"], strict=True)
        cfm_ckpt = bundle["cfm"].get("checkpoint", merged_checkpoint)
        source = {"cfm_checkpoint": cfm_ckpt, "cfm_source": "merged_checkpoint"}
    else:
        cfm_cp_dir = Path(cfm_cfg.project_dir) / cfm_cfg.checkpoint_dir
        cfm_step = cfm_cfg.checkpoint_step or latest_step(str(cfm_cp_dir), "encoder")
        cfm_ckpt = cfm_cp_dir / f"encoder_{cfm_step:08d}"
        cfm.load_state_dict(load_checkpoint(str(cfm_ckpt), device)["generator"], strict=True)
        source = {"cfm_checkpoint": str(cfm_ckpt), "cfm_source": str(cfm_cp_dir), "cfm_step": cfm_step}

    return cfm.eval(), source


def save_merged_checkpoint(
    path: str,
    cfg: PipelineConfig,
    encoder: Encoder,
    decoder: Decoder,
    generator: Generator,
    cfm: Optional[torch.nn.Module] = None,
    extra: Optional[dict] = None,
) -> None:
    bundle = {
        "meta": cfg.to_dict(),
        "codec": {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        },
        "vocoder": {
            "generator": generator.state_dict(),
        },
    }
    if cfm is not None:
        bundle["cfm"] = {
            "generator": cfm.state_dict(),
            "checkpoint_step": cfg.cfm.checkpoint_step,
            "sample_steps": cfg.cfm.sample_steps,
            "project_dir": cfg.cfm.project_dir,
            "checkpoint_dir": cfg.cfm.checkpoint_dir,
            "in_channels": cfg.cfm.in_channels,
            "out_channels": cfg.cfm.out_channels,
            "cfm_params": cfg.cfm.cfm_params,
            "decoder_params": cfg.cfm.decoder_params,
        }
    if extra:
        bundle.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def forward_pipeline(
    encoder: Encoder,
    decoder: Decoder,
    cfm: torch.nn.Module,
    generator: Generator,
    mel: torch.Tensor,
    cfm_steps: int,
) -> torch.Tensor:
    latent, _, _ = encoder(mel)
    coarse = decoder(latent)
    if coarse.size(-1) % 2 != 0:
        coarse = coarse[:, :, :-1]
    mask = torch.ones(coarse.size(0), 1, coarse.size(-1), device=coarse.device, dtype=coarse.dtype)
    refined = cfm(coarse, mask, cfm_steps, 1.0)
    return generator(refined)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--merged_checkpoint", default=None)
    parser.add_argument("--export_merged_checkpoint", default=None)
    parser.add_argument("--cfm_steps", type=int, default=None)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.merged_checkpoint:
        cfg.merged_checkpoint = args.merged_checkpoint
    if args.cfm_steps is not None:
        cfg.cfm.sample_steps = int(args.cfm_steps)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this bundle.")

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    encoder, decoder, generator, bundle = load_codec_and_vocoder(cfg, device, merged_checkpoint=cfg.merged_checkpoint)
    cfm, cfm_source = load_cfm(cfg, device, merged_checkpoint=cfg.merged_checkpoint)

    if args.export_merged_checkpoint:
        save_merged_checkpoint(
            args.export_merged_checkpoint,
            cfg,
            encoder,
            decoder,
            generator,
            cfm=cfm,
            extra={"source": bundle, "cfm_source": cfm_source},
        )
        print(f"saved merged checkpoint to {args.export_merged_checkpoint}")
        return

    mel_basis = build_mel_basis(cfg, device)
    hann_window = torch.hann_window(cfg.codec.win_size, device=device)

    files = list_audio_files(cfg.input_dir)
    if args.max_files > 0:
        files = files[: args.max_files]
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = MelCodecHifiGanPipeline(encoder, decoder, generator).to(device).eval()
    bundle_info = bundle
    if cfg.merged_checkpoint:
        bundle_info = {"merged_checkpoint": cfg.merged_checkpoint, "cfm_source": cfm_source}
    print(
        json.dumps(
            {
                "input_dir": cfg.input_dir,
                "output_dir": cfg.output_dir,
                "num_files": len(files),
                "bundle": bundle_info,
                "cfm_steps": cfg.cfm.sample_steps,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    total_audio_sec = 0.0
    total_gpu_sec = 0.0
    start_all = time.time()

    with torch.no_grad():
        if files and args.warmup > 0:
            wav_np, _ = librosa.load(files[0], sr=cfg.codec.input_sr, mono=True)
            wav = torch.from_numpy(wav_np).float().to(device)
            mel = mel_spectrogram(wav.unsqueeze(0), cfg, mel_basis, hann_window)
            for _ in range(args.warmup):
                _ = forward_pipeline(encoder, decoder, cfm, generator, mel, cfg.cfm.sample_steps)
            torch.cuda.synchronize()

        for idx, filename in enumerate(files, 1):
            wav_np, _ = librosa.load(filename, sr=cfg.codec.input_sr, mono=True)
            total_audio_sec += len(wav_np) / float(cfg.codec.input_sr)
            wav = torch.from_numpy(wav_np).float().to(device)
            mel = mel_spectrogram(wav.unsqueeze(0), cfg, mel_basis, hann_window)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            audio = forward_pipeline(encoder, decoder, cfm, generator, mel, cfg.cfm.sample_steps)
            end.record()
            torch.cuda.synchronize()
            total_gpu_sec += start.elapsed_time(end) / 1000.0

            audio = audio.squeeze().detach().cpu().numpy()
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * MAX_WAV_VALUE).astype(np.int16)
            rel = os.path.relpath(filename, cfg.input_dir)
            out = os.path.join(cfg.output_dir, os.path.splitext(rel)[0] + ".wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            sf.write(out, audio, cfg.codec.input_sr)
            if idx == 1 or idx % 50 == 0 or idx == len(files):
                rtf = total_gpu_sec / max(total_audio_sec, 1e-9)
                print(f"[{idx}/{len(files)}] gpu_sec={total_gpu_sec:.2f} audio_sec={total_audio_sec:.2f} rtf={rtf:.6f}")

    summary = {
        "input_dir": cfg.input_dir,
        "output_dir": cfg.output_dir,
        "num_files": len(files),
        "total_audio_sec": total_audio_sec,
        "total_gpu_sec": total_gpu_sec,
        "rtf_gpu": total_gpu_sec / max(total_audio_sec, 1e-9),
        "wall_sec": time.time() - start_all,
        "merged_checkpoint": cfg.merged_checkpoint,
        "cfm_steps": cfg.cfm.sample_steps,
    }
    with open(os.path.join(cfg.output_dir, "rtf_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
