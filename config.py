from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass
class CodecConfig:
    project_dir: str = "/disk1/duhuipeng/vocoder2/melcodec_fm_libri"
    checkpoint_dir: str = "cp_Encoder_Decoder"
    encoder_step: int = 1400000
    decoder_step: int = 1400000
    input_sr: int = 16000
    n_fft: int = 1024
    num_mels: int = 80
    hop_size: int = 160
    win_size: int = 640
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    latent_dim: int = 32
    amp_encoder_channel: int = 256
    amp_encoder_input_conv_kernel_size: int = 7
    amp_encoder_output_downconv_kernel_size: int = 7
    amp_decoder_channel: int = 256
    amp_decoder_input_upconv_kernel_size: int = 16
    pha_decoder_channel: int = 256
    pha_decoder_output_r_conv_kernel_size: int = 7
    latent_input_conv_kernel_size: int = 7
    latent_output_conv_kernel_size: int = 7
    ratio: int = 4

    @property
    def AMP_Encoder_channel(self) -> int:
        return self.amp_encoder_channel

    @property
    def AMP_Encoder_input_conv_kernel_size(self) -> int:
        return self.amp_encoder_input_conv_kernel_size

    @property
    def AMP_Encoder_output_downconv_kernel_size(self) -> int:
        return self.amp_encoder_output_downconv_kernel_size

    @property
    def AMP_Decoder_channel(self) -> int:
        return self.amp_decoder_channel

    @property
    def AMP_Decoder_input_upconv_kernel_size(self) -> int:
        return self.amp_decoder_input_upconv_kernel_size

    @property
    def PHA_Decoder_channel(self) -> int:
        return self.pha_decoder_channel

    @property
    def PHA_Decoder_output_R_conv_kernel_size(self) -> int:
        return self.pha_decoder_output_r_conv_kernel_size


@dataclass
class VocoderConfig:
    project_dir: str = "/disk1/duhuipeng/vocoder2/hifi-gan-master_lirbi"
    checkpoint_dir: str = "cp_hifigan"
    generator_step: int = 500000
    input_sr: int = 16000
    n_fft: int = 1024
    num_mels: int = 80
    hop_size: int = 160
    win_size: int = 640
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    resblock: str = "1"
    upsample_rates: list[int] = field(default_factory=lambda: [5, 4, 4, 2])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [11, 8, 8, 4])
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])


@dataclass
class CfmConfig:
    project_dir: str = "/disk1/duhuipeng/vocoder2/melcodec_fm_libri"
    checkpoint_dir: str = "cp_Encoder_Decoder"
    checkpoint_step: int = 1400000
    train_yaml: str = "train.yaml"
    sample_steps: int = 4
    in_channels: int = 160
    out_channels: int = 80
    cfm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "name": "CFM",
            "solver": "euler",
            "sigma_min": 1e-4,
        }
    )
    decoder_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "channels": [256, 256],
            "dropout": 0.05,
            "attention_head_dim": 64,
            "n_blocks": 1,
            "num_mid_blocks": 2,
            "num_heads": 2,
            "act_fn": "snakebeta",
        }
    )


@dataclass
class PipelineConfig:
    codec: CodecConfig = field(default_factory=CodecConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    cfm: CfmConfig = field(default_factory=CfmConfig)
    output_dir: str = "generated_files_open_source"
    input_dir: str = "/disk1/duhuipeng/datasets/LibriTTS-16k/test"
    merged_checkpoint: Optional[str] = None
    batch_warmup: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "codec": asdict(self.codec),
            "vocoder": asdict(self.vocoder),
            "cfm": asdict(self.cfm),
            "output_dir": self.output_dir,
            "input_dir": self.input_dir,
            "merged_checkpoint": self.merged_checkpoint,
            "batch_warmup": self.batch_warmup,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        codec = CodecConfig(**data.get("codec", {}))
        vocoder = VocoderConfig(**data.get("vocoder", {}))
        cfm = CfmConfig(**data.get("cfm", {}))
        return cls(
            codec=codec,
            vocoder=vocoder,
            cfm=cfm,
            output_dir=data.get("output_dir", "generated_files_open_source"),
            input_dir=data.get("input_dir", "/disk1/duhuipeng/datasets/LibriTTS-16k/test"),
            merged_checkpoint=data.get("merged_checkpoint"),
            batch_warmup=int(data.get("batch_warmup", 1)),
        )


def load_config(path: Optional[str] = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    with open(path, "r", encoding="utf-8") as f:
        return PipelineConfig.from_dict(json.load(f))


def save_config(config: PipelineConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
