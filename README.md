# FMelCodec

Official implementation of **FMelCodec**, an ultra-low-bitrate neural speech codec based on mel-spectrogram coding, flow-matching-based refinement, and vocoding-driven waveform reconstruction.

> **Paper:** *Ultra-Low-Bitrate Mel-Spectrogram-based Neural Speech Coding with Flow-Matching-based Refinement and Vocoding-driven Reconstruction*  
> **Status:** Accepted  
> **Demo:** https://redmist328.github.io/FMelCodec  
> **Code:** https://github.com/redmist328/FMelCodec  
> **Checkpoints:** https://huggingface.co/redmist2/FMelCodec <!-- TODO: replace with the final Hugging Face model repo if different -->

## Authors

**Hui-Peng Du**, **Yang Ai**, **Xiao-Hang Jiang**, **Yuan Tian**, and **Zhen-Hua Ling**

National Engineering Research Center of Speech and Language Information Processing,  
University of Science and Technology of China, Hefei, China

Contact: `redmist@mail.ustc.edu.cn`, `yangai@ustc.edu.cn`

## Abstract

Ultra-low-bitrate speech coding is pivotal for bandwidth-constrained communication and deep compression, yet maintaining naturalness and speaker identity at such extreme bit budgets remains challenging due to pronounced information loss and quantization instability. To this end, we propose FMelCodec, an ultra-low-bitrate neural speech codec in the mel-spectrogram domain, cast as a three-stage coding–refinement–reconstruction (CRR) framework that can operate at as low as 250 bps. In the CRR framework, the front-end mel-spectrogram coding stage employs a highly aggressive 640× compression/decompression encoder–decoder structure with a single 1024-entry VQ codebook, coupled with an online clustering strategy that reassigns underused codewords to prevent codebook collapse and preserve codebook diversity. The subsequent conditional flow matching (CFM)-based mel-spectrogram refinement stage leverages a lightweight velocity-field estimator and CFM-based solver to refine the codec-degraded mel-spectrogram produced by the preceding decoder, and adopts a self-consistency training scheme that supports fewer iterative inference steps for the purpose of reducing computational overhead. Finally, the vocoding-driven waveform reconstruction stage employs a HiFi-GAN vocoder to faithfully reconstruct waveform from the refined mel-spectrogram. Experiments conducted on two datasets spanning two sampling rates show that, under ultra-low-bitrate constraints of 250 bps for 16 kHz and 750 bps for 48 kHz, both objective and subjective evaluations consistently demonstrate that FMelCodec achieves higher speech reconstruction quality and speaker similarity, while incurring lower computational and model complexity.

## Overview

FMelCodec is formulated as a three-stage **coding–refinement–reconstruction (CRR)** framework:

1. **Mel-spectrogram coding** compresses mel-spectrograms with a ConvNeXt-v2-style encoder–decoder and a single 1024-entry VQ codebook.
2. **Flow-matching-based refinement** improves the coarse decoded mel-spectrogram with a lightweight conditional flow matching model.
3. **Vocoding-driven waveform reconstruction** uses a HiFi-GAN vocoder to synthesize the final waveform from the refined mel-spectrogram.

The open-source inference bundle supports batch inference for `.wav` and `.flac` files and writes reconstructed waveforms to the output directory while preserving the relative input directory structure.

## Installation

Create a Python environment and install dependencies:

```bash
conda create -n fmelcodec python=3.10 -y
conda activate fmelcodec
pip install -r requirements.txt
```

The basic dependencies include:

```text
numpy
scipy
librosa
soundfile
torch
pyyaml
diffusers
einops
```

For downloading checkpoints from Hugging Face, install:

```bash
pip install -U huggingface_hub
```

## Download Checkpoints

The pretrained checkpoints will be released on Hugging Face:

```bash
# TODO: replace redmist328/FMelCodec with the final Hugging Face repo if different
hf download redmist328/FMelCodec --local-dir checkpoints
```

A merged checkpoint is recommended for inference. The expected usage is:

```text
checkpoints/
└── fmelcodec_16k_250bps_merged.pt
```

## Inference

Prepare an input directory containing `.wav` or `.flac` files:

```text
examples/input/
├── sample_1.wav
└── sample_2.flac
```

Run inference with a merged checkpoint:

```bash
python inference.py \
  --input_dir examples/input \
  --output_dir generated_files \
  --merged_checkpoint checkpoints/fmelcodec_16k_250bps_merged.pt \
  --cfm_steps 4 \
  --device cuda
```

The generated waveforms will be saved to:

```text
generated_files/
├── sample_1.wav
├── sample_2.wav
└── rtf_summary.json
```

### Common arguments

| Argument | Description |
|---|---|
| `--input_dir` | Directory containing input `.wav` or `.flac` files. |
| `--output_dir` | Directory for generated waveforms. |
| `--merged_checkpoint` | Path to the merged checkpoint. Recommended for easy inference. |
| `--config` | Optional JSON config file. If omitted, default values in `config.py` are used. |
| `--cfm_steps` | Number of CFM inference steps. The default paper setting is 4. |
| `--max_files` | Maximum number of files to process. Use `0` to process all files. |
| `--device` | Device string. CUDA is required by the current inference script. |
| `--warmup` | Number of warm-up runs before measuring RTF. |

### Exporting a merged checkpoint

If you have separate codec, vocoder, and CFM checkpoints configured in `config.py` or a JSON config file, you can export a merged checkpoint:

```bash
python inference.py \
  --config config.json \
  --export_merged_checkpoint checkpoints/fmelcodec_16k_250bps_merged.pt \
  --device cuda
```

## Repository Structure

```text
FMelCodec/
├── README.md
├── requirements.txt
├── config.py
├── model.py
├── inference.py
├── checkpoints/              # optional, ignored by git
├── examples/
│   └── input/
└── generated_files/          # inference outputs
```

## Notes

- The current inference script requires CUDA.
- The script processes `.wav` and `.flac` files recursively under `--input_dir`.
- When using separate checkpoints rather than a merged checkpoint, please update the checkpoint paths and steps in `config.py` or provide a JSON config file through `--config`.
- The CFM module used by `inference.py` should be available in this repository or on `PYTHONPATH`. Please make sure the corresponding `CFM` implementation is included before release.
- Checkpoints should not be committed to Git. Please release them through Hugging Face or GitHub Releases.

## Citation

If you find this work useful, please cite:

```bibtex
@article{du2026fmelcodec,
  title={Ultra-Low-Bitrate Mel-Spectrogram-based Neural Speech Coding with Flow-Matching-based Refinement and Vocoding-driven Reconstruction},
  author={Du, Hui-Peng and Ai, Yang and Jiang, Xiao-Hang and Tian, Yuan and Ling, Zhen-Hua},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2026},
}
```

## License

MIT
