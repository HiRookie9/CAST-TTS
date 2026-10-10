<div align="center">

# CAST-TTS: A Simple Cross-Attention Framework for Unified Timbre Control in TTS

[![arxiv](https://img.shields.io/badge/arxiv-2603.16280-B31B1B.svg)](https://arxiv.org/abs/2603.16280)
[![project page](https://img.shields.io/badge/project-CAST--TTS%20Page-blue.svg)](https://hirookie9.github.io/CAST-TTS-Page/)

[project page](https://hirookie9.github.io/CAST-TTS-Page/) | [paper](https://arxiv.org/abs/2603.16280)

</div>

CAST-TTS is a conditional flow-matching TTS model that unifies timbre control in
TTS through a simple cross-attention framework. It generates speech either from
a reference audio prompt or from a natural-language description of the target
timbre, using a single model for both modes.

The code in this repository is derived from
[AccelHydra](https://github.com/wsntxxn/AccelHydra), and parts of the model
implementation originate from the F5-TTS and CapSpeech projects.

The runnable CAST-TTS example lives in
`src/accel_hydra/examples/cast-tts`. Everything else under `src/accel_hydra` is
the AccelHydra training framework itself.

## Environment

CAST-TTS was developed with Python 3.10 and PyTorch 2.3.0. PyTorch 2.3.0 is
recommended because the model code uses APIs and dtype behaviour that were
validated against that release.

```bash
conda create -n cast-tts python=3.10 -y
conda activate cast-tts

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

`requirements.txt` covers both the AccelHydra core and the CAST-TTS example
dependencies.

### Installing AccelHydra

Install AccelHydra from this repository, not from PyPI or the upstream GitHub
project:

```bash
cd /path/to/CAST
pip install -e .
```

Installing the released package instead

```bash
pip install accel_hydra
# or
pip install git+https://github.com/wsntxxn/AccelHydra
```

will pull a different AccelHydra version. That version is not compatible with
the CAST-TTS example and will cause configuration, trainer, and import errors.
The CAST-TTS code in this repository was written against the AccelHydra source
included here.

## Datasets

The training manifests are not stored in this repository. Download:

**[Datasets](https://drive.google.com/drive/folders/1nC7uNGBMH6EvVWhWuBeFlF2HEAktQIKA?usp=drive_link)**

<!-- DATA_LINK: replace the link above with the Google Drive URL once uploaded. -->

After downloading, place the JSONL files in the `data` directory next to
`train.py`, creating it if necessary:

```bash
cd src/accel_hydra/examples/cast-tts
mkdir -p data
# copy the downloaded JSONL files into ./data
```

The manifests consist of:

- `libritts_pmt_{train,test}.jsonl`: prompt/main text data, used in Stage 1 and Stage 3.
- `libritts_cap_{train,test}.jsonl`: LibriTTS caption data, used in Stage 2 and Stage 3.
- `giga_cap_{train,test}.jsonl`: GigaSpeech caption data, used in Stage 2 and Stage 3.

The audio referenced by these manifests is **not** distributed either. Prepare
the LibriTTS and GigaSpeech audio yourself and rewrite the `location` field of
each JSONL row to point at your local audio files.

Each JSONL row uses the following fields:

```json
{
  "location": "/absolute/path/to/audio.wav",
  "text": "transcript of the full utterance",
  "duration": 4.83,
  "mask_len": 0.0,
  "caption": "spoken description of the delivery, empty for prompt data"
}
```

The expected layout after downloading is:

```text
src/accel_hydra/examples/cast-tts/
├── data/
│   ├── libritts_pmt_train.jsonl
│   ├── libritts_pmt_test.jsonl
│   ├── libritts_cap_train.jsonl
│   ├── libritts_cap_test.jsonl
│   ├── giga_cap_train.jsonl
│   └── giga_cap_test.jsonl
├── train.py
└── configs/
```

This matches the default `data_root: data` setting in `configs/train.yaml`. If
you prefer to keep the manifests elsewhere, point `data_root` at that directory
instead of copying, for example:

```bash
accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
  train.py epochs=1 epoch_length=400000 \
  data_root=/path/to/your/data data@data_dict=stage1 model='stage1&3' \
  optimizer.lr=7.5e-5 exp_name=cast_stage1
```

## Checkpoints

The following pretrained models are downloaded automatically from Hugging Face
by ID, so no local cache path is hard-coded:

- Vocoder: [`nvidia/bigvgan_v2_24khz_100band_256x`](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x)
- Text encoder: [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large)
- Speaker/content encoder and duration predictor: [`OpenSound/CapSpeech-models`](https://huggingface.co/OpenSound/CapSpeech-models)

The speaker encoder loads `wavlm_large_finetune.pth` from the CapSpeech model
repository. Set `HF_HOME` if you want the Hugging Face cache in a specific
place.

## CAST-TTS checkpoints

Download: **[coming soon](./src/accel_hydra/examples/cast-tts/checkpoints)**

<!-- CKPT_LINK: replace the link above with the Google Drive URL once uploaded. -->

*(Pretrained CAST-TTS checkpoints are being uploaded to Google Drive and the
link above will be updated as soon as they are available. Users can already
train the three stages from scratch on their own data.)*

Each stage directory contains a `model.safetensors` file plus the `config.yaml`
used to train it. Download the checkpoint you need and pass its directory to
inference, or point a later training stage at it with
`+trainer.pretrained_weights=<path/to/checkpoint>`.

## Training

All commands below are run from the CAST-TTS example directory:

```bash
cd src/accel_hydra/examples/cast-tts
```

Training uses 8 GPUs by default. Adjust the Accelerate config in the shell
scripts if you have a different number of GPUs, for example
`configs/accelerate/nvidia/4gpus.yaml` or `configs/accelerate/nvidia/1gpu.yaml`.

The three stages are:

| Stage | Script | Updates | Peak LR | Data |
| --- | --- | --- | --- | --- |
| 1 | `train_stage1.sh` | 400K | 7.5e-5 | prompt data |
| 2 | `train_stage2.sh` | 200K | 1.5e-5 | caption data |
| 3 | `train_stage3.sh` | 100K | 2.5e-5 | prompt + caption data |

Run all three stages in sequence with:

```bash
bash bash_scripts/train_all.sh
```

Stage 2 loads the Stage 1 checkpoint through `trainer.pretrained_weights`, and
Stage 3 loads the Stage 2 checkpoint the same way. Checkpoints are written
under `experiments/cast_stage{1,2,3}/`.

To train a single stage manually, for example Stage 2 from a Stage 1 result:

```bash
bash bash_scripts/train_stage2.sh
```

To continue an interrupted run instead of loading only the model weights:

```bash
accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
  train.py epochs=1 epoch_length=200000 \
  data@data_dict=stage2 model=stage2 \
  optimizer.lr=1.5e-5 exp_name=cast_stage2 \
  +trainer.resume_from_checkpoint=experiments/cast_stage2/checkpoints/last
```

## Inference

Batch inference is implemented in `infer_batch.py`. Point `exp_dir` or
`ckpt_dir_or_file` at a CAST-TTS experiment directory or checkpoint file, and
provide a test JSONL, then run:

```bash
accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
  infer_batch.py exp_dir=experiments/cast_stage3 \
  test_jsonl=data/libritts_pmt_test.jsonl
```

## Acknowledgements

- [AccelHydra](https://github.com/wsntxxn/AccelHydra), the training framework
  this repository is built on and ships in `src/accel_hydra`.
- [F5-TTS](https://github.com/SWivid/F5-TTS), the flow-matching TTS work that
  the CAST-TTS model implementation builds upon.
- [CapSpeech](https://github.com/OpenSound/CapSpeech), the source of the
  caption data and the speaker and duration models used here.

## Citation

If you find this work useful, please cite the CAST-TTS paper and the projects
listed above.
