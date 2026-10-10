#!/usr/bin/env bash
set -euo pipefail

# Stage 1: 400K updates, peak learning rate 7.5e-5.
accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
  train.py epochs=1 epoch_length=400000 \
  data@data_dict=stage1 model='stage1&3' \
  optimizer.lr=7.5e-5 exp_name=cast_stage1 \
  trainer.permanent_save_every_n_steps=50000
