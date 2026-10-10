#!/usr/bin/env bash
set -euo pipefail

# Stage 3: 100K updates, peak learning rate 2.5e-5.
# Stage 3 is initialized from the final Stage 2 checkpoint.
accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
  train.py epochs=1 epoch_length=100000 \
  data@data_dict=stage3 model='stage1&3' \
  optimizer.lr=2.5e-5 exp_name=cast_stage3 \
  trainer.permanent_save_every_n_steps=25000 \
  +trainer.pretrained_weights=experiments/cast_stage2/ckpt_step_200000
