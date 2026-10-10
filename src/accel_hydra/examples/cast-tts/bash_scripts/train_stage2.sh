#!/usr/bin/env bash
set -euo pipefail

# Stage 2: 200K updates, peak learning rate 1.5e-5.
# Stage 2 is initialized from the final Stage 1 checkpoint.
accelerate launch --config_file configs/accelerate/nvidia/8gpus.yaml \
  train.py epochs=1 epoch_length=200000 \
  data@data_dict=stage2 model=stage2 \
  optimizer.lr=1.5e-5 exp_name=cast_stage2 \
  trainer.permanent_save_every_n_steps=50000 \
  +trainer.pretrained_weights=experiments/cast_stage1/ckpt_step_400000
