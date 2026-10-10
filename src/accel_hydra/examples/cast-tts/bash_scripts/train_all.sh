#!/usr/bin/env bash
set -euo pipefail

# Run the three CAST-TTS training stages back to back.
cd "$(dirname "$0")/.."

bash bash_scripts/train_stage1.sh
bash bash_scripts/train_stage2.sh
bash bash_scripts/train_stage3.sh
