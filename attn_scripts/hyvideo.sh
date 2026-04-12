#!/bin/bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 sample_video.py \
  --seed "${HYVIDEO_SEED:-42}" \
  --video-size "${HYVIDEO_HEIGHT:-720}" "${HYVIDEO_WIDTH:-1280}" \
  --video-length "${HYVIDEO_LENGTH:-125}" \
  --infer-steps "${HYVIDEO_STEPS:-50}" \
  --prompt "${HYVIDEO_PROMPT:-A cat and a dog baking a cake together in a kitchen.}" \
  --flow-reverse \
  --use-cpu-offload \
  --save-path "${HYVIDEO_SAVE_PATH:-./results}" \
  "$@"
