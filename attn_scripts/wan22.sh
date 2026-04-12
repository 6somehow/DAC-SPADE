#!/bin/bash
set -euo pipefail

: "${WAN22_CKPT_DIR:?Set WAN22_CKPT_DIR to the Wan 2.2 checkpoint directory}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python generate.py \
  --base_seed "${BASE_SEED:-42}" \
  --task "${WAN22_TASK:-t2v-A14B}" \
  --size "${WAN22_SIZE:-1280*720}" \
  --ckpt_dir "${WAN22_CKPT_DIR}" \
  --prompt "${WAN22_PROMPT:-A serene garden in late afternoon.}" \
  --frame_num "${WAN22_FRAME_NUM:-61}" \
  --sample_steps "${WAN22_SAMPLE_STEPS:-50}" \
  --convert_model_dtype \
  --t5_cpu \
  --offload_model "${WAN22_OFFLOAD_MODEL:-True}" \
  "$@"
