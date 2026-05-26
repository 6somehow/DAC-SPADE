#!/bin/bash
set -euo pipefail

: "${WAN21_CKPT_DIR:?Set WAN21_CKPT_DIR to the Wan 2.1 checkpoint directory}"

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
=======
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WAN21_ROOT="${REPO_ROOT}/model/Wan2.1"

cd "${WAN21_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
>>>>>>> dev
python generate.py \
  --base_seed "${BASE_SEED:-42}" \
  --task "${WAN21_TASK:-t2v-14B}" \
  --size "${WAN21_SIZE:-1280*720}" \
  --ckpt_dir "${WAN21_CKPT_DIR}" \
  --prompt "${WAN21_PROMPT:-A cat and a dog baking a cake together in a kitchen.}" \
  --frame_num "${WAN21_FRAME_NUM:-61}" \
  --sample_steps "${WAN21_SAMPLE_STEPS:-50}" \
  --offload_model "${WAN21_OFFLOAD_MODEL:-True}" \
  "$@"
