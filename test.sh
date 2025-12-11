#!/usr/bin/env bash
set -e          # exit on error
set -u          # undefined vars are errors
set -x          # print executed commands
set -o pipefail # fail a pipeline if any command fails

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Execute release test on GPU: ${GPU_ID}"

python test.py \
  --data-root ./dataset/R2C7K \
  --checkpoint RefOnce.pth \
  --batch-size 22 \
  --input-size 384 \
  --save-dir ./output/release/ \
  --save-preds False

# Test script includes the online evaluation and will print the evaluation results after testing
# As the evaluation is on a single CPU, this process will take some time to finish.