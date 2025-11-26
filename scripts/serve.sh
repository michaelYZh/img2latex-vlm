#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

cd /u/scottc/img2latex-vlm
conda activate vllm

vllm serve ./outputs/1/checkpoint-6000/merged/ \
    --served-model-name Qwen2-VL-2B-Instruct-img2latex-vlm \
    --port 8000 \
    --gpu-memory-utilization 0.9