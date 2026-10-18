#!/usr/bin/env bash
set -e
export R2_USE_CUDA=1
export R2_CUDA_DEVICE=0
export R2_N_ITER=25
python cuda_check.py
python main.py
