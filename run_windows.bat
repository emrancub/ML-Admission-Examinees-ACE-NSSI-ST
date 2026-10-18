@echo off
set R2_USE_CUDA=1
set R2_CUDA_DEVICE=0
set R2_N_ITER=25
python cuda_check.py
python main.py
pause
