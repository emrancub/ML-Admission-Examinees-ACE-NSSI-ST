import subprocess
from src.config import USE_CUDA, CUDA_REQUESTED, CUDA_DEVICE

print('CUDA requested:', CUDA_REQUESTED)
print('CUDA detected:', USE_CUDA)
print('Configured CUDA device:', CUDA_DEVICE)

try:
    print(subprocess.check_output(['nvidia-smi'], text=True))
except Exception as e:
    print('nvidia-smi unavailable:', e)

try:
    import xgboost
    from xgboost import XGBClassifier
    print('XGBoost:', xgboost.__version__)
except Exception as e:
    print('XGBoost check failed:', e)

try:
    import catboost
    print('CatBoost:', catboost.__version__)
except Exception as e:
    print('CatBoost check failed:', e)
