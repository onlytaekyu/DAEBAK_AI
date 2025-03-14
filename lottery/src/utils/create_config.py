"""
YAML 설정 파일 생성 스크립트
"""

import yaml
from pathlib import Path

# 설정 딕셔너리 생성
config = {
    'model': {
        'input_dim': 100,
        'hidden_dim': 256,
        'num_layers': 2,
        'output_dim': 6,
        'dropout': 0.2
    },
    'training': {
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'patience': 10,
        'gradient_accumulation_steps': 2
    },
    'pytorch': {
        'compile_mode': 'reduce-overhead',
        'use_amp': True,
        'scheduler': {
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        }
    },
    'lightgbm': {
        'use_gpu': True,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': -1,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': 42,
        'early_stopping_rounds': 50
    },
    'ensemble': {
        'weights': None,
        'model_save_dir': 'models/ensemble',
        'use_stacking': False
    },
    'data': {
        'sequence_length': 10,
        'test_size': 0.2,
        'validation_size': 0.1,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4
    }
}

# 설정 파일 경로
config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'

# 디렉토리 생성
config_path.parent.mkdir(parents=True, exist_ok=True)

# YAML 파일로 저장
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"설정 파일이 생성되었습니다: {config_path}") 