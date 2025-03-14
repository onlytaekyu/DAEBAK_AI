"""
설정 관리 모듈

이 모듈은 프로젝트의 설정을 관리하는 Config 클래스를 제공합니다.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """데이터 설정"""
    historical_data_path: str
    processed_data_path: str
    numbers_to_select: int
    min_number: int
    max_number: int
    batch_size: int
    num_workers: int

@dataclass
class ModelConfig:
    """모델 설정"""
    model_type: str
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    early_stopping_patience: int
    model_save_path: str

@dataclass
class TrainingConfig:
    """학습 설정"""
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    device: str
    num_workers: int
    batch_size: int
    shuffle: bool
    pin_memory: bool

@dataclass
class CudaConfig:
    """CUDA 설정"""
    use_gpu: bool
    memory_fraction: float = 0.95
    allow_tf32: bool = True
    benchmark: bool = True
    deterministic: bool = False
    cudnn_enabled: bool = True
    compile_mode: bool = True
    device: str = 'cuda'
    gradient_accumulation_steps: int = 4
    num_workers: int = 8
    pin_memory: bool = True
    precision: str = 'float16'

@dataclass
class Config:
    """설정 관리 클래스"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        설정 객체 초기화

        Args:
            config_dict: 설정 딕셔너리
        """
        self._config = config_dict or {}

        # CUDA 설정 초기화
        if 'cuda' in self._config:
            self.cuda = CudaConfig(**self._config['cuda'])
        else:
            self.cuda = CudaConfig(use_gpu=True)

        # 데이터 설정 초기화
        if 'data' in self._config:
            self.data = DataConfig(**self._config['data'])
        else:
            self.data = DataConfig(
                historical_data_path='lottery/data/raw/lottery.csv',
                processed_data_path='lottery/data/processed/processed_data.pkl',
                numbers_to_select=6,
                min_number=1,
                max_number=45,
                batch_size=32,
                num_workers=4
            )

        # 모델 설정 초기화
        if 'model' in self._config:
            self.model = ModelConfig(**self._config['model'])
        else:
            self.model = ModelConfig(
                model_type='lstm',
                input_size=45,
                hidden_size=128,
                num_layers=2,
                dropout=0.1,
                learning_rate=0.001,
                weight_decay=0.0001,
                epochs=100,
                early_stopping_patience=10,
                model_save_path='lottery/models'
            )

        # 학습 설정 초기화
        if 'training' in self._config:
            self.training = TrainingConfig(**self._config['training'])
        else:
            self.training = TrainingConfig(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
                device='cuda',
                num_workers=4,
                batch_size=32,
                shuffle=True,
                pin_memory=True
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        설정값 조회

        Args:
            key: 설정 키
            default: 기본값

        Returns:
            설정값
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        설정값 설정

        Args:
            key: 설정 키
            value: 설정값
        """
        self._config[key] = value

    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        설정 업데이트

        Args:
            config_dict: 업데이트할 설정 딕셔너리
        """
        self._config.update(config_dict)

    def save(self, filepath: str) -> None:
        """
        설정 저장

        Args:
            filepath: 저장할 파일 경로
        """
        try:
            save_dir = Path(filepath).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # YAML 형식으로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f'설정 저장 완료: {filepath}')
        except Exception as e:
            logger.error(f'설정 저장 실패: {str(e)}')
            raise

    def load(self, filepath: str) -> None:
        """
        설정 로드

        Args:
            filepath: 로드할 파일 경로
        """
        try:
            if not Path(filepath).exists():
                raise FileNotFoundError(f'설정 파일을 찾을 수 없습니다: {filepath}')

            # YAML 형식으로 로드
            with open(filepath, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            logger.info(f'설정 로드 완료: {filepath}')

            # 설정 객체 재초기화
            self.__init__(self._config)
        except Exception as e:
            logger.error(f'설정 로드 실패: {str(e)}')
            raise

    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 딕셔너리로 변환

        Returns:
            설정 딕셔너리
        """
        return {
            'cuda': self.cuda.__dict__,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__
        }

    def __str__(self) -> str:
        """문자열 표현"""
        return str(self.to_dict())

    def __repr__(self) -> str:
        """표현식 문자열"""
        return f'Config({self.to_dict()})' 