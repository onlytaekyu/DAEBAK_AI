"""
통합 모델 관리 시스템

이 모듈은 다양한 머신러닝/딥러닝 모델을 통합하고 최적화하는 
범용 앙상블 시스템을 구현합니다.
"""

import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import yaml
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import itertools
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from shared.error_handler import get_logger, safe_execute, log_performance
from shared.cuda_optimizers import clear_memory, get_optimal_batch_size

logger = get_logger(__name__)

class ModelIntegrator:
    """
    범용 모델 통합 및 앙상블 관리 시스템
    
    특징:
    1. 다양한 모델 타입 지원 (PyTorch, LightGBM, 등)
    2. 동적 앙상블 가중치 최적화
    3. GPU 가속 및 메모리 최적화
    4. 자동 성능 모니터링
    5. 다양한 최적화 전략 지원
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        통합 모듈 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.models = {}
        self.model_weights = {}
        self.diversity_scores = {}
        self.weight_history = []
        
        # 성능 메트릭
        self.performance_metrics = {
            'inference_time': [],
            'memory_usage': [],
            'accuracy': [],
            'confidence': [],
            'roi': [],
            'sharpe_ratio': []
        }
        
        # 학습 파라미터
        self.weight_update_alpha = self.config.get('weight_update_alpha', 0.1)
        self.diversity_threshold = self.config.get('diversity_threshold', 0.3)
        self.uncertainty_weight = self.config.get('uncertainty_weight', 0.2)
        
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self._setup_cuda()
        
        # 메모리 캐시 관리
        self.memory_cache = {}
        self.max_cache_size = config.get('max_cache_size', 1000)
        
        # 최적화 설정
        self.dynamic_weights = self.config.get('dynamic_weights', True)
        self.min_diversity_score = self.config.get('min_diversity_score', 0.3)
        
        # 로깅 설정
        self._setup_logging()
    
    def _setup_cuda(self):
        """CUDA 최적화 설정"""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        
        # CUDA 메모리 관리
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.cuda.memory.set_per_process_memory_fraction(0.9)
        
        # CUDA 스트림 설정
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "model_integrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_model(self, name: str, model: Any, model_type: str, weight: float = 1.0):
        """
        모델 추가
        
        Args:
            name: 모델 이름
            model: 모델 인스턴스
            model_type: 모델 유형 ('pytorch', 'lightgbm', 등)
            weight: 초기 가중치
        """
        if name in self.models:
            self.logger.warning(f"모델 {name}이(가) 이미 존재합니다. 덮어쓰기를 수행합니다.")
        
        self.models[name] = {
            'model': model,
            'type': model_type
        }
        self.model_weights[name] = weight
        self._normalize_weights()
    
    def _normalize_weights(self):
        """가중치 정규화"""
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v/total for k, v in self.model_weights.items()}
    
    @safe_execute(default_return=None)
    def optimize_weights(self, validation_data: np.ndarray, 
                        validation_labels: np.ndarray,
                        method: str = 'bayesian') -> Dict[str, float]:
        """
        앙상블 가중치 최적화
        
        Args:
            validation_data: 검증 데이터
            validation_labels: 검증 레이블
            method: 최적화 방법 ('bayesian', 'grid', 'random')
            
        Returns:
            최적화된 가중치 딕셔너리
        """
        if method == 'bayesian':
            return self._optimize_weights_bayesian(validation_data, validation_labels)
        elif method == 'grid':
            return self._optimize_weights_grid(validation_data, validation_labels)
        else:
            return self._optimize_weights_random(validation_data, validation_labels)
    
    def _optimize_weights_bayesian(self, validation_data: np.ndarray,
                                 validation_labels: np.ndarray) -> Dict[str, float]:
        """베이지안 최적화"""
        def objective(weights):
            predictions = self._ensemble_predict(validation_data, weights)
            return -np.mean(predictions == validation_labels)
        
        # 초기 가중치 설정
        initial_weights = np.array(list(self.model_weights.values()))
        
        # 베이지안 최적화
        optimizer = BayesianOptimization(
            f=objective,
            pbounds={f'w{i}': (0, 1) for i in range(len(self.model_weights))},
            random_state=42
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=20
        )
        
        # 최적 가중치 적용
        optimal_weights = optimizer.max['params']
        return {name: optimal_weights[f'w{i}'] 
                for i, name in enumerate(self.model_weights.keys())}
    
    def _optimize_weights_grid(self, validation_data: np.ndarray,
                             validation_labels: np.ndarray) -> Dict[str, float]:
        """그리드 서치 최적화"""
        best_score = -float('inf')
        best_weights = self.model_weights.copy()
        
        # 그리드 포인트 생성
        grid_points = np.linspace(0, 1, 10)
        
        for weights in itertools.product(grid_points, repeat=len(self.model_weights)):
            if sum(weights) == 0:
                continue
                
            # 가중치 정규화
            normalized_weights = {name: w/sum(weights) 
                                for name, w in zip(self.model_weights.keys(), weights)}
            
            # 성능 평가
            predictions = self._ensemble_predict(validation_data, normalized_weights)
            score = np.mean(predictions == validation_labels)
            
            if score > best_score:
                best_score = score
                best_weights = normalized_weights
        
        return best_weights
    
    def _optimize_weights_random(self, validation_data: np.ndarray,
                               validation_labels: np.ndarray) -> Dict[str, float]:
        """랜덤 서치 최적화"""
        best_score = -float('inf')
        best_weights = self.model_weights.copy()
        
        for _ in range(100):
            # 랜덤 가중치 생성
            weights = np.random.random(len(self.model_weights))
            weights = weights / weights.sum()
            
            normalized_weights = {name: w 
                                for name, w in zip(self.model_weights.keys(), weights)}
            
            # 성능 평가
            predictions = self._ensemble_predict(validation_data, normalized_weights)
            score = np.mean(predictions == validation_labels)
            
            if score > best_score:
                best_score = score
                best_weights = normalized_weights
        
        return best_weights
    
    @safe_execute(default_return=None)
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        앙상블 예측 및 신뢰도 계산
        
        Args:
            data: 입력 데이터
            
        Returns:
            예측값과 신뢰도 점수
        """
        # 예측 수행
        predictions = self._ensemble_predict(data)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(predictions)
        
        # 성능 메트릭 업데이트
        self.performance_metrics['confidence'].append(confidence)
        
        return predictions, confidence
    
    def _ensemble_predict(self, data: np.ndarray, 
                         weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """앙상블 예측"""
        if weights is None:
            weights = self.model_weights
        
        predictions = []
        for name, info in self.models.items():
            try:
                # 캐시 확인
                cache_key = f"{name}_{hash(str(data))}"
                if cache_key in self.memory_cache:
                    pred = self.memory_cache[cache_key]
                else:
                    # 예측 수행
                    start_time = time.time()
                    model = info['model']
                    
                    if info['type'] == 'pytorch':
                        with torch.no_grad():
                            model.eval()
                            if isinstance(data, np.ndarray):
                                data_tensor = torch.tensor(data, device=self.device)
                            pred = model(data_tensor).cpu().numpy()
                    else:
                        pred = model.predict(data)
                    
                    end_time = time.time()
                    
                    # 성능 메트릭 기록
                    self.performance_metrics['inference_time'].append(end_time - start_time)
                    if torch.cuda.is_available():
                        self.performance_metrics['memory_usage'].append(
                            torch.cuda.memory_allocated() / 1024**2
                        )
                    
                    # 캐시 저장
                    if len(self.memory_cache) >= self.max_cache_size:
                        self.memory_cache.pop(next(iter(self.memory_cache)))
                    self.memory_cache[cache_key] = pred
                
                predictions.append(pred * weights[name])
                
            except Exception as e:
                self.logger.error(f"모델 {name} 예측 중 오류 발생: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("모든 모델 예측 실패")
        
        return np.sum(predictions, axis=0)
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """예측 신뢰도 계산"""
        try:
            # 1. 모델 간 일관성
            model_predictions = []
            for info in self.models.values():
                try:
                    model = info['model']
                    if info['type'] == 'pytorch':
                        with torch.no_grad():
                            model.eval()
                            pred = model(torch.tensor(predictions, device=self.device))
                            pred = pred.cpu().numpy()
                    else:
                        pred = model.predict(predictions)
                    model_predictions.append(pred)
                except:
                    continue
            
            if not model_predictions:
                return 0.5
            
            # 모델 간 일치도 계산
            consistency = np.mean([
                np.mean(p1 == p2) 
                for p1, p2 in itertools.combinations(model_predictions, 2)
            ])
            
            # 2. 예측 분산
            prediction_std = np.std(model_predictions, axis=0).mean()
            normalized_std = 1 - min(prediction_std / 0.5, 1)
            
            # 3. 모델 성능 가중치
            performance_weights = np.array([
                np.mean(self.performance_metrics.get(f'{name}_accuracy', [0.5]))
                for name in self.models.keys()
            ])
            performance_weights = performance_weights / performance_weights.sum()
            
            # 최종 신뢰도 계산
            confidence = (
                0.4 * consistency +
                0.3 * normalized_std +
                0.3 * np.mean(performance_weights)
            )
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 중 오류 발생: {str(e)}")
            return 0.5
    
    def save_state(self, path: str):
        """현재 상태 저장"""
        state = {
            'config': self.config,
            'model_weights': self.model_weights,
            'performance_metrics': self.performance_metrics,
            'models': {
                name: {
                    'type': info['type'],
                    'state': (
                        info['model'].state_dict() if info['type'] == 'pytorch'
                        else info['model'].get_model_state()
                    )
                }
                for name, info in self.models.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"상태 저장됨: {path}")
    
    @classmethod
    def load_state(cls, path: str) -> 'ModelIntegrator':
        """저장된 상태 로드"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        integrator = cls(state['config'])
        integrator.model_weights = state['model_weights']
        integrator.performance_metrics = state['performance_metrics']
        
        # 모델 상태 복원
        for name, info in state['models'].items():
            if name in integrator.models:
                model = integrator.models[name]['model']
                if info['type'] == 'pytorch':
                    model.load_state_dict(info['state'])
                else:
                    model.set_model_state(info['state'])
        
        return integrator
    
    def plot_performance(self, save_path: Optional[str] = None):
        """성능 지표 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('모델 통합 성능 지표', fontsize=16)
        
        # 추론 시간
        axes[0, 0].plot(self.performance_metrics['inference_time'])
        axes[0, 0].set_title('추론 시간')
        axes[0, 0].set_xlabel('반복')
        axes[0, 0].set_ylabel('시간 (초)')
        
        # 메모리 사용량
        if self.performance_metrics['memory_usage']:
            axes[0, 1].plot(self.performance_metrics['memory_usage'])
            axes[0, 1].set_title('GPU 메모리 사용량')
            axes[0, 1].set_xlabel('반복')
            axes[0, 1].set_ylabel('메모리 (MB)')
        
        # 정확도
        axes[1, 0].plot(self.performance_metrics['accuracy'])
        axes[1, 0].set_title('정확도')
        axes[1, 0].set_xlabel('반복')
        axes[1, 0].set_ylabel('정확도')
        
        # 신뢰도
        axes[1, 1].plot(self.performance_metrics['confidence'])
        axes[1, 1].set_title('예측 신뢰도')
        axes[1, 1].set_xlabel('반복')
        axes[1, 1].set_ylabel('신뢰도')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"성능 그래프 저장됨: {save_path}")
        else:
            plt.show()
