"""
모델 통합 및 앙상블 시스템

이 모듈은 다양한 머신러닝/딥러닝 모델을 통합하고 최적화하는 시스템을 제공합니다.
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
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from .cuda_optimizers import CUDAOptimizer, TensorCache, InferenceBuffer
from .memory_manager import MemoryManager
from .error_handler import safe_execute, log_performance, get_logger
from dataclasses import dataclass
from queue import PriorityQueue
import hashlib
from scipy import stats

logger = get_logger(__name__)

@dataclass
class ModelConfig:
    """모델 설정 데이터 클래스"""
    name: str
    type: str
    params: Dict[str, Any]
    device: str
    batch_size: int
    precision: str
    use_amp: bool
    num_workers: int
    pin_memory: bool

@dataclass
class EnsembleConfig:
    """앙상블 설정 데이터 클래스"""
    method: str
    weights: Optional[List[float]]
    n_splits: int
    random_state: int
    n_trials: int
    timeout: int

class ModelIntegrator:
    """모델 통합 시스템"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        모델 통합 시스템 초기화
        
        Args:
            config: 설정 정보
        """
        self.config = config
        self.models: Dict[str, Any] = {}
        self.weights: List[float] = []
        self.ensemble_config = EnsembleConfig(**config.get('ensemble', {}))
        self.cuda_optimizer = CUDAOptimizer(config.get('cuda', {}))
        self.memory_manager = MemoryManager()
        self._setup_logging()
        
        # 스레드 안전성을 위한 락
        self.model_lock = threading.Lock()
        self.weight_lock = threading.Lock()
        
        # 성능 모니터링
        self.performance_metrics = {
            'inference_times': [],
            'memory_usage': [],
            'errors': []
        }
    
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
    
    @safe_execute
    def add_model(self, model: Any, config: ModelConfig):
        """
        모델 추가
        
        Args:
            model: 모델 객체
            config: 모델 설정
        """
        with self.model_lock:
            if config.name in self.models:
                raise ValueError(f"모델 '{config.name}'이(가) 이미 존재합니다.")
            
            # CUDA 최적화 적용
            if config.device == 'cuda':
                model = self.cuda_optimizer.optimize_model(model)
            
            self.models[config.name] = {
                'model': model,
                'config': config
            }
            
            # 가중치 초기화
            with self.weight_lock:
                self.weights = [1.0 / len(self.models)] * len(self.models)
            
            self.logger.info(f"모델 '{config.name}' 추가 완료")
    
    @safe_execute
    def remove_model(self, model_name: str):
        """
        모델 제거
        
        Args:
            model_name: 모델 이름
        """
        with self.model_lock:
            if model_name not in self.models:
                raise ValueError(f"모델 '{model_name}'이(가) 존재하지 않습니다.")
            
            del self.models[model_name]
            
            # 가중치 재조정
            with self.weight_lock:
                if self.models:
                    self.weights = [1.0 / len(self.models)] * len(self.models)
                else:
                    self.weights = []
            
            self.logger.info(f"모델 '{model_name}' 제거 완료")
    
    @safe_execute
    @log_performance
    def optimize_weights(self, X: np.ndarray, y: np.ndarray, method: str = None):
        """
        앙상블 가중치 최적화
        
        Args:
            X: 입력 데이터
            y: 타겟 데이터
            method: 최적화 방법 (기본값: 설정된 방법)
        """
        if not self.models:
            raise ValueError("최적화할 모델이 없습니다.")
        
        method = method or self.ensemble_config.method
        
        with self.weight_lock:
            if method == 'bayesian':
                self._optimize_weights_bayesian(X, y)
            elif method == 'grid':
                self._optimize_weights_grid(X, y)
            elif method == 'random':
                self._optimize_weights_random(X, y)
            else:
                raise ValueError(f"지원하지 않는 최적화 방법입니다: {method}")
    
    def _optimize_weights_bayesian(self, X: np.ndarray, y: np.ndarray):
        """베이지안 최적화로 가중치 최적화"""
        def objective(trial):
            weights = []
            for _ in range(len(self.models)):
                weights.append(trial.suggest_float(f'weight_{_}', 0, 1))
            weights = np.array(weights) / np.sum(weights)
            
            predictions = self._get_predictions(X, weights)
            mse = mean_squared_error(y, predictions)
            return mse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=self.ensemble_config.n_trials,
            timeout=self.ensemble_config.timeout
        )
        
        best_weights = []
        for i in range(len(self.models)):
            best_weights.append(study.best_params[f'weight_{i}'])
        self.weights = np.array(best_weights) / np.sum(best_weights)
    
    def _optimize_weights_grid(self, X: np.ndarray, y: np.ndarray):
        """그리드 서치로 가중치 최적화"""
        n_splits = self.ensemble_config.n_splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.ensemble_config.random_state)
        
        best_weights = None
        best_score = float('inf')
        
        # 그리드 포인트 생성
        grid_points = np.linspace(0, 1, 10)
        for weights in self._generate_weight_combinations(grid_points, len(self.models)):
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 각 모델 학습
                for model_info in self.models.values():
                    model = model_info['model']
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                
                # 검증 세트로 예측
                predictions = self._get_predictions(X_val, weights)
                score = mean_squared_error(y_val, predictions)
                scores.append(score)
            
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_weights = weights
        
        self.weights = best_weights
    
    def _optimize_weights_random(self, X: np.ndarray, y: np.ndarray):
        """랜덤 서치로 가중치 최적화"""
        n_splits = self.ensemble_config.n_splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.ensemble_config.random_state)
        
        best_weights = None
        best_score = float('inf')
        
        for _ in range(self.ensemble_config.n_trials):
            weights = np.random.rand(len(self.models))
            weights = weights / np.sum(weights)
            
            scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 각 모델 학습
                for model_info in self.models.values():
                    model = model_info['model']
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                
                # 검증 세트로 예측
                predictions = self._get_predictions(X_val, weights)
                score = mean_squared_error(y_val, predictions)
                scores.append(score)
            
            mean_score = np.mean(scores)
            if mean_score < best_score:
                best_score = mean_score
                best_weights = weights
        
        self.weights = best_weights
    
    def _generate_weight_combinations(self, grid_points: np.ndarray, n_models: int) -> List[np.ndarray]:
        """가중치 조합 생성"""
        if n_models == 1:
            return [np.array([1.0])]
        
        combinations = []
        for weights in self._generate_weights_recursive(grid_points, n_models):
            if np.isclose(np.sum(weights), 1.0):
                combinations.append(weights)
        return combinations
    
    def _generate_weights_recursive(self, grid_points: np.ndarray, n_models: int) -> List[np.ndarray]:
        """재귀적으로 가중치 조합 생성"""
        if n_models == 1:
            return [[w] for w in grid_points]
        
        combinations = []
        for w in grid_points:
            sub_combinations = self._generate_weights_recursive(grid_points, n_models - 1)
            for sub_combo in sub_combinations:
                combinations.append([w] + sub_combo)
        return combinations
    
    @safe_execute
    @log_performance
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        앙상블 예측 수행
        
        Args:
            X: 입력 데이터
            
        Returns:
            예측 결과
        """
        if not self.models:
            raise ValueError("예측할 모델이 없습니다.")
        
        with self.weight_lock:
            weights = self.weights
        
        return self._get_predictions(X, weights)
    
    def _get_predictions(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        가중치를 적용한 예측 수행
        
        Args:
            X: 입력 데이터
            weights: 모델 가중치
            
        Returns:
            가중 평균 예측 결과
        """
        predictions = []
        
        # 병렬 예측 수행
        with ThreadPoolExecutor() as executor:
            future_to_model = {
                executor.submit(self._predict_single_model, model_info['model'], X): model_info
                for model_info in self.models.values()
            }
            
            for future in future_to_model:
                try:
                    pred = future.result()
                    predictions.append(pred)
                except Exception as e:
                    self.logger.error(f"모델 예측 중 오류 발생: {e}")
                    raise
        
        # 가중 평균 계산
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_pred += pred * weight
        
        return weighted_pred
    
    def _predict_single_model(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        단일 모델 예측
        
        Args:
            model: 모델 객체
            X: 입력 데이터
            
        Returns:
            예측 결과
        """
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif hasattr(model, 'forward'):
            # PyTorch 모델
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                if next(model.parameters()).is_cuda:
                    X_tensor = X_tensor.cuda()
                return model(X_tensor).cpu().numpy()
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {type(model)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        info = {}
        for name, model_info in self.models.items():
            info[name] = {
                'type': model_info['config'].type,
                'device': model_info['config'].device,
                'batch_size': model_info['config'].batch_size,
                'precision': model_info['config'].precision,
                'use_amp': model_info['config'].use_amp
            }
        return info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        성능 메트릭 반환
        
        Returns:
            성능 메트릭 딕셔너리
        """
        return {
            'inference_times': self.performance_metrics['inference_times'],
            'memory_usage': self.performance_metrics['memory_usage'],
            'errors': self.performance_metrics['errors']
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.memory_manager.cleanup()
        self.cuda_optimizer.cleanup()
