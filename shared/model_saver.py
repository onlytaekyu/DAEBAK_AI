"""
모델 저장/로드 시스템

이 모듈은 모델의 저장, 로드, 버전 관리, 체크포인트 검증을 위한 유틸리티를 제공합니다.
"""

import os
import torch
import pickle
import json
import hashlib
import shutil
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import time
from dataclasses import dataclass
import zlib
import base64
from .memory_manager import MemoryManager
from .error_handler import safe_execute, log_performance

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """체크포인트 메타데이터"""
    version: str
    timestamp: datetime
    model_type: str
    model_hash: str
    config_hash: str
    size: int
    dependencies: List[str]
    metrics: Dict[str, float]
    is_valid: bool = True

class ModelSaver:
    """모델 저장/로드 시스템"""
    
    def __init__(self, base_dir: str = "checkpoints"):
        """
        모델 저장 시스템 초기화
        
        Args:
            base_dir: 체크포인트 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory_manager = MemoryManager()
        self._setup_logging()
        
        # 스레드 안전성을 위한 락
        self.save_lock = threading.Lock()
        self.load_lock = threading.Lock()
        
        # 저장/로드 큐
        self.save_queue = queue.PriorityQueue()
        self.load_queue = queue.PriorityQueue()
        
        # 백그라운드 작업자
        self.save_worker = threading.Thread(target=self._save_worker, daemon=True)
        self.load_worker = threading.Thread(target=self._load_worker, daemon=True)
        self.save_worker.start()
        self.load_worker.start()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "model_saver.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    @safe_execute
    def save_model(
        self,
        model: Any,
        name: str,
        version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        priority: int = 1
    ) -> str:
        """
        모델 저장
        
        Args:
            model: 저장할 모델
            name: 모델 이름
            version: 모델 버전
            config: 모델 설정
            metrics: 성능 메트릭
            priority: 저장 우선순위 (낮을수록 높은 우선순위)
            
        Returns:
            저장된 체크포인트 경로
        """
        # 메타데이터 준비
        metadata = self._prepare_metadata(model, name, version, config, metrics)
        
        # 저장 작업 큐에 추가
        self.save_queue.put((priority, {
            'model': model,
            'name': name,
            'version': version,
            'metadata': metadata,
            'config': config
        }))
        
        return str(self.base_dir / name / version)
    
    def _save_worker(self):
        """백그라운드 저장 작업자"""
        while True:
            try:
                _, task = self.save_queue.get()
                self._save_checkpoint(task)
            except Exception as e:
                self.logger.error(f"모델 저장 중 오류 발생: {e}")
            finally:
                self.save_queue.task_done()
    
    def _save_checkpoint(self, task: Dict[str, Any]):
        """
        체크포인트 저장
        
        Args:
            task: 저장 작업 정보
        """
        with self.save_lock:
            try:
                model = task['model']
                name = task['name']
                version = task['version']
                metadata = task['metadata']
                config = task['config']
                
                # 체크포인트 디렉토리 생성
                checkpoint_dir = self.base_dir / name / version
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # 모델 상태 저장
                if hasattr(model, 'state_dict'):
                    # PyTorch 모델
                    state_dict = model.state_dict()
                    torch.save(state_dict, checkpoint_dir / "model.pt")
                else:
                    # 일반 모델
                    with open(checkpoint_dir / "model.pkl", 'wb') as f:
                        pickle.dump(model, f)
                
                # 설정 저장
                with open(checkpoint_dir / "config.json", 'w') as f:
                    json.dump(config, f, indent=2)
                
                # 메타데이터 저장
                with open(checkpoint_dir / "metadata.json", 'w') as f:
                    json.dump(metadata.__dict__, f, indent=2, default=str)
                
                # 체크포인트 검증
                if self._validate_checkpoint(checkpoint_dir):
                    self.logger.info(f"체크포인트 저장 완료: {checkpoint_dir}")
                else:
                    raise ValueError("체크포인트 검증 실패")
                
            except Exception as e:
                self.logger.error(f"체크포인트 저장 중 오류 발생: {e}")
                raise
    
    @safe_execute
    def load_model(
        self,
        name: str,
        version: str,
        device: Optional[str] = None,
        priority: int = 1
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        모델 로드
        
        Args:
            name: 모델 이름
            version: 모델 버전
            device: 로드할 디바이스
            priority: 로드 우선순위 (낮을수록 높은 우선순위)
            
        Returns:
            (로드된 모델, 설정 정보)
        """
        # 로드 작업 큐에 추가
        future = queue.Future()
        self.load_queue.put((priority, {
            'name': name,
            'version': version,
            'device': device,
            'future': future
        }))
        
        # 결과 대기
        return future.result()
    
    def _load_worker(self):
        """백그라운드 로드 작업자"""
        while True:
            try:
                _, task = self.load_queue.get()
                self._load_checkpoint(task)
            except Exception as e:
                self.logger.error(f"모델 로드 중 오류 발생: {e}")
            finally:
                self.load_queue.task_done()
    
    def _load_checkpoint(self, task: Dict[str, Any]):
        """
        체크포인트 로드
        
        Args:
            task: 로드 작업 정보
        """
        with self.load_lock:
            try:
                name = task['name']
                version = task['version']
                device = task['device']
                future = task['future']
                
                checkpoint_dir = self.base_dir / name / version
                
                # 체크포인트 검증
                if not self._validate_checkpoint(checkpoint_dir):
                    raise ValueError("체크포인트 검증 실패")
                
                # 메타데이터 로드
                with open(checkpoint_dir / "metadata.json", 'r') as f:
                    metadata = CheckpointMetadata(**json.load(f))
                
                # 설정 로드
                with open(checkpoint_dir / "config.json", 'r') as f:
                    config = json.load(f)
                
                # 모델 로드
                if (checkpoint_dir / "model.pt").exists():
                    # PyTorch 모델
                    state_dict = torch.load(checkpoint_dir / "model.pt")
                    if device:
                        state_dict = {k: v.to(device) for k, v in state_dict.items()}
                    model = self._create_model_from_config(config)
                    model.load_state_dict(state_dict)
                else:
                    # 일반 모델
                    with open(checkpoint_dir / "model.pkl", 'rb') as f:
                        model = pickle.load(f)
                
                future.set_result((model, config))
                
            except Exception as e:
                self.logger.error(f"체크포인트 로드 중 오류 발생: {e}")
                future.set_exception(e)
    
    def _prepare_metadata(
        self,
        model: Any,
        name: str,
        version: str,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> CheckpointMetadata:
        """
        메타데이터 준비
        
        Args:
            model: 모델 객체
            name: 모델 이름
            version: 모델 버전
            config: 모델 설정
            metrics: 성능 메트릭
            
        Returns:
            체크포인트 메타데이터
        """
        # 모델 해시 계산
        if hasattr(model, 'state_dict'):
            state_dict = model.state_dict()
            model_bytes = pickle.dumps(state_dict)
        else:
            model_bytes = pickle.dumps(model)
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        # 설정 해시 계산
        config_bytes = json.dumps(config, sort_keys=True).encode()
        config_hash = hashlib.sha256(config_bytes).hexdigest()
        
        # 체크포인트 크기 계산
        checkpoint_size = len(model_bytes) + len(config_bytes)
        
        return CheckpointMetadata(
            version=version,
            timestamp=datetime.now(),
            model_type=type(model).__name__,
            model_hash=model_hash,
            config_hash=config_hash,
            size=checkpoint_size,
            dependencies=self._get_model_dependencies(model),
            metrics=metrics or {}
        )
    
    def _validate_checkpoint(self, checkpoint_dir: Path) -> bool:
        """
        체크포인트 검증
        
        Args:
            checkpoint_dir: 체크포인트 디렉토리
            
        Returns:
            검증 성공 여부
        """
        try:
            # 필수 파일 존재 확인
            required_files = ["model.pt", "config.json", "metadata.json"]
            for file in required_files:
                if not (checkpoint_dir / file).exists():
                    return False
            
            # 메타데이터 로드
            with open(checkpoint_dir / "metadata.json", 'r') as f:
                metadata = CheckpointMetadata(**json.load(f))
            
            # 모델 파일 검증
            if (checkpoint_dir / "model.pt").exists():
                state_dict = torch.load(checkpoint_dir / "model.pt")
                model_bytes = pickle.dumps(state_dict)
            else:
                with open(checkpoint_dir / "model.pkl", 'rb') as f:
                    model_bytes = f.read()
            
            # 해시 검증
            model_hash = hashlib.sha256(model_bytes).hexdigest()
            if model_hash != metadata.model_hash:
                return False
            
            # 설정 파일 검증
            with open(checkpoint_dir / "config.json", 'r') as f:
                config = json.load(f)
            config_bytes = json.dumps(config, sort_keys=True).encode()
            config_hash = hashlib.sha256(config_bytes).hexdigest()
            if config_hash != metadata.config_hash:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"체크포인트 검증 중 오류 발생: {e}")
            return False
    
    def _get_model_dependencies(self, model: Any) -> List[str]:
        """
        모델 의존성 정보 수집
        
        Args:
            model: 모델 객체
            
        Returns:
            의존성 목록
        """
        dependencies = []
        
        if hasattr(model, 'state_dict'):
            # PyTorch 모델
            for name, param in model.named_parameters():
                if param.requires_grad:
                    dependencies.append(f"param_{name}")
            
            for name, buffer in model.named_buffers():
                dependencies.append(f"buffer_{name}")
        else:
            # 일반 모델
            for attr in dir(model):
                if not attr.startswith('_'):
                    value = getattr(model, attr)
                    if isinstance(value, (torch.nn.Module, torch.Tensor)):
                        dependencies.append(attr)
        
        return dependencies
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> Any:
        """
        설정으로부터 모델 생성
        
        Args:
            config: 모델 설정
            
        Returns:
            생성된 모델
        """
        # 모델 클래스 가져오기
        model_class = self._get_model_class(config['type'])
        
        # 모델 생성
        return model_class(**config['params'])
    
    def _get_model_class(self, model_type: str) -> type:
        """
        모델 클래스 가져오기
        
        Args:
            model_type: 모델 타입
            
        Returns:
            모델 클래스
        """
        # 모델 클래스 매핑
        model_classes = {
            'Linear': torch.nn.Linear,
            'Conv2d': torch.nn.Conv2d,
            'LSTM': torch.nn.LSTM,
            'Transformer': torch.nn.Transformer,
            # 추가 모델 클래스...
        }
        
        if model_type not in model_classes:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")
        
        return model_classes[model_type]
    
    def get_checkpoint_info(self, name: str, version: str) -> Dict[str, Any]:
        """
        체크포인트 정보 조회
        
        Args:
            name: 모델 이름
            version: 모델 버전
            
        Returns:
            체크포인트 정보
        """
        checkpoint_dir = self.base_dir / name / version
        
        if not checkpoint_dir.exists():
            raise ValueError(f"체크포인트가 존재하지 않습니다: {checkpoint_dir}")
        
        with open(checkpoint_dir / "metadata.json", 'r') as f:
            metadata = CheckpointMetadata(**json.load(f))
        
        return {
            'path': str(checkpoint_dir),
            'version': metadata.version,
            'timestamp': metadata.timestamp,
            'model_type': metadata.model_type,
            'size': metadata.size,
            'metrics': metadata.metrics,
            'is_valid': metadata.is_valid
        }
    
    def cleanup(self):
        """리소스 정리"""
        self.memory_manager.cleanup() 