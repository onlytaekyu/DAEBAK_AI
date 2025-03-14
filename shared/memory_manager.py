"""
메모리 관리 유틸리티

이 모듈은 시스템 메모리 관리, 메모리 누수 감지, 메모리 사용량 예측,
메모리 최적화 제안을 위한 유틸리티를 제공합니다.
"""

import torch
import psutil
import gc
from functools import wraps
import logging
from typing import Dict, Any, Callable, List, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from collections import deque
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import os
from .error_handler import setup_logger

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """메모리 통계 데이터 클래스"""
    total: int
    available: int
    used: int
    free: int
    cached: int
    buffers: int
    swap_total: int
    swap_used: int
    swap_free: int
    gpu_total: Optional[int] = None
    gpu_used: Optional[int] = None
    gpu_free: Optional[int] = None
    gpu_cached: Optional[int] = None

class MemoryLeakDetector:
    """메모리 누수 감지기"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.memory_history = deque(maxlen=window_size)
        self.leak_threshold = 0.1  # 10% 이상의 지속적인 메모리 증가를 누수로 간주
        self.lock = threading.Lock()
    
    def update(self, stats: MemoryStats):
        """메모리 통계 업데이트"""
        with self.lock:
            self.memory_history.append(stats)
    
    def detect_leak(self) -> Tuple[bool, float]:
        """
        메모리 누수 감지
        
        Returns:
            (누수 여부, 누수 점수)
        """
        if len(self.memory_history) < 2:
            return False, 0.0
            
        with self.lock:
            # 메모리 사용량 변화 계산
            memory_changes = []
            for i in range(1, len(self.memory_history)):
                prev = self.memory_history[i-1]
                curr = self.memory_history[i]
                change = (curr.process_memory - prev.process_memory) / prev.process_memory
                memory_changes.append(change)
            
            # 누수 점수 계산 (지속적인 증가 추세)
            leak_score = np.mean(memory_changes) if memory_changes else 0.0
            has_leak = leak_score > self.leak_threshold
            
            return has_leak, leak_score

class MemoryPredictor:
    """메모리 사용량 예측기"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.memory_history = deque(maxlen=history_size)
        self.lock = threading.Lock()
    
    def update(self, stats: MemoryStats):
        """메모리 통계 업데이트"""
        with self.lock:
            self.memory_history.append(stats)
    
    def predict_peak_memory(self, time_window: timedelta) -> float:
        """
        미래 메모리 사용량 예측
        
        Args:
            time_window: 예측 시간 범위
            
        Returns:
            예측된 최대 메모리 사용량 (MB)
        """
        if len(self.memory_history) < 2:
            return 0.0
            
        with self.lock:
            # 시계열 데이터 준비
            timestamps = np.array([(s.timestamp - self.memory_history[0].timestamp).total_seconds() 
                                 for s in self.memory_history])
            memory_values = np.array([s.process_memory for s in self.memory_history])
            
            # 선형 회귀로 추세 예측
            z = np.polyfit(timestamps, memory_values, 1)
            p = np.poly1d(z)
            
            # 미래 시점 예측
            future_time = (self.memory_history[-1].timestamp + time_window - 
                         self.memory_history[0].timestamp).total_seconds()
            predicted_memory = p(future_time)
            
            return max(0.0, predicted_memory)

class MemoryOptimizer:
    """메모리 최적화 제안기"""
    
    def __init__(self):
        self.optimization_history = []
        self.lock = threading.Lock()
    
    def analyze_memory_usage(self, stats: MemoryStats) -> List[str]:
        """
        메모리 사용량 분석 및 최적화 제안
        
        Args:
            stats: 메모리 통계
            
        Returns:
            최적화 제안 목록
        """
        suggestions = []
        
        # 1. 시스템 메모리 분석
        if stats.system_memory > 90:
            suggestions.append("시스템 메모리 사용량이 매우 높습니다. 불필요한 프로세스를 종료하거나 메모리를 확보하세요.")
        
        # 2. CUDA 메모리 분석
        if stats.cuda_allocated is not None:
            if stats.cuda_allocated > 0.9 * stats.cuda_reserved:
                suggestions.append("CUDA 메모리 사용량이 높습니다. 배치 크기를 줄이거나 모델을 최적화하세요.")
            
            if stats.fragmentation > 0.3:
                suggestions.append("CUDA 메모리 단편화가 심각합니다. 메모리 정리를 실행하세요.")
        
        # 3. 프로세스 메모리 분석
        if stats.process_memory > 1000:  # 1GB 이상
            suggestions.append("프로세스 메모리 사용량이 높습니다. 메모리 누수를 확인하세요.")
        
        # 4. 메모리 누수 분석
        if stats.memory_leak_score > 0.05:
            suggestions.append(f"메모리 누수가 감지되었습니다 (누수 점수: {stats.memory_leak_score:.2f}).")
        
        return suggestions

class MemoryProfiler:
    """메모리 프로파일러"""
    
    def __init__(self):
        self.profiling_data = []
        self.lock = threading.Lock()
    
    def start_profiling(self):
        """프로파일링 시작"""
        self.profiling_data = []
    
    def record_memory_event(self, event_type: str, size: float, timestamp: datetime):
        """
        메모리 이벤트 기록
        
        Args:
            event_type: 이벤트 유형
            size: 메모리 크기 (MB)
            timestamp: 이벤트 시간
        """
        with self.lock:
            self.profiling_data.append({
                'type': event_type,
                'size': size,
                'timestamp': timestamp
            })
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        프로파일링 요약 정보
        
        Returns:
            프로파일링 요약 정보
        """
        if not self.profiling_data:
            return {}
            
        with self.lock:
            # 이벤트 유형별 통계
            event_stats = {}
            for event in self.profiling_data:
                event_type = event['type']
                if event_type not in event_stats:
                    event_stats[event_type] = {
                        'count': 0,
                        'total_size': 0.0,
                        'max_size': 0.0
                    }
                stats = event_stats[event_type]
                stats['count'] += 1
                stats['total_size'] += event['size']
                stats['max_size'] = max(stats['max_size'], event['size'])
            
            return {
                'total_events': len(self.profiling_data),
                'event_stats': event_stats,
                'timeline': self.profiling_data
            }

class MemoryManager:
    """메모리 관리 유틸리티"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        메모리 관리자 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self.logger = setup_logger('memory_manager')
        
        # 메모리 통계 초기화
        self.memory_stats = MemoryStats(
            total=0,
            available=0,
            used=0,
            free=0,
            cached=0,
            buffers=0,
            swap_total=0,
            swap_used=0,
            swap_free=0
        )
        
        # GPU 메모리 통계 초기화
        if torch.cuda.is_available():
            self.memory_stats.gpu_total = torch.cuda.get_device_properties(0).total_memory
            self.memory_stats.gpu_used = 0
            self.memory_stats.gpu_free = self.memory_stats.gpu_total
            self.memory_stats.gpu_cached = 0
        
        # 메모리 사용량 기록
        self.memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        
        # 메모리 임계값 설정
        self.memory_threshold = self.config.get('memory_threshold', 0.9)  # 90%
        self.gpu_memory_threshold = self.config.get('gpu_memory_threshold', 0.9)  # 90%
        
        # 메모리 모니터링 간격
        self.monitoring_interval = self.config.get('monitoring_interval', 1.0)  # 1초
        
        # 마지막 메모리 정리 시간
        self.last_cleanup_time = time.time()
        
        # 메모리 정리 간격
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5분
        
        self.logger.info("메모리 관리자 초기화 완료")
        
        self.leak_detector = MemoryLeakDetector()
        self.predictor = MemoryPredictor()
        self.optimizer = MemoryOptimizer()
        self.profiler = MemoryProfiler()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "memory_manager.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def clear_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def report_memory_usage(self) -> MemoryStats:
        """
        메모리 사용량 보고
        
        Returns:
            메모리 통계 정보
        """
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024**2  # MB
        
        stats = MemoryStats(
            timestamp=datetime.now(),
            system_memory=psutil.virtual_memory().percent,
            cuda_allocated=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
            cuda_reserved=torch.cuda.memory_reserved() / 1024**2 if torch.cuda.is_available() else None,
            process_memory=process_memory,
            memory_leak_score=0.0,
            fragmentation=0.0
        )
        
        # CUDA 메모리 단편화 계산
        if stats.cuda_allocated is not None and stats.cuda_reserved is not None:
            if stats.cuda_reserved > 0:
                stats.fragmentation = 1 - (stats.cuda_allocated / stats.cuda_reserved)
        
        # 메모리 누수 점수 계산
        has_leak, leak_score = self.leak_detector.detect_leak()
        stats.memory_leak_score = leak_score
        
        # 통계 업데이트
        self.leak_detector.update(stats)
        self.predictor.update(stats)
        
        return stats
    
    def memory_cleanup_decorator(self, func: Callable) -> Callable:
        """
        함수 실행 전후 메모리 정리를 수행하는 데코레이터
        
        Args:
            func: 대상 함수
            
        Returns:
            래핑된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.clear_gpu_memory()
            gc.collect()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.clear_gpu_memory()
                gc.collect()
                
        return wrapper
    
    @contextmanager
    def track_memory_usage(self, tag: str = ""):
        """
        메모리 사용량 변화를 추적하는 컨텍스트 매니저
        
        Args:
            tag: 추적 태그
        """
        before = self.report_memory_usage()
        
        try:
            yield
        finally:
            after = self.report_memory_usage()
            
            # 메모리 사용량 변화 계산
            diff = {
                'system_memory': after.system_memory - before.system_memory,
                'process_memory': after.process_memory - before.process_memory
            }
            
            if before.cuda_allocated is not None and after.cuda_allocated is not None:
                diff['cuda_allocated'] = after.cuda_allocated - before.cuda_allocated
            
            # 변화량 로깅
            self.logger.info(f"Memory usage change{f' ({tag})' if tag else ''}:")
            for key, value in diff.items():
                if value > 0:
                    self.logger.info(f"  {key}: +{value:.1f} MB")
                else:
                    self.logger.info(f"  {key}: {value:.1f} MB")
    
    def get_optimal_batch_size(
        self,
        sample_input_size: tuple,
        target_gpu_utilization: float = 0.8,
        safety_factor: float = 0.9
    ) -> int:
        """
        최적의 배치 크기 계산
        
        Args:
            sample_input_size: 샘플 입력 크기
            target_gpu_utilization: 목표 GPU 사용률 (0~1)
            safety_factor: 안전 계수 (0~1)
            
        Returns:
            계산된 최적 배치 크기
        """
        if not torch.cuda.is_available():
            return 32  # CPU 기본값
            
        try:
            # 테스트용 텐서 생성
            test_tensor = torch.randn(1, *sample_input_size, device='cuda')
            tensor_size = test_tensor.element_size() * test_tensor.nelement()
            
            # 가용 GPU 메모리 계산
            total_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = total_memory * target_gpu_utilization * safety_factor
            
            # 최적 배치 크기 계산
            optimal_batch_size = int(available_memory / tensor_size)
            
            # 메모리 정리
            del test_tensor
            torch.cuda.empty_cache()
            
            return max(1, min(optimal_batch_size, 1024))  # 상한선 설정
            
        except Exception as e:
            self.logger.warning(f"최적 배치 크기 계산 실패: {e}")
            return 32  # 기본값
    
    def log_memory_stats(self, tag: str = ""):
        """
        현재 메모리 상태 로깅
        
        Args:
            tag: 로깅 태그
        """
        stats = self.report_memory_usage()
        
        self.logger.info(f"Memory stats{f' ({tag})' if tag else ''}:")
        self.logger.info(f"  System Memory: {stats.system_memory}%")
        self.logger.info(f"  Process Memory: {stats.process_memory:.1f} MB")
        
        if stats.cuda_allocated is not None:
            self.logger.info(f"  CUDA Allocated: {stats.cuda_allocated:.1f} MB")
            self.logger.info(f"  CUDA Reserved: {stats.cuda_reserved:.1f} MB")
            self.logger.info(f"  Memory Fragmentation: {stats.fragmentation:.2%}")
        
        if stats.memory_leak_score > 0:
            self.logger.warning(f"  Memory Leak Score: {stats.memory_leak_score:.2f}")
    
    def get_optimization_suggestions(self) -> List[str]:
        """
        메모리 최적화 제안 반환
        
        Returns:
            최적화 제안 목록
        """
        stats = self.report_memory_usage()
        return self.optimizer.analyze_memory_usage(stats)
    
    def predict_memory_usage(self, time_window: timedelta) -> float:
        """
        미래 메모리 사용량 예측
        
        Args:
            time_window: 예측 시간 범위
            
        Returns:
            예측된 최대 메모리 사용량 (MB)
        """
        return self.predictor.predict_peak_memory(time_window)
    
    def start_profiling(self):
        """메모리 프로파일링 시작"""
        self.profiler.start_profiling()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """
        메모리 프로파일링 중지 및 결과 반환
        
        Returns:
            프로파일링 요약 정보
        """
        return self.profiler.get_profiling_summary()
    
    def cleanup(self):
        """리소스 정리"""
        self.clear_gpu_memory()
        gc.collect() 