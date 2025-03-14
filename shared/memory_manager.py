"""
메모리 관리 유틸리티

이 모듈은 GPU 및 시스템 메모리 관리를 위한 유틸리티를 제공합니다.
"""

import torch
import psutil
import gc
from functools import wraps
import logging
from typing import Dict, Any, Callable
from contextlib import contextmanager

# 로거 설정
logger = logging.getLogger(__name__)

class MemoryManager:
    """메모리 관리 유틸리티"""
    
    @staticmethod
    def clear_gpu_memory():
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    @staticmethod
    def report_memory_usage() -> Dict[str, Any]:
        """
        메모리 사용량 보고
        
        Returns:
            메모리 사용량 정보를 담은 딕셔너리
        """
        usage_info = {
            'cuda_allocated': None,
            'cuda_reserved': None,
            'system_memory': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            usage_info.update({
                'cuda_allocated': torch.cuda.memory_allocated() / 1024**2,
                'cuda_reserved': torch.cuda.memory_reserved() / 1024**2
            })
            
        return usage_info
    
    @staticmethod
    def memory_cleanup_decorator(func: Callable) -> Callable:
        """
        함수 실행 전후 메모리 정리를 수행하는 데코레이터
        
        Args:
            func: 대상 함수
            
        Returns:
            래핑된 함수
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            MemoryManager.clear_gpu_memory()
            gc.collect()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                MemoryManager.clear_gpu_memory()
                gc.collect()
                
        return wrapper
    
    @staticmethod
    @contextmanager
    def track_memory_usage(tag: str = ""):
        """
        메모리 사용량 변화를 추적하는 컨텍스트 매니저
        
        Args:
            tag: 추적 태그
        """
        before = MemoryManager.report_memory_usage()
        
        try:
            yield
        finally:
            after = MemoryManager.report_memory_usage()
            
            # 메모리 사용량 변화 계산
            diff = {}
            for key in before:
                if before[key] is not None and after[key] is not None:
                    diff[key] = after[key] - before[key]
            
            # 변화량 로깅
            logger.info(f"Memory usage change{f' ({tag})' if tag else ''}:")
            for key, value in diff.items():
                if value > 0:
                    logger.info(f"  {key}: +{value:.1f} MB")
                else:
                    logger.info(f"  {key}: {value:.1f} MB")
    
    @staticmethod
    def get_optimal_batch_size(
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
            logger.warning(f"최적 배치 크기 계산 실패: {e}")
            return 32  # 기본값
    
    @staticmethod
    def log_memory_stats(tag: str = ""):
        """
        현재 메모리 상태 로깅
        
        Args:
            tag: 로깅 태그
        """
        stats = MemoryManager.report_memory_usage()
        
        logger.info(f"Memory stats{f' ({tag})' if tag else ''}:")
        logger.info(f"  System Memory: {stats['system_memory']}%")
        
        if stats['cuda_allocated'] is not None:
            logger.info(f"  CUDA Allocated: {stats['cuda_allocated']:.1f} MB")
            logger.info(f"  CUDA Reserved: {stats['cuda_reserved']:.1f} MB") 