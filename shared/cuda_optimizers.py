"""
CUDA 최적화 및 메모리 관리 모듈

이 모듈은 GPU 가속, 메모리 최적화, 성능 모니터링을 위한
통합 CUDA 최적화 시스템을 구현합니다.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue
import psutil
from collections import deque
import torch.nn.functional as F

@dataclass
class MemoryStats:
    """메모리 통계 데이터 클래스"""
    allocated: List[int]
    cached: List[int]
    peak_allocated: int
    peak_cached: int
    fragmentation: float

@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    inference_time: List[float]
    memory_usage: List[int]
    batch_size: List[int]
    throughput: List[float]
    gpu_utilization: List[float]
    cpu_utilization: List[float]
    data_transfer_time: List[float]
    kernel_time: List[float]  # 커널 실행 시간
    queue_time: List[float]   # 큐잉 지연 시간

class TensorCache:
    """텐서 캐시 관리"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = {}
        self.usage_history = deque(maxlen=capacity)

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            self.usage_history.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, tensor: torch.Tensor):
        if len(self.cache) >= self.capacity:
            self._evict()
        self.cache[key] = tensor
        self.usage_history.append(key)

    def _evict(self):
        if not self.cache:
            return
        # LRU 정책으로 가장 오래된 항목 제거
        while self.usage_history and self.usage_history[0] not in self.cache:
            self.usage_history.popleft()
        if self.usage_history:
            del self.cache[self.usage_history.popleft()]

class InferenceBuffer:
    """추론 버퍼 관리"""
    def __init__(self, buffer_size: int = 8, prefetch_factor: int = 2):
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.input_queue = Queue(maxsize=buffer_size)
        self.output_queue = Queue(maxsize=buffer_size)
        self.prefetch_queue = Queue(maxsize=buffer_size * prefetch_factor)
        self.running = False
        self.threads = []
        self.tensor_cache = TensorCache()

    def start(self):
        self.running = True
        # 메인 처리 스레드
        self.threads.append(threading.Thread(target=self._process_buffer))
        # 프리페치 스레드
        self.threads.append(threading.Thread(target=self._prefetch_data))
        # 후처리 스레드
        self.threads.append(threading.Thread(target=self._postprocess_results))
        
        for thread in self.threads:
            thread.daemon = True
            thread.start()

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()

    def _process_buffer(self):
        while self.running:
            try:
                if not self.input_queue.empty():
                    batch = []
                    while len(batch) < self.buffer_size and not self.input_queue.empty():
                        data = self.input_queue.get_nowait()
                        # 캐시된 텐서 확인
                        cache_key = str(hash(data.tobytes()))
                        cached_tensor = self.tensor_cache.get(cache_key)
                        if cached_tensor is not None:
                            batch.append(cached_tensor)
                        else:
                            tensor = torch.as_tensor(data)
                            self.tensor_cache.put(cache_key, tensor)
                            batch.append(tensor)
                    if batch:
                        self.output_queue.put(batch)
            except Exception:
                continue
            time.sleep(0.001)

    def _prefetch_data(self):
        while self.running:
            try:
                if not self.prefetch_queue.empty():
                    data = self.prefetch_queue.get_nowait()
                    self.input_queue.put(data)
            except Exception:
                continue
            time.sleep(0.001)

    def _postprocess_results(self):
        while self.running:
            try:
                if not self.output_queue.empty():
                    batch = self.output_queue.get_nowait()
                    # 결과 후처리 로직
                    self._optimize_memory_layout(batch)
            except Exception:
                continue
            time.sleep(0.001)

    def _optimize_memory_layout(self, batch: List[torch.Tensor]):
        """메모리 레이아웃 최적화"""
        try:
            for i, tensor in enumerate(batch):
                # 연속적인 메모리 레이아웃으로 변환
                if not tensor.is_contiguous():
                    batch[i] = tensor.contiguous()
        except Exception:
            pass

class CUDAOptimizer:
    """
    통합 GPU 최적화기
    
    특징:
    1. 향상된 GPU 메모리 관리
    2. 동적 배치 크기 최적화
    3. 멀티 스트림 파이프라이닝
    4. 자동 혼합 정밀도(AMP) 최적화
    5. CUDA 그래프 캐싱
    6. 스마트 메모리 디프래그먼테이션
    7. 고급 성능 모니터링
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        최적화기 초기화
        
        Args:
            config: 최적화 설정
        """
        self.config = config or {}
        
        # GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        
        # RTX 4090 최적화 설정
        self._setup_rtx4090_optimization()
        
        # AMP 설정
        self.scaler = amp.GradScaler()
        
        # 스트림 관리
        self.streams = {}
        self.max_streams = self.config.get('max_streams', 4)
        
        # 텐서 캐시
        self.tensor_cache = TensorCache(
            capacity=self.config.get('tensor_cache_size', 1000)
        )
        
        # 메모리 관리
        self.memory_stats = MemoryStats(
            allocated=[],
            cached=[],
            peak_allocated=0,
            peak_cached=0,
            fragmentation=0.0
        )
        
        # 성능 모니터링
        self.performance_metrics = PerformanceMetrics(
            inference_time=[],
            memory_usage=[],
            batch_size=[],
            throughput=[],
            gpu_utilization=[],
            cpu_utilization=[],
            data_transfer_time=[],
            kernel_time=[],
            queue_time=[]
        )
        
        # 최적화 설정
        self.optimization_config = {
            'use_amp': self.config.get('use_amp', True),
            'use_cuda_graph': self.config.get('use_cuda_graph', True),
            'use_jit': self.config.get('use_jit', True),
            'use_channels_last': self.config.get('use_channels_last', True),
            'use_benchmark': self.config.get('use_benchmark', True),
            'memory_fraction': self.config.get('memory_fraction', 0.95),  # RTX 4090는 더 많은 메모리 사용 가능
            'defrag_threshold': self.config.get('defrag_threshold', 0.3),
            'cache_size': self.config.get('cache_size', 2048),  # 캐시 크기 증가
            'use_pinned_memory': self.config.get('use_pinned_memory', True),
            'use_inference_buffer': self.config.get('use_inference_buffer', True),
            'inference_buffer_size': self.config.get('inference_buffer_size', 16),  # 버퍼 크기 증가
            'warmup_iterations': self.config.get('warmup_iterations', 10),
            'use_kernel_fusion': self.config.get('use_kernel_fusion', True),
            'use_tensor_cores': self.config.get('use_tensor_cores', True),
            'prefetch_factor': self.config.get('prefetch_factor', 3),
            'use_async_copy': self.config.get('use_async_copy', True),
            'use_float16': self.config.get('use_float16', True),  # RTX 4090 FP16 지원
            'use_stream_reuse': self.config.get('use_stream_reuse', True)
        }
        
        # CUDA 그래프 캐시
        self.graph_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 추론 버퍼
        if self.optimization_config['use_inference_buffer']:
            self.inference_buffer = InferenceBuffer(
                buffer_size=self.optimization_config['inference_buffer_size'],
                prefetch_factor=self.optimization_config['prefetch_factor']
            )
            self.inference_buffer.start()
        
        # 로깅 설정
        self._setup_logging()
        
        # 초기 최적화
        self._setup_cuda_optimization()
    
    def _setup_logging(self):
        """로깅 설정"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "cuda_optimizer.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_cuda_optimization(self):
        """CUDA 초기 최적화 설정"""
        # 1. 메모리 관리 최적화
        torch.cuda.memory.set_per_process_memory_fraction(
            self.optimization_config['memory_fraction']
        )
        
        # 2. cuDNN 최적화
        if self.optimization_config['use_benchmark']:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 3. 캐시 초기화
        torch.cuda.empty_cache()
        
        # 4. 스트림 초기화
        self._initialize_streams()
        
        self.logger.info("CUDA 초기 최적화 완료")
    
    def _initialize_streams(self):
        """스트림 초기화 및 설정"""
        # 1. 기본 스트림
        self.streams['default'] = torch.cuda.default_stream()
        
        # 2. 추론 전용 스트림
        self.streams['inference'] = torch.cuda.Stream(priority=-1)
        
        # 3. 데이터 전송 스트림
        self.streams['data'] = torch.cuda.Stream(priority=0)
        
        # 4. 메모리 관리 스트림
        self.streams['memory'] = torch.cuda.Stream(priority=1)
        
        # 5. 보조 스트림
        for i in range(self.max_streams - 4):
            self.streams[f'auxiliary_{i}'] = torch.cuda.Stream()
    
    @contextmanager
    def autocast(self):
        """자동 혼합 정밀도 컨텍스트"""
        with amp.autocast(enabled=self.optimization_config['use_amp']):
            yield
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        모델 최적화
        
        Args:
            model: 최적화할 PyTorch 모델
            
        Returns:
            최적화된 모델
        """
        try:
            self.logger.info("모델 최적화 시작")
            
            # 1. GPU 이동
            model = model.to(self.device)
            
            # 2. 메모리 최적화
            self._optimize_memory(model)
            
            # 3. 모델 최적화
            if self.optimization_config['use_channels_last']:
                model = model.to(memory_format=torch.channels_last)
            
            if self.optimization_config['use_jit']:
                model = self._optimize_jit(model)
            
            # 4. CUDA 그래프 준비
            if self.optimization_config['use_cuda_graph']:
                self._prepare_cuda_graph(model)
            
            self.logger.info("모델 최적화 완료")
            return model
            
        except Exception as e:
            self.logger.error(f"모델 최적화 중 오류 발생: {str(e)}")
            raise
    
    def _optimize_memory(self, model: nn.Module):
        """
        메모리 최적화
        
        특징:
        1. 스마트 메모리 디프래그먼테이션
        2. 캐시 최적화
        3. 메모리 누수 방지
        """
        try:
            # 1. 현재 메모리 상태 확인
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            # 2. 메모리 단편화 계산
            if reserved > 0:
                fragmentation = 1 - (allocated / reserved)
                self.memory_stats.fragmentation = fragmentation
                
                # 3. 단편화가 심각한 경우 디프래그먼테이션
                if fragmentation > self.optimization_config['defrag_threshold']:
                    self.logger.info(f"메모리 디프래그먼테이션 실행 (단편화: {fragmentation:.2%})")
                    with torch.cuda.stream(self.streams['memory']):
                        torch.cuda.empty_cache()
                        torch.cuda.memory.empty_cache()
            
            # 4. 그래디언트 메모리 최적화
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            
            # 5. 캐시 정리
            if len(self.graph_cache) > self.optimization_config['cache_size']:
                self._clean_graph_cache()
            
            # 6. 메모리 통계 업데이트
            self._update_memory_stats()
            
        except Exception as e:
            self.logger.error(f"메모리 최적화 중 오류 발생: {str(e)}")
            raise
    
    def _clean_graph_cache(self):
        """CUDA 그래프 캐시 정리"""
        # LRU 캐시 정리
        if len(self.graph_cache) > self.optimization_config['cache_size']:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.graph_cache))
            del self.graph_cache[oldest_key]
            torch.cuda.empty_cache()
    
    def _prepare_cuda_graph(self, model: nn.Module):
        """CUDA 그래프 준비"""
        try:
            # 1. 워밍업
            dummy_input = torch.randn(1, model.input_size, device=self.device)
            with torch.cuda.stream(self.streams['default']):
                model(dummy_input)
            
            # 2. 그래프 캡처
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = model(dummy_input)
            
            # 3. 캐시 저장
            self.graph_cache[model.__class__.__name__] = {
                'graph': g,
                'input': dummy_input,
                'output': static_output,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"CUDA 그래프 준비 중 오류 발생: {str(e)}")
            self.optimization_config['use_cuda_graph'] = False
    
    def optimize_inference(self, model: nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        try:
            start_time = time.time()
            queue_start = time.time()
            
            # 1. 큐잉 시간 측정
            if self.optimization_config['use_inference_buffer']:
                self.inference_buffer.prefetch_queue.put(input_data)
                while self.inference_buffer.input_queue.empty():
                    time.sleep(0.001)
                input_data = self.inference_buffer.input_queue.get()
            queue_time = time.time() - queue_start
            
            transfer_start = time.time()
            
            # 2. 데이터 전송 최적화
            if self.optimization_config['use_pinned_memory']:
                with torch.cuda.stream(self.streams['data']):
                    if not input_data.is_pinned() and self.optimization_config['use_async_copy']:
                        input_data = input_data.pin_memory()
                    input_data = input_data.to(self.device, non_blocking=True)
            
            transfer_time = time.time() - transfer_start
            
            # 3. 커널 실행 시간 측정 시작
            kernel_start = time.time()
            
            # 4. FP16 변환 (RTX 4090 최적화)
            if self.optimization_config['use_float16']:
                input_data = input_data.to(torch.float16)
            
            # 5. 추론 실행
            output = self._optimized_inference(model, input_data)
            
            kernel_time = time.time() - kernel_start
            
            # 6. 성능 메트릭 업데이트
            inference_time = time.time() - start_time
            self._update_performance_metrics(
                inference_time=inference_time,
                batch_size=input_data.size(0),
                transfer_time=transfer_time,
                kernel_time=kernel_time,
                queue_time=queue_time
            )
            
            return output
            
        except Exception as e:
            self.logger.error(f"추론 최적화 중 오류 발생: {str(e)}")
            raise

    def _optimized_inference(self, model: nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """최적화된 추론 처리"""
        try:
            # 1. CUDA 그래프 캐시 확인
            cache_key = model.__class__.__name__
            if (self.optimization_config['use_cuda_graph'] and 
                cache_key in self.graph_cache and 
                input_data.shape == self.graph_cache[cache_key]['input'].shape):
                
                self.cache_hits += 1
                cached = self.graph_cache[cache_key]
                
                with torch.cuda.stream(self.streams['inference']):
                    cached['input'].copy_(input_data)
                    cached['graph'].replay()
                    output = cached['output']
            
            else:
                self.cache_misses += 1
                
                # 2. 커널 융합 최적화
                if self.optimization_config['use_kernel_fusion']:
                    input_data = self._fuse_operations(input_data)
                
                # 3. 추론 실행
                with torch.cuda.stream(self.streams['inference']):
                    with torch.no_grad(), self.autocast():
                        if not hasattr(model, '_warmed_up'):
                            self._warmup_model(model, input_data)
                            model._warmed_up = True
                        output = model(input_data)
            
            return output
            
        except Exception as e:
            self.logger.error(f"최적화된 추론 중 오류 발생: {str(e)}")
            raise

    def _fuse_operations(self, input_data: torch.Tensor) -> torch.Tensor:
        """커널 융합 최적화"""
        try:
            # 연속적인 연산들을 하나의 커널로 융합
            with torch.cuda.stream(self.streams['inference']):
                # 예: 정규화 + 활성화 함수를 하나의 커널로
                if input_data.dtype == torch.float32:
                    input_data = F.layer_norm(input_data, input_data.shape[1:])
                    input_data = F.relu(input_data)
            return input_data
        except Exception as e:
            self.logger.error(f"커널 융합 최적화 중 오류 발생: {str(e)}")
            return input_data

    def _update_performance_metrics(self, inference_time: float, batch_size: int,
                                 transfer_time: float, kernel_time: float, queue_time: float):
        """성능 메트릭 업데이트"""
        try:
            # 1. 기본 메트릭
            self.performance_metrics.inference_time.append(inference_time)
            self.performance_metrics.batch_size.append(batch_size)
            self.performance_metrics.memory_usage.append(torch.cuda.memory_allocated())
            self.performance_metrics.data_transfer_time.append(transfer_time)
            self.performance_metrics.kernel_time.append(kernel_time)
            self.performance_metrics.queue_time.append(queue_time)
            
            # 2. 처리량 계산 (순수 연산 시간 기준)
            compute_time = inference_time - transfer_time - queue_time
            throughput = batch_size / compute_time if compute_time > 0 else 0
            self.performance_metrics.throughput.append(throughput)
            
            # 3. GPU 활용도
            if hasattr(torch.cuda, 'utilization'):
                gpu_util = torch.cuda.utilization()
            else:
                gpu_util = 0
            self.performance_metrics.gpu_utilization.append(gpu_util)
            
            # 4. CPU 사용률
            self.performance_metrics.cpu_utilization.append(psutil.cpu_percent())
            
            # 5. 성능 로깅
            if len(self.performance_metrics.inference_time) % 100 == 0:
                self.logger.info(
                    f"성능 메트릭:\n"
                    f"- 전체 추론 시간: {inference_time*1000:.2f}ms\n"
                    f"- 큐잉 시간: {queue_time*1000:.2f}ms\n"
                    f"- 데이터 전송 시간: {transfer_time*1000:.2f}ms\n"
                    f"- 커널 실행 시간: {kernel_time*1000:.2f}ms\n"
                    f"- 처리량: {throughput:.2f} samples/s\n"
                    f"- GPU 활용도: {gpu_util:.1f}%\n"
                    f"- 캐시 효율성: {self.cache_hits/(self.cache_hits+self.cache_misses):.2%}"
                )
            
        except Exception as e:
            self.logger.error(f"성능 메트릭 업데이트 중 오류 발생: {str(e)}")
    
    def _update_memory_stats(self):
        """메모리 통계 업데이트"""
        try:
            # 1. 현재 메모리 상태
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            
            # 2. 통계 업데이트
            self.memory_stats.allocated.append(allocated)
            self.memory_stats.cached.append(cached)
            self.memory_stats.peak_allocated = max(self.memory_stats.peak_allocated, allocated)
            self.memory_stats.peak_cached = max(self.memory_stats.peak_cached, cached)
            
            # 3. 주기적 로깅
            if len(self.memory_stats.allocated) % 100 == 0:
                self.logger.info(
                    f"메모리 사용량: {allocated/1e6:.1f}MB (캐시: {cached/1e6:.1f}MB, "
                    f"단편화: {self.memory_stats.fragmentation:.1%})"
                )
                
        except Exception as e:
            self.logger.error(f"메모리 통계 업데이트 중 오류 발생: {str(e)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        return {
            'avg_inference_time': np.mean(self.performance_metrics.inference_time),
            'avg_throughput': np.mean(self.performance_metrics.throughput),
            'avg_gpu_utilization': np.mean(self.performance_metrics.gpu_utilization),
            'memory_stats': {
                'current_allocated': torch.cuda.memory_allocated(),
                'peak_allocated': self.memory_stats.peak_allocated,
                'fragmentation': self.memory_stats.fragmentation
            },
            'cache_efficiency': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0 else 0
            }
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 1. 스트림 동기화
            torch.cuda.synchronize()
            
            # 2. 캐시 정리
            self.graph_cache.clear()
            torch.cuda.empty_cache()
            
            # 3. 메모리 정리
            torch.cuda.memory.empty_cache()
            
            # 4. 추론 버퍼 정리
            if hasattr(self, 'inference_buffer'):
                self.inference_buffer.stop()
            
            self.logger.info("CUDA 최적화기 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 중 오류 발생: {str(e)}")
            raise

    def _setup_rtx4090_optimization(self):
        """RTX 4090 특화 최적화 설정"""
        try:
            # 1. GPU 정보 확인
            gpu_name = torch.cuda.get_device_name()
            if 'RTX 4090' in gpu_name:
                self.logger.info("RTX 4090 감지됨, 특화 최적화 적용")
                
                # 2. RTX 4090 특화 설정
                torch.backends.cuda.matmul.allow_tf32 = True  # TF32 활성화
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # 3. CUDA 캐시 워밍업
                self._warmup_cuda_cache()
                
                # 4. RTX 4090 메모리 최적화
                torch.cuda.set_per_process_memory_fraction(0.95)  # 더 많은 메모리 사용
                
        except Exception as e:
            self.logger.error(f"RTX 4090 최적화 설정 중 오류 발생: {str(e)}")

    def _warmup_cuda_cache(self):
        """CUDA 캐시 워밍업"""
        try:
            # 1. 더미 텐서로 캐시 워밍업
            dummy = torch.randn(1024, 1024, device=self.device)
            torch.mm(dummy, dummy)
            torch.cuda.synchronize()
            
            # 2. 캐시 정리
            del dummy
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"CUDA 캐시 워밍업 중 오류 발생: {str(e)}")

    def _warmup_model(self, model: nn.Module, input_data: torch.Tensor):
        """모델 워밍업"""
        try:
            self.logger.info("모델 워밍업 시작")
            warmup_iterations = self.optimization_config['warmup_iterations']
            
            with torch.cuda.stream(self.streams['inference']):
                with torch.no_grad(), self.autocast():
                    for _ in range(warmup_iterations):
                        model(input_data)
            
            torch.cuda.synchronize()
            self.logger.info(f"모델 워밍업 완료 ({warmup_iterations} 반복)")
            
        except Exception as e:
            self.logger.error(f"모델 워밍업 중 오류 발생: {str(e)}")
            raise 