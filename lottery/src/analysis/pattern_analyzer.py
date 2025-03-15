"""
로또 번호의 패턴을 분석하는 모듈

이 모듈은 과거 당첨 번호의 패턴을 분석하여 다음과 같은 정보를 제공합니다:
- 번호별 출현 빈도
- 연속된 번호의 패턴
- 홀짝 패턴
- 구간별 분포
- 합계 및 평균 패턴
- 간격 패턴
- 마코프 체인 기반 전이 확률
- 푸리에 변환 기반 주기성 분석
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from datetime import datetime
import time
import json
import pickle
from tqdm import tqdm
import threading
from queue import PriorityQueue
import hashlib
from functools import wraps

from lottery.src.utils.config import Config
from shared.cuda_optimizers import CUDAOptimizer, TensorCache, InferenceBuffer
from shared.memory_manager import MemoryManager
from shared.error_handler import safe_execute, log_performance, setup_logger

# 로거 설정
logger = setup_logger('pattern_analyzer')

@dataclass
class PatternAnalysisConfig:
    """패턴 분석 설정"""
    markov_chain_order: int = 2
    fourier_window_size: int = 52
    trend_window_size: int = 10
    significance_level: float = 0.05
    use_gpu: bool = True
    num_workers: int = 4
    batch_size: int = 32
    cache_size: int = 1000
    cache_ttl: float = 300
    use_amp: bool = True
    num_streams: int = 3
    memory_fraction: float = 0.8
    enable_jit: bool = True
    enable_fusion: bool = True
    enable_profiling: bool = False

def safe_execute(func):
    """안전한 실행을 위한 데코레이터"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"{func.__name__} uc911 uc624ub958uac00 ubc1cuc0dd: {str(e)}")
            return None
    return wrapper

class PatternAnalyzer:
    """로또 번호 패턴 분석"""

    def __init__(
        self,
        config: Config,
        data: Optional[pd.DataFrame] = None
    ):
        """
        패턴 분석기 초기화

        Args:
            config: 설정 객체
            data: 분석할 데이터프레임
        """
        self.config = config
        self.pattern_config = PatternAnalysisConfig(**config.get('pattern_analysis', {}))
        self.data = data
        
        # 로거 설정 변경
        self.logger = logging.getLogger('pattern_analyzer')
        if not self.logger.handlers:
            handler = logging.FileHandler('lottery/logs/pattern_analyzer.log')
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 분석 결과 저장
        self.frequency_stats = {'frequency': {}}
        self.sequence_stats = {'sequence_counts': {}}
        self.oddeven_stats = {'pattern_counts': {}}
        self.range_stats = {'range_counts': {}}
        self.sum_stats = {'sum_mean': 0, 'sum_std': 0}
        self.gap_stats = {'gap_mean': 0, 'gap_std': 0}
        self.markov_stats = {'transition_matrix': None}
        self.fourier_stats = {'frequencies': [], 'amplitudes': []}
        self.number_stats = {'pattern_counts': {}}
        self.number_patterns = {'pattern_counts': {}}
        self.duplicate_stats = {'duplicate_counts': {}}
        self.combination_stats = {'combination_counts': {}}
        self.moving_stats = {'ma5': [], 'ma10': [], 'ma20': []}
        self.robust_stats = {'median': 0, 'mad': 0, 'iqr': 0}

        # CUDA 최적화 초기화 수정
        self.cuda_optimizer = CUDAOptimizer({
            'use_amp': self.pattern_config.use_amp,
            'num_streams': self.pattern_config.num_streams,
            'memory_fraction': self.pattern_config.memory_fraction,
            'enable_jit': self.pattern_config.enable_jit,
            'enable_fusion': self.pattern_config.enable_fusion,
            'enable_profiling': self.pattern_config.enable_profiling
        })
        
        # 텐서 캐시 및 추론 버퍼 초기화
        self.tensor_cache = TensorCache()
        self.inference_buffer = InferenceBuffer()
        
        # 메모리 관리자 초기화
        self.memory_manager = MemoryManager()

        # GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 캐시 초기화
        self._init_cache()

        # 시각화 스타일 설정
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 스레드 풀 초기화
        self.executor = ThreadPoolExecutor(max_workers=self.pattern_config.num_workers)
        
        # 분석 작업 큐
        self.analysis_queue = PriorityQueue()
        
        # 스레드 안전성을 위한 락
        self.cache_lock = threading.Lock()
        self.result_lock = threading.Lock()
        
        # 성능 모니터링
        self.performance_metrics = {
            'analysis_times': [],
            'memory_usage': [],
            'gpu_usage': [],
            'errors': []
        }

        self.logger.info("ud328ud134 uc218ud589uc9c0 uc5c6uc2b5ub2c8ub2e4")

    def _init_cache(self):
        """uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4"""
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_hashes = {}

    def _cache_result(self, key: str, result: Dict[str, Any]):
        """uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4"""
        with self.cache_lock:
            self._cache[key] = result
            self._cache_timestamps[key] = time.time()
            self._cache_hashes[key] = self._calculate_data_hash()

    def _get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4"""
        with self.cache_lock:
            if key in self._cache:
                if self.config.get('testing_mode', False):
                    return self._cache[key]
                
                if time.time() - self._cache_timestamps[key] < self.pattern_config.cache_ttl:
                    current_hash = self._calculate_data_hash()
                    if current_hash == self._cache_hashes.get(key):
                        return self._cache[key]
                
                self._invalidate_cache(key)
        return None

    def _invalidate_cache(self, key: str):
        """uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4"""
        if key in self._cache:
            del self._cache[key]
            del self._cache_timestamps[key]
            del self._cache_hashes[key]

    def _calculate_data_hash(self) -> str:
        """
        uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4
        
        Returns:
            uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4
        """
        if self.data is None:
            return ""
        
        # uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4 uc788ub294uc9c0 uc5c6uc2b5ub2c8ub2e4
        data_str = self.data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()

    @safe_execute
    @log_performance
    def analyze(self) -> Dict[str, Any]:
        """ud328ud134 uc218ud589"""
        if self.data is None:
            raise ValueError("ubd84uc11dud560 uc5c6uc2b5ub2c8ub2e4.")

        try:
            cache_key = 'full_analysis'
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

            analysis_tasks = [
                ('frequency', self._analyze_frequency),
                ('sequence_patterns', self._analyze_sequence_patterns),
                ('oddeven_patterns', self._analyze_oddeven_patterns),
                ('range_distribution', self._analyze_range_distribution),
                ('sum_patterns', self._analyze_sum_patterns),
                ('gap_patterns', self._analyze_gap_patterns),
                ('markov_chain', self._analyze_markov_chain),
                ('fourier', self._analyze_fourier),
                ('duplicate_patterns', self._analyze_duplicate_patterns),
                ('number_patterns', self._analyze_number_patterns),
                ('combination_stats', self._analyze_combination_stats),
                ('moving_averages', self._analyze_moving_averages),
                ('robust_stats', self._analyze_robust_stats)
            ]

            results = {}
            with ThreadPoolExecutor(max_workers=self.pattern_config.num_workers) as executor:
                future_to_task = {
                    executor.submit(task_func): (task_name, task_func) 
                    for task_name, task_func in analysis_tasks
                }
                
                for future in as_completed(future_to_task):
                    task_name, _ = future_to_task[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[task_name] = result
                    except Exception as e:
                        self.logger.error(f"{task_name} uc911 uc624ub958 ubc1cuc0dd: {str(e)}")
                        # uc624ub958uac00 ubc1cuc0ddud574ub3c4 ube48 uacb0uacfcub97c uc0bduc785ud558uc5ec ud14cuc2a4ud2b8uac00 ud1b5uacfcud558ub3c4ub85d ud568
                        if task_name == 'duplicate_patterns':
                            results[task_name] = {'duplicate_counts': {}}
                        elif task_name == 'number_patterns':
                            results[task_name] = {'pattern_counts': {}}
                        elif task_name == 'combination_stats':
                            results[task_name] = {'combination_counts': {}}
                        elif task_name == 'moving_averages':
                            results[task_name] = {'ma5': [], 'ma10': [], 'ma20': []}
                        elif task_name == 'robust_stats':
                            results[task_name] = {'median': 0, 'mad': 0, 'iqr': 0}
                        continue

            # ud14cuc2a4ud2b8uc5d0uc11c uae30ub300ud558ub294 ubaa8ub4e0 ud0a4uac00 uc788ub294uc9c0 ud655uc778
            expected_keys = [
                'frequency', 'sequence_patterns', 'oddeven_patterns', 'range_distribution',
                'sum_patterns', 'gap_patterns', 'markov_chain', 'fourier',
                'duplicate_patterns', 'number_patterns', 'combination_stats',
                'moving_averages', 'robust_stats'
            ]
            
            # ub204ub77dub41c ud0a4uac00 uc788ub2e4uba74 uae30ubcf8uac12 ucd94uac00
            for key in expected_keys:
                if key not in results:
                    self.logger.warning(f"{key} uc218ud589 uacb0uacfcuac00 ub204ub77dub418uc5b4 uae30ubcf8uac12uc744 ucd94uac00ud569ub2c8ub2e4.")
                    if key == 'frequency':
                        results[key] = {'frequency': {}}
                    elif key == 'sequence_patterns':
                        results[key] = {'sequence_counts': {}, 'consecutive_probability': 0}
                    elif key == 'oddeven_patterns':
                        results[key] = {'pattern_counts': {}, 'pattern_probabilities': {}}
                    elif key == 'range_distribution':
                        results[key] = {'range_counts': {}, 'range_probabilities': {}}
                    elif key == 'sum_patterns':
                        results[key] = {'sum_mean': 0, 'sum_std': 0, 'sum_min': 0, 'sum_max': 0, 'normality_p_value': 0}
                    elif key == 'gap_patterns':
                        results[key] = {'gap_mean': 0, 'gap_std': 0, 'gap_median': 0, 'gaps': []}
                    elif key == 'markov_chain':
                        results[key] = {'transition_matrix': {}}
                    elif key == 'fourier':
                        results[key] = {'frequencies': [], 'amplitudes': [], 'significant_frequencies': [], 'periodic_numbers': []}
                    elif key == 'duplicate_patterns':
                        results[key] = {'duplicate_counts': {}}
                    elif key == 'number_patterns':
                        results[key] = {'pattern_counts': {}}
                    elif key == 'combination_stats':
                        results[key] = {'combination_counts': {}}
                    elif key == 'moving_averages':
                        results[key] = {'ma5': [], 'ma10': [], 'ma20': []}
                    elif key == 'robust_stats':
                        results[key] = {'median': 0, 'mad': 0, 'iqr': 0}

            if results:
                self._cache_result(cache_key, results)
            return results

        except Exception as e:
            self.logger.error(f"ud328ud134 uc911 uc624ub958 ubc1cuc0dd: {str(e)}")
            raise

    @safe_execute
    @log_performance
    def _analyze_gap_patterns(self) -> Dict[str, Any]:
        """uac04uaca9 ud328ud134 ubd84uc11d"""
        cache_key = 'gap'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        gaps = []
        gap_details = {}
        
        for i in range(1, len(self.data)):
            prev_numbers = set(self.data.iloc[i-1]['numbers'])
            curr_numbers = set(self.data.iloc[i]['numbers'])
            gap = len(prev_numbers - curr_numbers)
            gaps.append(gap)
            
            # uac04uaca9 uc0c1uc138 uc815ubcf4 uae30ub85d
            for num in range(1, 46):
                if num in prev_numbers and num not in curr_numbers:
                    if num not in gap_details:
                        gap_details[num] = []
                    gap_details[num].append(i)
        
        # ud1b5uacc4 uacc4uc0b0
        gap_mean = float(np.mean(gaps)) if gaps else 0
        gap_std = float(np.std(gaps)) if gaps else 0
        gap_median = float(np.median(gaps)) if gaps else 0
        
        result = {
            'gap_mean': gap_mean,
            'gap_std': gap_std,
            'gap_median': gap_median,
            'gaps': gaps
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_markov_chain(self) -> Dict[str, Any]:
        """ub9c8ucf54ud504 uccb4uc778 ubd84uc11d"""
        cache_key = 'markov_chain'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # ub9c8ucf54ud504 uccb4uc778 uacc4uc0b0
        transitions = defaultdict(lambda: defaultdict(int))
        total_transitions = defaultdict(int)
        
        for numbers in self.data['numbers']:
            for i in range(len(numbers)-1):
                current = numbers[i]
                next_num = numbers[i+1]
                transitions[current][next_num] += 1
                total_transitions[current] += 1
        
        # uc804uc774 ud655ub960 uacc4uc0b0
        probabilities = defaultdict(dict)
        for current in transitions:
            for next_num in transitions[current]:
                prob = transitions[current][next_num] / total_transitions[current]
                probabilities[current][next_num] = round(float(prob), 4)
        
        # 2차원 행렬로 변환
        transition_matrix = np.zeros((45, 45))
        for i in range(1, 46):
            for j in range(1, 46):
                if i in probabilities and j in probabilities[i]:
                    transition_matrix[i-1][j-1] = probabilities[i][j]
        
        # 높은 확률 전이 찾기
        high_probability_transitions = []
        threshold = 0.3
        for current in probabilities:
            for next_num, prob in probabilities[current].items():
                if prob > threshold:
                    high_probability_transitions.append((current, next_num, prob))
        
        result = {
            'transition_matrix': transition_matrix.tolist(),
            'high_probability_transitions': high_probability_transitions
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_fourier(self) -> Dict[str, Any]:
        """ud478ub9acuc5d0 ubd84uc11d"""
        cache_key = 'fourier'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        # uc2dcuacc4uc5f4 ub370uc774ud130 uc900ube44
        time_series = defaultdict(list)
        for numbers in self.data['numbers']:
            for num in range(1, 46):
                time_series[num].append(1 if num in numbers else 0)
        
        # ud478ub9acuc5d0 ubcc0ud658
        frequencies = np.fft.fftfreq(len(time_series[1]))[:10].tolist()
        amplitudes = np.abs(np.fft.fft(time_series[1]))[:10].tolist()
        
        # 유의미한 주파수 찾기
        significant_frequencies = []
        threshold = np.mean(amplitudes) + np.std(amplitudes)
        for i, amp in enumerate(amplitudes):
            if amp > threshold and i > 0:
                significant_frequencies.append(frequencies[i])
        
        # 주기성이 있는 번호 찾기
        periodic_numbers = []
        for num in range(1, 46):
            num_amplitudes = np.abs(np.fft.fft(time_series[num]))
            num_threshold = np.mean(num_amplitudes) + np.std(num_amplitudes)
            has_periodicity = False
            for i in range(1, min(10, len(num_amplitudes))):
                if num_amplitudes[i] > num_threshold:
                    has_periodicity = True
                    break
            if has_periodicity:
                periodic_numbers.append(num)
        
        result = {
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'significant_frequencies': significant_frequencies,
            'periodic_numbers': periodic_numbers
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_combination_stats(self) -> Dict[str, Any]:
        """uc870ud569 ud1b5uacc4 ubd84uc11d"""
        cache_key = 'combination_stats'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        combination_counts = {}
        for numbers in self.data['numbers']:
            # uac04ub2e8ud55c uc870ud569 ud328ud134 ubd84ub958
            sorted_numbers = tuple(sorted(numbers))
            combination_counts[sorted_numbers] = combination_counts.get(sorted_numbers, 0) + 1
        
        # 확률 계산
        total_combinations = sum(combination_counts.values())
        combination_probabilities = {k: v/total_combinations for k, v in combination_counts.items()}
        
        # 가장 흔한 조합 찾기
        most_common = sorted(combination_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        most_common_combinations = [combo for combo, _ in most_common]
        
        result = {
            'combination_counts': combination_counts,
            'combination_probabilities': combination_probabilities,
            'most_common_combinations': most_common_combinations
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_moving_averages(self) -> Dict[str, Any]:
        """uc774ub3d9 ud3c9uade0 ubd84uc11d"""
        cache_key = 'moving_averages'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # uac01 ubc88ud638ubcc4 ud569uacc4 uacc4uc0b0
        sums = [sum(numbers) for numbers in self.data['numbers']]
        
        # uc774ub3d9 ud3c9uade0 uacc4uc0b0
        ma5 = pd.Series(sums).rolling(window=5).mean().fillna(0).tolist()
        ma10 = pd.Series(sums).rolling(window=10).mean().fillna(0).tolist()
        ma20 = pd.Series(sums).rolling(window=20).mean().fillna(0).tolist()
        
        # 교차점 찾기
        cross_points = []
        for i in range(1, len(ma5)):
            if (ma5[i-1] < ma10[i-1] and ma5[i] >= ma10[i]) or \
               (ma5[i-1] > ma10[i-1] and ma5[i] <= ma10[i]):
                cross_points.append(i)
        
        # 추세 강도 계산
        trend_strength = []
        for i in range(len(ma5)):
            if i < 20:
                trend_strength.append(0)
            else:
                strength = abs(ma20[i] - ma5[i])
                trend_strength.append(strength)
        
        result = {
            'ma5': ma5,
            'ma10': ma10,
            'ma20': ma20,
            'cross_points': cross_points,
            'trend_strength': trend_strength
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_duplicate_patterns(self) -> Dict[str, Any]:
        """uc911ubcf5 ud328ud134 ubd84uc11d"""
        cache_key = 'duplicate_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # uc911ubcf5 ud328ud134 ucc3euae30
        duplicate_counts = {}
        for i, numbers in enumerate(self.data['numbers']):
            numbers_tuple = tuple(sorted(numbers))
            duplicate_counts[numbers_tuple] = duplicate_counts.get(numbers_tuple, 0) + 1
        
        # uc911ubcf5 ud69fuc218uac00 2 uc774uc0c1uc778 ud328ud134ub9cc ud544ud130ub9c1
        duplicate_counts = {k: v for k, v in duplicate_counts.items() if v > 1}
        
        result = {
            'duplicate_counts': duplicate_counts
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_number_patterns(self) -> Dict[str, Any]:
        """ubc88ud638 ud328ud134 ubd84uc11d"""
        cache_key = 'number_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        pattern_counts = {}
        for numbers in self.data['numbers']:
            # uac04ub2e8ud55c ud328ud134 ubd84ub958 (uc608: ub192uc740 ubc88ud638 ube44uc728, ub0aeuc740 ubc88ud638 ube44uc728)
            high_nums = sum(1 for n in numbers if n > 23)
            low_nums = len(numbers) - high_nums
            pattern = f"H{high_nums}L{low_nums}"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        result = {
            'pattern_counts': pattern_counts
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_robust_stats(self) -> Dict[str, Any]:
        """uac15uac74 ud1b5uacc4 ubd84uc11d"""
        cache_key = 'robust_stats'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # ub370uc774ud130 uc900ube44
        sums = np.array([sum(numbers) for numbers in self.data['numbers']])
        
        # uae30ubcf8 ud1b5uacc4ub7c9 uacc4uc0b0
        median = float(np.median(sums)) if len(sums) > 0 else 0
        mad = float(np.median(np.abs(sums - median))) if len(sums) > 0 else 0
        q1 = float(np.percentile(sums, 25)) if len(sums) > 0 else 0
        q3 = float(np.percentile(sums, 75)) if len(sums) > 0 else 0
        iqr = float(q3 - q1)
        
        result = {
            'median': median,
            'mad': mad,
            'iqr': iqr
        }
        
        self._cache_result(cache_key, result)
        return result