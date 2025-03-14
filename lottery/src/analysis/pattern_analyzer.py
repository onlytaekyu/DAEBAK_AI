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
from concurrent.futures import ThreadPoolExecutor
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
            self.logger.error(f"{func.__name__} 실행 중 오류 발생: {str(e)}")
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
        self.logger = setup_logger('pattern_analyzer')

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

        # CUDA 최적화 초기화
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
        self.device = self.cuda_optimizer.device

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

        self.logger.info("패턴 분석기 초기화 완료")

    def _init_cache(self):
        """캐시 초기화"""
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_hashes = {}

    @safe_execute
    def _get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """
        캐시된 결과 조회

        Args:
            key: 캐시 키

        Returns:
            캐시된 결과 또는 None
        """
        with self.cache_lock:
            if key in self._cache:
                # 캐시 유효 시간 확인
                if time.time() - self._cache_timestamps[key] < self.pattern_config.cache_ttl:
                    # 데이터 해시 확인
                    current_hash = self._calculate_data_hash()
                    if current_hash == self._cache_hashes.get(key):
                        logger.debug(f"캐시된 결과 사용: {key}")
                        return self._cache[key]
                    else:
                        # 데이터가 변경된 경우 캐시 무효화
                        del self._cache[key]
                        del self._cache_timestamps[key]
                        del self._cache_hashes[key]
                else:
                    # 캐시 만료
                    del self._cache[key]
                    del self._cache_timestamps[key]
                    del self._cache_hashes[key]
        return None

    def _calculate_data_hash(self) -> str:
        """
        데이터 해시 계산
        
        Returns:
            데이터 해시값
        """
        if self.data is None:
            return ""
        
        # 데이터프레임을 문자열로 변환하여 해시 계산
        data_str = self.data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()

    @safe_execute
    def _cache_result(self, key: str, result: Dict[str, Any]):
        """
        결과 캐싱

        Args:
            key: 캐시 키
            result: 캐시할 결과
        """
        with self.cache_lock:
            self._cache[key] = result
            self._cache_timestamps[key] = time.time()
            self._cache_hashes[key] = self._calculate_data_hash()
            logger.debug(f"결과 캐싱 완료: {key}")

    @safe_execute
    @log_performance
    def analyze(self) -> Dict[str, Any]:
        """
        패턴 분석 수행

        Returns:
            분석 결과 딕셔너리
        """
        if self.data is None:
            raise ValueError("분석할 데이터가 없습니다.")

        try:
            # 캐시 확인
            cache_key = 'full_analysis'
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.info("캐시된 분석 결과 사용")
                return cached_result

            # 분석 작업 실행
            results = {}
            analysis_tasks = [
                ('frequency', self._analyze_frequency),
                ('sequence', self._analyze_sequence_patterns),
                ('oddeven', self._analyze_oddeven_patterns),
                ('range', self._analyze_range_distribution),
                ('sum', self._analyze_sum_patterns),
                ('gap', self._analyze_gap_patterns),
                ('markov', self._analyze_markov_chain),
                ('fourier', self._analyze_fourier),
                ('duplicate', self._analyze_duplicate_patterns),
                ('number', self._analyze_number_patterns),
                ('combination', self._analyze_combination_stats),
                ('moving', self._analyze_moving_averages),
                ('robust', self._analyze_robust_stats)
            ]

            # 작업 실행
            for task_name, task_func in analysis_tasks:
                try:
                    result = task_func()
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"분석 작업 실패: {task_name}, 오류: {e}")

            # 결과 캐싱
            self._cache_result(cache_key, results)
            logger.info("패턴 분석 완료")
            return results

        except Exception as e:
            logger.error(f"패턴 분석 실패: {str(e)}")
            raise

    @safe_execute
    @log_performance
    def _analyze_frequency(self) -> Dict[str, Any]:
        """빈도 분석"""
        cache_key = 'frequency'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        frequency = {}
        for numbers in self.data['numbers']:
            for num in numbers:
                frequency[num] = frequency.get(num, 0) + 1
                
        # 확률 계산
        total_count = sum(frequency.values())
        probabilities = {num: count/total_count for num, count in frequency.items()}
        
        # 카이제곱 검정
        expected_freq = total_count / 45  # 균등 분포 가정
        chi2_stat = sum((count - expected_freq)**2 / expected_freq for count in frequency.values())
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=44)  # 자유도 = 45-1
        
        result = {
            'frequency': frequency,
            'probabilities': probabilities,
            'chi2_stat': chi2_stat,
            'p_value': p_value
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_sequence_patterns(self) -> Dict[str, Any]:
        """연속 패턴 분석"""
        cache_key = 'sequence'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        sequence_counts = {}
        consecutive_count = 0
        total_pairs = 0
        
        for numbers in self.data['numbers']:
            for i in range(len(numbers)-1):
                seq = (numbers[i], numbers[i+1])
                sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
                total_pairs += 1
                if numbers[i+1] - numbers[i] == 1:
                    consecutive_count += 1
                    
        consecutive_probability = consecutive_count / total_pairs if total_pairs > 0 else 0
                
        self.sequence_stats = {
            'sequence_counts': sequence_counts,
            'consecutive_probability': consecutive_probability
        }
        
        self._cache_result(cache_key, self.sequence_stats)
        return self.sequence_stats

    @safe_execute
    @log_performance
    def _analyze_oddeven_patterns(self) -> Dict[str, Any]:
        """홀짝 패턴 분석"""
        cache_key = 'oddeven'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        pattern_counts = {}
        total_count = 0
        
        # 패턴 카운트 초기화
        for numbers in self.data['numbers']:
            pattern = ''.join(['O' if n % 2 == 1 else 'E' for n in numbers])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_count += 1
            
        # 패턴 확률 계산
        pattern_probabilities = {
            pattern: count / total_count 
            for pattern, count in pattern_counts.items()
        }
        
        result = {
            'pattern_counts': pattern_counts,
            'pattern_probabilities': pattern_probabilities
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_range_distribution(self) -> Dict[str, Any]:
        """구간 분포 분석"""
        cache_key = 'range'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        range_counts = {i: 0 for i in range(1, 6)}
        for numbers in self.data['numbers']:
            for num in numbers:
                range_idx = (num - 1) // 10
                range_counts[range_idx + 1] += 1
                
        self.range_stats['range_counts'] = range_counts
        self._cache_result(cache_key, self.range_stats)
        return self.range_stats

    @safe_execute
    @log_performance
    def _analyze_sum_patterns(self) -> Dict[str, Any]:
        """합계 패턴 분석"""
        cache_key = 'sum'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        sums = [sum(numbers) for numbers in self.data['numbers']]
        
        # 기본 통계량
        self.sum_stats = {
            'sum_mean': np.mean(sums),
            'sum_std': np.std(sums),
            'sum_min': min(sums),
            'sum_max': max(sums)
        }
        
        # 정규성 검정
        _, p_value = stats.normaltest(sums)
        self.sum_stats['normality_p_value'] = p_value
        
        self._cache_result(cache_key, self.sum_stats)
        return self.sum_stats

    @safe_execute
    @log_performance
    def _analyze_gap_patterns(self) -> Dict[str, Any]:
        """간격 패턴 분석"""
        cache_key = 'gap'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        gaps = []
        for i in range(1, len(self.data)):
            prev_numbers = set(self.data.iloc[i-1]['numbers'])
            curr_numbers = set(self.data.iloc[i]['numbers'])
            gap = len(prev_numbers - curr_numbers)
            gaps.append(gap)
            
        self.gap_stats['gap_mean'] = np.mean(gaps)
        self.gap_stats['gap_std'] = np.std(gaps)
        self._cache_result(cache_key, self.gap_stats)
        return self.gap_stats

    @safe_execute
    @log_performance
    def _analyze_markov_chain(self) -> Dict[str, Any]:
        """마코프 체인 분석"""
        cache_key = 'markov'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # 전이 행렬 초기화
        transition_matrix = np.zeros((45, 45))
        
        # 전이 횟수 계산
        for numbers in self.data['numbers']:
            for i in range(len(numbers)-1):
                from_num = numbers[i] - 1  # 0-based index
                to_num = numbers[i+1] - 1
                transition_matrix[from_num, to_num] += 1
                
        # 정규화
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
        
        # 고유값과 고유벡터 계산
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        
        # 정상 분포 계산
        stationary_dist = np.real(eigenvectors[:, 0])
        stationary_dist = stationary_dist / stationary_dist.sum()
        
        # 높은 확률 전이 추출
        high_probability_transitions = []
        threshold = np.mean(transition_matrix) + np.std(transition_matrix)
        for i in range(45):
            for j in range(45):
                if transition_matrix[i, j] > threshold:
                    high_probability_transitions.append((i+1, j+1, transition_matrix[i, j]))
        
        result = {
            'transition_matrix': transition_matrix.tolist(),
            'high_probability_transitions': high_probability_transitions,
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist(),
            'stationary_distribution': stationary_dist.tolist()
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_fourier(self) -> Dict[str, Any]:
        """푸리에 변환 분석"""
        cache_key = 'fourier'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # 모든 번호를 1차원 배열로 변환
        numbers = np.array([num for nums in self.data['numbers'] for num in nums])
        
        # 푸리에 변환 수행
        fft_result = np.fft.fft(numbers)
        frequencies = np.fft.fftfreq(len(numbers))
        
        # 진폭 계산
        amplitudes = np.abs(fft_result)
        
        # 주요 주파수 성분 추출
        significant_freq_mask = amplitudes > np.mean(amplitudes) + np.std(amplitudes)
        significant_frequencies = frequencies[significant_freq_mask]
        
        # 주기성 있는 번호 추출
        periodic_numbers = []
        for freq in significant_frequencies:
            if freq > 0:  # 양의 주파수만 고려
                period = int(1/freq)
                if 1 <= period <= 45:  # 유효한 번호 범위
                    periodic_numbers.append(period)
        
        result = {
            'frequencies': frequencies.tolist(),
            'amplitudes': amplitudes.tolist(),
            'significant_frequencies': significant_frequencies.tolist(),
            'periodic_numbers': periodic_numbers
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_duplicate_patterns(self) -> Dict[str, Any]:
        """중복 패턴 분석"""
        cache_key = 'duplicate'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        duplicate_patterns = {}
        pattern_details = {}
        
        for i, numbers in enumerate(self.data['numbers']):
            pattern = tuple(sorted(numbers))
            if pattern in duplicate_patterns:
                duplicate_patterns[pattern] += 1
                pattern_details[pattern]['occurrence_count'] += 1
                pattern_details[pattern]['gaps'].append(i - pattern_details[pattern]['last_occurrence'])
                pattern_details[pattern]['last_occurrence'] = i
            else:
                duplicate_patterns[pattern] = 1
                pattern_details[pattern] = {
                    'occurrence_count': 1,
                    'gaps': [],
                    'last_occurrence': i
                }
        
        # 통계 계산
        total_patterns = len(duplicate_patterns)
        duplicate_count = sum(1 for count in duplicate_patterns.values() if count > 1)
        most_duplicated = max(duplicate_patterns.values()) if duplicate_patterns else 0
        
        # 평균 중복 간격 계산
        all_gaps = [gap for details in pattern_details.values() for gap in details['gaps']]
        avg_gap = np.mean(all_gaps) if all_gaps else 0
        
        statistics = {
            'duplicate_rate': duplicate_count / total_patterns if total_patterns > 0 else 0,
            'most_duplicated_count': most_duplicated,
            'average_duplication_gap': avg_gap
        }
        
        result = {
            'duplicate_patterns': duplicate_patterns,
            'pattern_details': pattern_details,
            'statistics': statistics
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_number_patterns(self) -> Dict[str, Any]:
        """번호별 패턴 분석"""
        cache_key = 'number'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        pattern_counts = {'high': 0, 'low': 0}
        number_patterns = {}
        
        # 1-45 번호에 대한 패턴 분석
        for num in range(1, 46):
            appearances = []
            gaps = []
            consecutive = 0
            non_appearance = 0
            last_appearance = -1
            
            for i, numbers in enumerate(self.data['numbers']):
                if num in numbers:
                    if last_appearance == -1:
                        consecutive = 1
                    elif i - last_appearance == 1:
                        consecutive += 1
                    else:
                        consecutive = 1
                    appearances.append(i)
                    if last_appearance != -1:
                        gaps.append(i - last_appearance)
                    last_appearance = i
                else:
                    non_appearance += 1
                    
            number_patterns[num] = {
                'appearances': appearances,
                'gaps': gaps,
                'consecutive': consecutive,
                'non_appearance': non_appearance,
                'last_appearance': last_appearance
            }
            
        # 고/저 패턴 분석
        for numbers in self.data['numbers']:
            high_count = sum(1 for n in numbers if n > 23)
            low_count = 6 - high_count
            pattern_counts['high'] += high_count
            pattern_counts['low'] += low_count
            
        result = {
            'pattern_counts': pattern_counts,
            'number_patterns': number_patterns
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_combination_stats(self) -> Dict[str, Any]:
        """조합 통계 분석"""
        cache_key = 'combination'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        combination_counts = {}
        for numbers in self.data['numbers']:
            for i in range(len(numbers)-1):
                for j in range(i+1, len(numbers)):
                    combo = tuple(sorted([numbers[i], numbers[j]]))
                    combination_counts[combo] = combination_counts.get(combo, 0) + 1
                    
        self.combination_stats['combination_counts'] = combination_counts
        self._cache_result(cache_key, self.combination_stats)
        return self.combination_stats

    @safe_execute
    @log_performance
    def _analyze_moving_averages(self) -> Dict[str, Any]:
        """이동 평균 분석"""
        cache_key = 'moving'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        # 합계 계산
        sums = [sum(numbers) for numbers in self.data['numbers']]
        sums_series = pd.Series(sums)
        
        # 이동 평균 계산
        moving_averages = {}
        for period in [5, 10, 20]:
            ma = sums_series.rolling(window=period).mean()
            cross_points = []
            trend_strength = []
            
            # 골든/데드 크로스 포인트 계산
            for i in range(period, len(ma)):
                if pd.isna(ma[i]) or pd.isna(ma[i-1]):
                    continue
                if ma[i] > ma[i-1]:
                    cross_points.append(1)  # 골든 크로스
                else:
                    cross_points.append(-1)  # 데드 크로스
                
            # 트렌드 강도 계산
            for i in range(period, len(ma)):
                if pd.isna(ma[i]):
                    trend_strength.append(0)
                    continue
                strength = (ma[i] - ma[i-period]) / ma[i-period] * 100
                trend_strength.append(strength)
            
            moving_averages[f'ma_{period}'] = {
                'moving_averages': ma.tolist(),
                'cross_points': cross_points,
                'trend_strength': trend_strength
            }
        
        # 트렌드 방향성 계산
        trend_direction = {}
        for period in [5, 10, 20]:
            ma = sums_series.rolling(window=period).mean()
            direction = []
            for i in range(len(ma)):
                if pd.isna(ma[i]):
                    direction.append(0)
                    continue
                if i == 0:
                    direction.append(0)
                    continue
                if ma[i] > ma[i-1]:
                    direction.append(1)  # 상승
                elif ma[i] < ma[i-1]:
                    direction.append(-1)  # 하락
                else:
                    direction.append(0)  # 보합
            trend_direction[f'ma_{period}'] = direction
        
        result = {
            'moving_averages': moving_averages,
            'trend_direction': trend_direction
        }
        
        self._cache_result(cache_key, result)
        return result

    @safe_execute
    @log_performance
    def _analyze_robust_stats(self) -> Dict[str, Any]:
        """로버스트 통계 분석"""
        cache_key = 'robust'
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
            
        # 합계 계산
        sums = [sum(numbers) for numbers in self.data['numbers']]
        sums_array = np.array(sums)
        
        # 로버스트 통계량 계산
        robust_stats = {
            'median': np.median(sums_array),
            'q1': np.percentile(sums_array, 25),
            'q3': np.percentile(sums_array, 75),
            'iqr': np.percentile(sums_array, 75) - np.percentile(sums_array, 25),
            'mad': np.median(np.abs(sums_array - np.median(sums_array)))
        }
        
        # 윈저화된 통계량 계산
        lower_bound = np.percentile(sums_array, 5)
        upper_bound = np.percentile(sums_array, 95)
        winsorized_array = np.clip(sums_array, lower_bound, upper_bound)
        
        winsorized_stats = {
            'mean': np.mean(winsorized_array),
            'std': np.std(winsorized_array),
            'median': np.median(winsorized_array),
            'q1': np.percentile(winsorized_array, 25),
            'q3': np.percentile(winsorized_array, 75)
        }
        
        # 극단값 제거 후 통계량 계산
        iqr = robust_stats['iqr']
        lower_fence = robust_stats['q1'] - 1.5 * iqr
        upper_fence = robust_stats['q3'] + 1.5 * iqr
        clean_array = sums_array[(sums_array >= lower_fence) & (sums_array <= upper_fence)]
        
        clean_stats = {
            'mean': np.mean(clean_array),
            'std': np.std(clean_array),
            'median': np.median(clean_array),
            'q1': np.percentile(clean_array, 25),
            'q3': np.percentile(clean_array, 75)
        }
        
        # 극단값 개수 계산
        outlier_count = np.sum((sums_array < lower_fence) | (sums_array > upper_fence))
        
        result = {
            'robust_stats': robust_stats,
            'winsorized_stats': winsorized_stats,
            'clean_stats': clean_stats,
            'outlier_count': int(outlier_count)
        }
        
        self._cache_result(cache_key, result)
        return result

    def _plot_all_patterns(self, plot_dir: Union[str, Path]):
        """
        모든 패턴 시각화

        Args:
            plot_dir: 그래프 저장 디렉토리 경로
        """
        # 기본 저장 경로 설정
        base_dir = Path('lottery/data/results/graph')
        plot_dir = Path(plot_dir)
        plot_dir = base_dir / plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

        # 병렬로 그래프 생성
        futures = [
            self.executor.submit(self._plot_frequency, str(plot_dir / 'frequency.png')),
            self.executor.submit(self._plot_oddeven_patterns, str(plot_dir / 'oddeven_patterns.png')),
            self.executor.submit(self._plot_range_distribution, str(plot_dir / 'range_distribution.png')),
            self.executor.submit(self._plot_sum_distribution, str(plot_dir / 'sum_distribution.png')),
            self.executor.submit(self._plot_gap_patterns, str(plot_dir / 'gap_patterns.png')),
            self.executor.submit(self._plot_markov_chain, str(plot_dir / 'markov_chain.png')),
            self.executor.submit(self._plot_fourier, str(plot_dir / 'fourier.png')),
            self.executor.submit(self._plot_duplicate_patterns, str(plot_dir / 'duplicate_patterns.png')),
            self.executor.submit(self._plot_number_patterns, str(plot_dir / 'number_patterns.png')),
            self.executor.submit(self._plot_combination_stats, str(plot_dir / 'combination_stats.png')),
            self.executor.submit(self._plot_moving_averages, str(plot_dir / 'moving_averages.png')),
            self.executor.submit(self._plot_robust_stats, str(plot_dir / 'robust_stats.png'))
        ]

        # 모든 그래프 생성 완료 대기
        for future in futures:
            future.result()

    def _plot_frequency(self, filepath: str):
        """빈도수 그래프 생성"""
        if self.frequency_stats is None:
            return

        plt.figure(figsize=(15, 6))
        numbers = list(range(1, 46))
        frequencies = [self.frequency_stats['frequency'].get(num, 0) for num in numbers]

        plt.bar(numbers, frequencies)
        plt.title('번호별 출현 빈도')
        plt.xlabel('번호')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_oddeven_patterns(self, filepath: str):
        """홀짝 패턴 그래프 생성"""
        if self.oddeven_stats is None:
            return

        plt.figure(figsize=(10, 6))
        patterns = list(self.oddeven_stats['pattern_counts'].keys())
        counts = list(self.oddeven_stats['pattern_counts'].values())

        plt.bar([str(p) for p in patterns], counts)
        plt.title('홀짝 조합 패턴')
        plt.xlabel('(홀수 개수, 짝수 개수)')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_range_distribution(self, filepath: str):
        """구간 분포 그래프 생성"""
        if self.range_stats is None:
            return

        plt.figure(figsize=(10, 6))
        ranges = list(self.range_stats['range_counts'].keys())
        counts = [self.range_stats['range_counts'][r]['count'] for r in ranges]

        plt.bar(ranges, counts)
        plt.title('구간별 번호 분포')
        plt.xlabel('구간')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_sum_distribution(self, filepath: str):
        """합계 분포 그래프 생성"""
        if self.sum_stats is None:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.sum_stats['sum_mean'], bins=30)
        plt.axvline(self.sum_stats['sum_mean'], color='r', linestyle='--', label='평균')

        plt.title('당첨 번호 합계 분포')
        plt.xlabel('합계')
        plt.ylabel('빈도수')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_gap_patterns(self, filepath: str):
        """간격 패턴 그래프 생성"""
        if self.gap_stats is None:
            return

        plt.figure(figsize=(10, 6))
        plt.hist(self.gap_stats['gap_mean'], bins=20)
        plt.axvline(self.gap_stats['gap_median'], color='r', linestyle='--', label='중앙값')

        plt.title('번호 간 간격 분포')
        plt.xlabel('간격')
        plt.ylabel('빈도수')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_markov_chain(self, filepath: str):
        """마코프 체인 그래프 생성"""
        if self.markov_stats is None:
            return

        plt.figure(figsize=(12, 8))
        transition_matrix = np.array(self.markov_stats['transition_matrix'])

        sns.heatmap(transition_matrix, cmap='YlOrRd')
        plt.title('번호 전이 확률')
        plt.xlabel('다음 번호')
        plt.ylabel('현재 번호')

        plt.savefig(filepath)
        plt.close()

    def _plot_fourier(self, filepath: str):
        """푸리에 변환 그래프 생성"""
        if self.fourier_stats is None:
            return

        plt.figure(figsize=(12, 6))
        frequencies = self.fourier_stats['frequencies']
        amplitudes = self.fourier_stats['amplitudes']

        plt.plot(frequencies, amplitudes)
        plt.title('주기성 분석')
        plt.xlabel('주파수')
        plt.ylabel('진폭')
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath)
        plt.close()

    def _plot_duplicate_patterns(self, filepath: str):
        """중복 패턴 그래프 생성"""
        if not hasattr(self, 'duplicate_stats') or self.duplicate_stats is None:
            return

        plt.figure(figsize=(15, 8))

        # 중복 패턴 빈도수 그래프
        plt.subplot(2, 1, 1)
        pattern_counts = [info['occurrence_count'] for info in self.duplicate_stats['pattern_details']]
        plt.hist(pattern_counts, bins=range(min(pattern_counts), max(pattern_counts) + 2), align='left')
        plt.title('중복 패턴 출현 빈도 분포')
        plt.xlabel('중복 횟수')
        plt.ylabel('패턴 수')
        plt.grid(True, alpha=0.3)

        # 중복 패턴 간격 그래프
        plt.subplot(2, 1, 2)
        all_gaps = [gap for info in self.duplicate_stats['pattern_details'] for gap in info['gaps']]
        plt.hist(all_gaps, bins=30)
        plt.title('중복 패턴 간격 분포')
        plt.xlabel('간격 (회차)')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_number_patterns(self, filepath: str):
        """번호별 패턴 그래프 생성"""
        if not hasattr(self, 'number_patterns') or self.number_patterns is None:
            return

        plt.figure(figsize=(15, 10))

        # 출현 빈도 그래프
        plt.subplot(2, 2, 1)
        numbers = list(range(1, 46))
        frequencies = [self.number_patterns['number_patterns'][num]['total_appearances'] for num in numbers]
        plt.bar(numbers, frequencies)
        plt.title('번호별 출현 빈도')
        plt.xlabel('번호')
        plt.ylabel('출현 횟수')
        plt.grid(True, alpha=0.3)

        # 출현 간격 분포 그래프
        plt.subplot(2, 2, 2)
        all_gaps = [gap for num in numbers for gap in self.number_patterns['number_patterns'][num]['gaps']]
        plt.hist(all_gaps, bins=30)
        plt.title('출현 간격 분포')
        plt.xlabel('간격 (회차)')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        # 연속 출현 패턴 그래프
        plt.subplot(2, 2, 3)
        consecutive_counts = [len(self.number_patterns['number_patterns'][num]['consecutive_appearances']) for num in numbers]
        plt.bar(numbers, consecutive_counts)
        plt.title('연속 출현 패턴 수')
        plt.xlabel('번호')
        plt.ylabel('연속 출현 패턴 수')
        plt.grid(True, alpha=0.3)

        # 미출현 기간 분포 그래프
        plt.subplot(2, 2, 4)
        non_appearances = [period for num in numbers for period in self.number_patterns['number_patterns'][num]['non_appearance_periods']]
        plt.hist(non_appearances, bins=30)
        plt.title('미출현 기간 분포')
        plt.xlabel('미출현 기간 (회차)')
        plt.ylabel('빈도수')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_combination_stats(self, filepath: str):
        """조합 통계 그래프 생성"""
        if not hasattr(self, 'combination_stats') or self.combination_stats is None:
            return

        plt.figure(figsize=(15, 10))

        # 기본 통계량 그래프
        plt.subplot(2, 2, 1)
        stats = self.combination_stats['combination_counts']
        plt.bar(stats.keys(), stats.values())
        plt.title('기본 통계량')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 시간에 따른 변화 패턴 그래프
        time_stats = self.combination_stats['time_series_stats']
        for i, (metric, values) in enumerate(time_stats.items(), 2):
            plt.subplot(2, 2, i)
            plt.plot(values)
            plt.title(f'{metric} 변화 패턴')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_moving_averages(self, filepath: str):
        """이동 평균 그래프 생성"""
        if not hasattr(self, 'moving_averages') or self.moving_averages is None:
            return

        plt.figure(figsize=(15, 10))

        # 이동 평균 그래프
        plt.subplot(2, 2, 1)
        for period in [5, 10, 20]:
            ma_data = self.moving_averages['moving_averages'][f'ma_{period}']['moving_averages']
            plt.plot(ma_data, label=f'MA{period}')
        plt.title('이동 평균 추이')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 골든/데드 크로스 그래프
        plt.subplot(2, 2, 2)
        for period in [5, 10, 20]:
            cross_points = self.moving_averages['moving_averages'][f'ma_{period}']['cross_points']
            plt.plot(cross_points, label=f'MA{period} 크로스')
        plt.title('골든/데드 크로스 발생 횟수')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 트렌드 강도 그래프
        plt.subplot(2, 2, 3)
        for period in [5, 10, 20]:
            strength = self.moving_averages['moving_averages'][f'ma_{period}']['trend_strength']
            plt.plot(strength, label=f'MA{period} 강도')
        plt.title('트렌드 강도')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 트렌드 방향성 그래프
        plt.subplot(2, 2, 4)
        for period in [5, 10, 20]:
            direction = self.moving_averages['trend_direction'][f'ma_{period}']
            plt.plot(direction, label=f'MA{period} 방향')
        plt.title('트렌드 방향성')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def _plot_robust_stats(self, filepath: str):
        """로버스트 통계 그래프 생성"""
        if not hasattr(self, 'robust_stats') or self.robust_stats is None:
            return

        plt.figure(figsize=(15, 10))

        # 로버스트 통계량 그래프
        plt.subplot(2, 2, 1)
        robust = self.robust_stats['robust_stats']
        plt.bar(robust.keys(), robust.values())
        plt.title('로버스트 통계량')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 윈저화된 통계량 그래프
        plt.subplot(2, 2, 2)
        winsorized = self.robust_stats['winsorized_stats']
        plt.bar(winsorized.keys(), winsorized.values())
        plt.title('윈저화된 통계량')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 극단값 제거 후 통계량 그래프
        plt.subplot(2, 2, 3)
        clean = self.robust_stats['clean_stats']
        plt.bar(clean.keys(), clean.values())
        plt.title('극단값 제거 후 통계량')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # 극단값 분포 그래프
        plt.subplot(2, 2, 4)
        outlier_count = self.robust_stats['outlier_count']
        plt.pie([outlier_count, len(self.data) - outlier_count], 
                labels=['극단값', '정상값'],
                autopct='%1.1f%%')
        plt.title('극단값 분포')

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

    def save_analysis_results(self, filepath: str):
        """분석 결과 저장"""
        results = {
            'frequency': self.frequency_stats,
            'sequence': self.sequence_stats,
            'oddeven': self.oddeven_stats,
            'range': self.range_stats,
            'sum': self.sum_stats,
            'gap': self.gap_stats,
            'markov': self.markov_stats,
            'fourier': self.fourier_stats
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, results)
        logger.info(f'분석 결과 저장 완료: {filepath}')

    def load_analysis_results(self, filepath: str):
        """분석 결과 로드"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f'파일을 찾을 수 없습니다: {filepath}')

        results = np.load(filepath, allow_pickle=True).item()

        self.frequency_stats = results['frequency']
        self.sequence_stats = results['sequence']
        self.oddeven_stats = results['oddeven']
        self.range_stats = results['range']
        self.sum_stats = results['sum']
        self.gap_stats = results['gap']
        self.markov_stats = results['markov']
        self.fourier_stats = results['fourier']

        logger.info(f'분석 결과 로드 완료: {filepath}')

    def __del__(self):
        """리소스 정리"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"리소스 정리 중 오류 발생: {str(e)}")


# 모듈 테스트
if __name__ == "__main__":
    print("=== 패턴 분석 테스트 ===")

    # 설정 객체 생성
    config_dict = {
        'data': {
            'historical_data_path': 'lottery/data/raw/lottery.csv',
            'processed_data_path': 'lottery/data/processed/processed_data.pkl',
            'numbers_to_select': 6,
            'min_number': 1,
            'max_number': 45,
            'batch_size': 32,
            'num_workers': 4
        },
        'pattern_analysis': {
            'markov_chain_order': 2,
            'fourier_window_size': 52,
            'trend_window_size': 10,
            'significance_level': 0.05,
            'use_gpu': torch.cuda.is_available(),
            'num_workers': 4,
            'batch_size': 32,
            'cache_size': 1000
        }
    }
    config = Config(config_dict)

    # 데이터 매니저를 통한 데이터 로드
    from lottery.src.utils.data_loader import DataManager
    data_manager = DataManager(config)
    data_manager.load_data()
    print(f"데이터 로드 완료: {len(data_manager.data)} 행")

    # 패턴 분석
    analyzer = PatternAnalyzer(config, data_manager.data)
    results = analyzer.analyze()

    print("\n빈도 분석 결과:")
    print(f"카이제곱 통계량: {results['frequency']['chi2_stat']:.2f}")
    print(f"p-값: {results['frequency']['p_value']:.4f}")

    print("\n연속 패턴 분석 결과:")
    print(f"연속 번호 출현 확률: {results['sequence_patterns']['consecutive_probability']:.4f}")

    print("\n홀짝 패턴 분석 결과:")
    print("패턴별 확률:", results['oddeven_patterns']['pattern_probabilities'])

    print("\n구간 분포 분석 결과:")
    print("구간별 확률:", results['range_distribution']['range_probabilities'])

    print("\n합계 패턴 분석 결과:")
    print(f"평균: {results['sum_patterns']['sum_mean']:.2f}")
    print(f"표준편차: {results['sum_patterns']['sum_std']:.2f}")

    print("\n간격 패턴 분석 결과:")
    print(f"평균 간격: {results['gap_patterns']['gap_mean']:.2f}")
    print(f"간격 표준편차: {results['gap_patterns']['gap_std']:.2f}")

    print("\n중복 패턴 분석 결과:")
    print("중복된 패턴 수:", results['duplicate_patterns']['duplicate_patterns'])
    print("중복 패턴 비율:", results['duplicate_patterns']['statistics']['duplicate_rate'])
    print("중복된 패턴 중 가장 많이 중복된 패턴의 중복 횟수:", results['duplicate_patterns']['statistics']['most_duplicated_count'])
    print("중복된 패턴의 평균 중복 간격:", results['duplicate_patterns']['statistics']['average_duplication_gap'])

    print("\n테스트 완료")