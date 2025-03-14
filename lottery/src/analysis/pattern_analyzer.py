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
from typing import Dict, List, Tuple, Optional, Any
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

from lottery.src.utils.config import Config

# 로거 설정
logger = logging.getLogger(__name__)

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
    cache_ttl: float = 300  # Added for the new cache method

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

        # 분석 결과 저장
        self.frequency_stats = None
        self.sequence_stats = None
        self.oddeven_stats = None
        self.range_stats = None
        self.sum_stats = None
        self.gap_stats = None
        self.markov_stats = None
        self.fourier_stats = None
        self.number_patterns = None

        # GPU 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.pattern_config.use_gpu else 'cpu')

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

    def _init_cache(self):
        """캐시 초기화"""
        self._cache = {}
        self._cache_timestamps = {}

    def _get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """
        캐시된 결과 조회

        Args:
            key: 캐시 키

        Returns:
            캐시된 결과 또는 None
        """
        if key in self._cache:
            # 캐시 유효 시간 확인
            if time.time() - self._cache_timestamps[key] < self.pattern_config.cache_ttl:
                logger.debug(f"캐시된 결과 사용: {key}")
                return self._cache[key]
            else:
                # 캐시 만료
                del self._cache[key]
                del self._cache_timestamps[key]
        return None

    def _cache_result(self, key: str, result: Dict[str, Any]):
        """
        결과 캐싱

        Args:
            key: 캐시 키
            result: 캐시할 결과
        """
        self._cache[key] = result
        self._cache_timestamps[key] = time.time()
        logger.debug(f"결과 캐싱 완료: {key}")

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

            # 병렬로 분석 수행
            futures = [
                self.executor.submit(self._analyze_frequency),
                self.executor.submit(self._analyze_sequence_patterns),
                self.executor.submit(self._analyze_oddeven_patterns),
                self.executor.submit(self._analyze_range_distribution),
                self.executor.submit(self._analyze_sum_patterns),
                self.executor.submit(self._analyze_gap_patterns),
                self.executor.submit(self._analyze_markov_chain),
                self.executor.submit(self._analyze_fourier),
                self.executor.submit(self._analyze_duplicate_patterns),
                self.executor.submit(self._analyze_number_patterns),
                self.executor.submit(self._analyze_combination_stats),
                self.executor.submit(self._analyze_moving_averages),
                self.executor.submit(self._analyze_robust_stats)
            ]

            # 결과 수집
            results = {
                'frequency': futures[0].result(),
                'sequence_patterns': futures[1].result(),
                'oddeven_patterns': futures[2].result(),
                'range_distribution': futures[3].result(),
                'sum_patterns': futures[4].result(),
                'gap_patterns': futures[5].result(),
                'markov_chain': futures[6].result(),
                'fourier': futures[7].result(),
                'duplicate_patterns': futures[8].result(),
                'number_patterns': futures[9].result(),
                'combination_stats': futures[10].result(),
                'moving_averages': futures[11].result(),
                'robust_stats': futures[12].result()
            }

            # 결과 캐싱
            self._cache_result(cache_key, results)
            logger.info("패턴 분석 완료")
            return results

        except Exception as e:
            logger.error(f"패턴 분석 실패: {str(e)}")
            raise

    def _analyze_frequency(self) -> Dict[str, Any]:
        """번호별 출현 빈도 분석"""
        # 캐시 확인
        cache_key = 'frequency_analysis'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 빈도수 계산 (GPU)
        unique_numbers, counts = torch.unique(numbers_tensor, return_counts=True)
        frequency = dict(zip(unique_numbers.cpu().numpy(), counts.cpu().numpy()))

        # 통계량 계산
        total_count = len(numbers_tensor.flatten())
        probabilities = {num: count/total_count for num, count in frequency.items()}

        # 카이제곱 검정을 위한 데이터 준비
        observed = np.zeros(45, dtype=np.float64)
        # unique_numbers를 정수형으로 변환하여 인덱스로 사용
        indices = unique_numbers.cpu().numpy().astype(np.int64) - 1
        observed[indices] = counts.cpu().numpy().astype(np.float64)
        expected = np.full(45, total_count / 45, dtype=np.float64)

        # 카이제곱 통계량 직접 계산 (float64 타입 유지)
        chi2_stat = np.sum((observed - expected) ** 2 / expected, dtype=np.float64)

        # 자유도는 44 (45개 카테고리 - 1)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=44)

        result = {
            'frequency': frequency,
            'probabilities': probabilities,
            'chi2_stat': chi2_stat,
            'p_value': p_value
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_sequence_patterns(self) -> Dict[str, Any]:
        """연속된 번호 패턴 분석"""
        # 캐시 확인
        cache_key = 'sequence_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)
        sorted_numbers = torch.sort(numbers_tensor, dim=1)[0]

        # 연속된 번호 찾기 (GPU)
        diffs = sorted_numbers[:, 1:] - sorted_numbers[:, :-1]
        consecutive_mask = diffs == 1
        sequence_counts = torch.sum(consecutive_mask.to(dtype=torch.float64), dim=1)

        result = {
            'sequence_counts': sequence_counts.cpu().numpy().tolist(),
            'consecutive_probability': torch.mean(sequence_counts).item()
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_oddeven_patterns(self) -> Dict[str, Any]:
        """홀짝 패턴 분석"""
        # 캐시 확인
        cache_key = 'oddeven_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 홀짝 패턴 계산 (GPU)
        odd_mask = numbers_tensor % 2 == 1
        odd_counts = torch.sum(odd_mask.to(dtype=torch.float64), dim=1)
        patterns = torch.stack([odd_counts, 6 - odd_counts], dim=1)

        # 패턴 빈도 계산
        unique_patterns, pattern_counts = torch.unique(patterns, dim=0, return_counts=True)
        pattern_dict = {
            tuple(pattern.cpu().numpy()): count.item()
            for pattern, count in zip(unique_patterns, pattern_counts)
        }

        result = {
            'pattern_counts': pattern_dict,
            'pattern_probabilities': {
                k: v/len(numbers_array) for k, v in pattern_dict.items()
            }
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_range_distribution(self) -> Dict[str, Any]:
        """구간별 분포 분석"""
        # 캐시 확인
        cache_key = 'range_distribution'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 구간 정의
        ranges = torch.tensor([[1,10], [11,20], [21,30], [31,40], [41,45]], device=self.device, dtype=torch.float64)

        # 구간별 카운트 계산 (GPU)
        range_counts = torch.zeros(5, device=self.device, dtype=torch.float64)
        for i, (start, end) in enumerate(ranges):
            mask = (numbers_tensor >= start) & (numbers_tensor <= end)
            range_counts[i] = torch.sum(mask.to(dtype=torch.float64))

        # 결과를 딕셔너리 형태로 변환
        range_dict = {
            f'range_{i+1}': {
                'count': count.item(),
                'probability': count.item() / (len(numbers_array) * 6)
            }
            for i, count in enumerate(range_counts)
        }

        result = {
            'range_counts': range_dict,
            'range_probabilities': {
                k: v['probability'] for k, v in range_dict.items()
            }
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_sum_patterns(self) -> Dict[str, Any]:
        """합계 패턴 분석"""
        # 캐시 확인
        cache_key = 'sum_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 합계 계산 (GPU)
        sums = torch.sum(numbers_tensor, dim=1)

        # 정규성 검정을 위한 데이터 준비
        sums_np = sums.cpu().numpy().astype(np.float64)

        result = {
            'sum_mean': torch.mean(sums).item(),
            'sum_std': torch.std(sums).item(),
            'sum_min': torch.min(sums).item(),
            'sum_max': torch.max(sums).item(),
            'normality_p_value': stats.normaltest(sums_np)[1]
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_gap_patterns(self) -> Dict[str, Any]:
        """간격 패턴 분석"""
        # 캐시 확인
        cache_key = 'gap_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)
        sorted_numbers = torch.sort(numbers_tensor, dim=1)[0]

        # 간격 계산 (GPU)
        gaps = sorted_numbers[:, 1:] - sorted_numbers[:, :-1]

        # float64 타입으로 변환하여 통계 계산
        gaps_float64 = gaps.to(dtype=torch.float64)

        result = {
            'gap_mean': torch.mean(gaps_float64).item(),
            'gap_std': torch.std(gaps_float64).item(),
            'gap_min': torch.min(gaps_float64).item(),
            'gap_max': torch.max(gaps_float64).item(),
            'gap_median': torch.median(gaps_float64).item()
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_markov_chain(self) -> Dict[str, Any]:
        """마코프 체인 분석"""
        # 리스트 형태의 numbers를 numpy 배열로 변환
        numbers_array = np.array(self.data['numbers'].tolist())

        # 전이 행렬 계산
        transition_matrix = np.zeros((45, 45))
        for row in numbers_array:
            sorted_row = np.sort(row)
            for i in range(len(sorted_row) - 1):
                from_num = sorted_row[i] - 1
                to_num = sorted_row[i + 1] - 1
                transition_matrix[from_num, to_num] += 1

        # 정규화
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        # 고확률 전이 찾기
        high_prob_transitions = []
        threshold = np.mean(transition_matrix) + np.std(transition_matrix)
        for i in range(45):
            for j in range(45):
                if transition_matrix[i, j] > threshold:
                    high_prob_transitions.append((i + 1, j + 1))

        return {
            'transition_matrix': transition_matrix.tolist(),
            'high_probability_transitions': high_prob_transitions
        }

    def _analyze_fourier(self) -> Dict[str, Any]:
        """푸리에 변환 기반 주기성 분석"""
        # 캐시 확인
        cache_key = 'fourier_analysis'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # 시계열 데이터 준비
        time_series = numbers_array.flatten()

        # 푸리에 변환
        fft_result = fft(time_series)
        frequencies = np.fft.fftfreq(len(time_series))

        # 주요 주기 찾기 (오버플로우 방지를 위한 수정)
        magnitudes = np.abs(fft_result)  # 복소수의 절대값 계산 (제곱 연산 사용하지 않음)

        # 중요도 임계값 계산 (float64 타입 유지)
        threshold = np.mean(magnitudes) + np.std(magnitudes)
        significant_freqs = frequencies[magnitudes > threshold]

        # 주기성 있는 번호 찾기 (오버플로우 방지 수정)
        periodic_numbers = []
        for i in range(1, 46):
            if i in time_series:
                idx = np.where(time_series == i)[0]
                if len(idx) > 1:
                    periods = np.diff(idx).astype(np.float64)  # float64로 변환
                    if np.std(periods) < np.mean(periods):
                        periodic_numbers.append(str(i))

        result = {
            'frequencies': frequencies.tolist(),
            'magnitudes': magnitudes.tolist(),
            'significant_frequencies': significant_freqs.tolist(),
            'periodic_numbers': periodic_numbers
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_duplicate_patterns(self) -> Dict[str, Any]:
        """
        중복 패턴 분석
        - 과거 당첨 번호 중 중복된 조합이 있는지 확인
        - 중복된 번호들의 출현 빈도 분석
        - 중복된 번호들의 간격 분석
        """
        # 캐시 확인
        cache_key = 'duplicate_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환
        numbers_array = np.array(self.data['numbers'].tolist())

        # 중복 패턴 찾기
        duplicate_patterns = {}
        for i in range(len(numbers_array)):
            current_numbers = tuple(sorted(numbers_array[i]))
            if current_numbers in duplicate_patterns:
                duplicate_patterns[current_numbers].append(i)
            else:
                duplicate_patterns[current_numbers] = [i]

        # 중복된 패턴만 필터링
        duplicate_patterns = {k: v for k, v in duplicate_patterns.items() if len(v) > 1}

        # 중복 패턴 분석
        analysis_results = {
            'total_patterns': len(numbers_array),
            'unique_patterns': len(set(tuple(sorted(row)) for row in numbers_array)),
            'duplicate_patterns': len(duplicate_patterns),
            'pattern_details': []
        }

        # 각 중복 패턴의 상세 정보 분석
        for pattern, indices in duplicate_patterns.items():
            pattern_info = {
                'numbers': list(pattern),
                'occurrence_count': len(indices),
                'occurrence_indices': indices,
                'gaps': []
            }

            # 간격 계산
            for i in range(len(indices) - 1):
                gap = indices[i + 1] - indices[i]
                pattern_info['gaps'].append(gap)

            pattern_info['average_gap'] = np.mean(pattern_info['gaps']) if pattern_info['gaps'] else 0
            pattern_info['min_gap'] = min(pattern_info['gaps']) if pattern_info['gaps'] else 0
            pattern_info['max_gap'] = max(pattern_info['gaps']) if pattern_info['gaps'] else 0

            analysis_results['pattern_details'].append(pattern_info)

        # 통계 정보 추가
        analysis_results['statistics'] = {
            'duplicate_rate': len(duplicate_patterns) / len(numbers_array),
            'most_duplicated_count': max(len(v) for v in duplicate_patterns.values()) if duplicate_patterns else 0,
            'average_duplication_gap': np.mean([np.mean(info['gaps']) for info in analysis_results['pattern_details']]) if analysis_results['pattern_details'] else 0
        }

        # 결과 캐싱
        self._cache_result(cache_key, analysis_results)
        return analysis_results

    def _analyze_number_patterns(self) -> Dict[str, Any]:
        """
        번호별 출현 패턴 분석
        - 각 번호의 출현 빈도
        - 연속 출현 패턴
        - 미출현 기간 패턴
        - 출현 간격 패턴
        """
        # 캐시 확인
        cache_key = 'number_patterns'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환
        numbers_array = np.array(self.data['numbers'].tolist())

        # 각 번호별 패턴 분석
        number_patterns = {}
        for number in range(1, 46):
            # 해당 번호의 출현 위치 찾기
            appearances = np.where(numbers_array == number)[0]

            # 출현 간격 계산
            gaps = np.diff(appearances) if len(appearances) > 1 else []

            # 연속 출현 패턴 찾기
            consecutive_appearances = []
            current_streak = 1
            for i in range(len(appearances) - 1):
                if appearances[i + 1] - appearances[i] == 1:
                    current_streak += 1
                else:
                    if current_streak > 1:
                        consecutive_appearances.append(current_streak)
                    current_streak = 1
            if current_streak > 1:
                consecutive_appearances.append(current_streak)

            # 미출현 기간 찾기
            non_appearance_periods = []
            for i in range(len(appearances) - 1):
                period = appearances[i + 1] - appearances[i] - 1
                if period > 0:
                    non_appearance_periods.append(period)

            # 패턴 정보 저장
            number_patterns[number] = {
                'total_appearances': len(appearances),
                'appearance_rate': len(appearances) / len(numbers_array),
                'appearance_indices': appearances.tolist(),
                'gaps': gaps.tolist(),
                'consecutive_appearances': consecutive_appearances,
                'non_appearance_periods': non_appearance_periods,
                'statistics': {
                    'average_gap': np.mean(gaps) if len(gaps) > 0 else 0,
                    'max_gap': np.max(gaps) if len(gaps) > 0 else 0,
                    'min_gap': np.min(gaps) if len(gaps) > 0 else 0,
                    'max_consecutive': max(consecutive_appearances) if consecutive_appearances else 0,
                    'average_non_appearance': np.mean(non_appearance_periods) if non_appearance_periods else 0,
                    'max_non_appearance': max(non_appearance_periods) if non_appearance_periods else 0
                }
            }

        # 전체 통계 계산
        overall_stats = {
            'total_numbers': len(number_patterns),
            'average_appearances': np.mean([info['total_appearances'] for info in number_patterns.values()]),
            'std_appearances': np.std([info['total_appearances'] for info in number_patterns.values()]),
            'average_non_appearance': np.mean([info['statistics']['average_non_appearance'] for info in number_patterns.values()]),
            'max_non_appearance': max(info['statistics']['max_non_appearance'] for info in number_patterns.values())
        }

        result = {
            'number_patterns': number_patterns,
            'overall_stats': overall_stats
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_combination_stats(self) -> Dict[str, Any]:
        """
        번호 조합 통계 분석
        - 당첨 번호 조합의 통계적 속성 분석
        - 분포 및 시간에 따른 변화 패턴 분석
        """
        # 캐시 확인
        cache_key = 'combination_stats'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 기본 통계량 계산
        stats_dict = {
            'variance': torch.var(numbers_tensor).item(),
            'std_dev': torch.std(numbers_tensor).item(),
            'range': (torch.max(numbers_tensor) - torch.min(numbers_tensor)).item(),
            'median': torch.median(numbers_tensor).item(),
            'skewness': torch.mean(((numbers_tensor - torch.mean(numbers_tensor)) / torch.std(numbers_tensor)) ** 3).item(),
            'kurtosis': torch.mean(((numbers_tensor - torch.mean(numbers_tensor)) / torch.std(numbers_tensor)) ** 4).item() - 3
        }

        # 시간에 따른 변화 패턴 분석
        window_size = 10
        n_windows = len(numbers_array) // window_size
        time_series_stats = {
            'variance': [],
            'std_dev': [],
            'range': [],
            'median': [],
            'skewness': [],
            'kurtosis': []
        }

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            window_data = numbers_tensor[start_idx:end_idx]

            time_series_stats['variance'].append(torch.var(window_data).item())
            time_series_stats['std_dev'].append(torch.std(window_data).item())
            time_series_stats['range'].append((torch.max(window_data) - torch.min(window_data)).item())
            time_series_stats['median'].append(torch.median(window_data).item())
            time_series_stats['skewness'].append(
                torch.mean(((window_data - torch.mean(window_data)) / torch.std(window_data)) ** 3).item()
            )
            time_series_stats['kurtosis'].append(
                torch.mean(((window_data - torch.mean(window_data)) / torch.std(window_data)) ** 4).item() - 3
            )

        result = {
            'basic_stats': stats_dict,
            'time_series_stats': time_series_stats
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_moving_averages(self) -> Dict[str, Any]:
        """
        이동 평균 기반 트렌드 분석
        - 다양한 기간의 이동 평균 계산
        - 골든/데드 크로스 분석
        - 트렌드 강도와 방향성 측정
        """
        # 캐시 확인
        cache_key = 'moving_averages'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 이동 평균 기간 설정
        periods = [5, 10, 20]
        moving_avg_results = {}

        for period in periods:
            # 각 번호별 이동 평균 계산
            ma = torch.zeros_like(numbers_tensor)
            for i in range(period - 1, len(numbers_tensor)):
                ma[i] = torch.mean(numbers_tensor[i-period+1:i+1], dim=0)

            # 골든/데드 크로스 분석
            cross_points = []
            for i in range(period, len(numbers_tensor)):
                prev_diff = ma[i-1].cpu().numpy() - ma[i-2].cpu().numpy()
                curr_diff = ma[i].cpu().numpy() - ma[i-1].cpu().numpy()
                cross_points.append(np.sum((prev_diff * curr_diff) < 0))

            # 트렌드 강도 계산
            trend_strength = torch.abs(ma[period:] - ma[period-1:-1]).mean(dim=0)

            moving_avg_results[f'ma_{period}'] = {
                'moving_averages': ma.cpu().numpy().tolist(),
                'cross_points': cross_points,
                'trend_strength': trend_strength.cpu().numpy().tolist()
            }

        # 트렌드 방향성 분석
        trend_direction = {}
        for period in periods:
            ma_data = np.array(moving_avg_results[f'ma_{period}']['moving_averages'])
            direction = np.zeros(len(ma_data))
            for i in range(1, len(ma_data)):
                direction[i] = np.mean(ma_data[i] - ma_data[i-1])
            trend_direction[f'ma_{period}'] = direction.tolist()

        result = {
            'moving_averages': moving_avg_results,
            'trend_direction': trend_direction
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _analyze_robust_stats(self) -> Dict[str, Any]:
        """
        로버스트 통계 분석
        - 중앙값, 사분위수, MAD 등 로버스트 통계 기법 활용
        - 극단값에 덜 민감한 패턴 분석
        """
        # 캐시 확인
        cache_key = 'robust_stats'
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result

        # 리스트 형태의 numbers를 numpy 배열로 변환 (float64 타입 사용)
        numbers_array = np.array(self.data['numbers'].tolist(), dtype=np.float64)

        # GPU 가속을 위한 텐서 변환
        numbers_tensor = torch.from_numpy(numbers_array).to(self.device)

        # 기본 로버스트 통계량 계산
        median = torch.median(numbers_tensor)
        robust_stats = {
            'median': median.item(),
            'q1': torch.quantile(numbers_tensor, 0.25).item(),
            'q3': torch.quantile(numbers_tensor, 0.75).item(),
            'iqr': torch.quantile(numbers_tensor, 0.75).item() - torch.quantile(numbers_tensor, 0.25).item(),
            'mad': torch.median(torch.abs(numbers_tensor - median)).item()
        }

        # 윈저화 적용 (상위/하위 5% 제한)
        lower_bound = torch.quantile(numbers_tensor, 0.05)
        upper_bound = torch.quantile(numbers_tensor, 0.95)
        winsorized_data = torch.clamp(numbers_tensor, lower_bound, upper_bound)

        # 윈저화된 데이터의 통계량
        winsorized_stats = {
            'mean': torch.mean(winsorized_data).item(),
            'std': torch.std(winsorized_data).item(),
            'median': torch.median(winsorized_data).item(),
            'q1': torch.quantile(winsorized_data, 0.25).item(),
            'q3': torch.quantile(winsorized_data, 0.75).item()
        }

        # 극단값 제거 후의 패턴 분석
        z_scores = torch.abs((numbers_tensor - torch.mean(numbers_tensor)) / torch.std(numbers_tensor))
        outlier_mask = z_scores > 2
        clean_data = numbers_tensor[~outlier_mask]

        clean_stats = {
            'mean': torch.mean(clean_data).item(),
            'std': torch.std(clean_data).item(),
            'median': torch.median(clean_data).item(),
            'q1': torch.quantile(clean_data, 0.25).item(),
            'q3': torch.quantile(clean_data, 0.75).item()
        }

        result = {
            'robust_stats': robust_stats,
            'winsorized_stats': winsorized_stats,
            'clean_stats': clean_stats,
            'outlier_count': torch.sum(outlier_mask).item()
        }

        # 결과 캐싱
        self._cache_result(cache_key, result)
        return result

    def _plot_all_patterns(self, plot_dir: str):
        """모든 패턴 시각화"""
        # 기본 저장 경로 설정
        base_dir = Path('lottery/data/results/graph')
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
        magnitudes = self.fourier_stats['magnitudes']

        plt.plot(frequencies, magnitudes)
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
        stats = self.combination_stats['basic_stats']
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
        self.executor.shutdown(wait=True)


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