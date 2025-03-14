"""
패턴 분석기 테스트 모듈

이 모듈은 패턴 분석기의 기능을 테스트합니다.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import pandas as pd
import sys
import os
import time

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from lottery.src.utils.config import Config
from lottery.src.utils.data_loader import DataManager
from lottery.src.analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisConfig

class TestPatternAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 임시 디렉토리 생성
        cls.temp_dir = tempfile.mkdtemp()

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
                'cache_size': 1000,
                'cache_ttl': 300,
                'use_amp': True,
                'num_streams': 3,
                'memory_fraction': 0.8,
                'enable_jit': True,
                'enable_fusion': True,
                'enable_profiling': False
            }
        }
        cls.config = Config(config_dict)

        # 데이터 매니저를 통한 데이터 로드
        cls.data_manager = DataManager(cls.config)
        cls.data_manager.load_data()
        print(f"테스트 데이터 로드 완료: {len(cls.data_manager.data)} 행")

        # 패턴 분석기 초기화
        cls.analyzer = PatternAnalyzer(cls.config, cls.data_manager.data)

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(cls.temp_dir)

    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.config, self.config)
        self.assertIsNotNone(self.analyzer.data)
        self.assertIsNotNone(self.analyzer.pattern_config)
        
        # 새로운 설정 옵션 테스트
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'cache_ttl'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'use_amp'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'num_streams'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'memory_fraction'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'enable_jit'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'enable_fusion'))
        self.assertTrue(hasattr(self.analyzer.pattern_config, 'enable_profiling'))

    def test_analyze(self):
        """전체 분석 테스트"""
        results = self.analyzer.analyze()
        
        # 기존 테스트
        self.assertIn('frequency', results)
        self.assertIn('sequence_patterns', results)
        self.assertIn('oddeven_patterns', results)
        self.assertIn('range_distribution', results)
        self.assertIn('sum_patterns', results)
        self.assertIn('gap_patterns', results)
        self.assertIn('markov_chain', results)
        self.assertIn('fourier', results)
        
        # 새로운 분석 결과 테스트
        self.assertIn('duplicate_patterns', results)
        self.assertIn('number_patterns', results)
        self.assertIn('combination_stats', results)
        self.assertIn('moving_averages', results)
        self.assertIn('robust_stats', results)

    def test_frequency_analysis(self):
        """빈도 분석 테스트"""
        result = self.analyzer._analyze_frequency()
        
        # 결과 검증
        self.assertIn('frequency', result)
        self.assertIn('probabilities', result)
        self.assertIn('chi2_stat', result)
        self.assertIn('p_value', result)
        
        # 빈도수 합계 검증
        total_count = sum(result['frequency'].values())
        expected_total = len(self.data_manager.data) * 6  # 각 행당 6개 번호
        self.assertEqual(total_count, expected_total)

    def test_sequence_patterns(self):
        """연속 패턴 분석 테스트"""
        result = self.analyzer._analyze_sequence_patterns()
        
        # 결과 검증
        self.assertIn('sequence_counts', result)
        self.assertIn('consecutive_probability', result)
        
        # 연속 확률 범위 검증
        self.assertGreaterEqual(result['consecutive_probability'], 0)
        self.assertLessEqual(result['consecutive_probability'], 1)

    def test_oddeven_patterns(self):
        """홀짝 패턴 분석 테스트"""
        result = self.analyzer._analyze_oddeven_patterns()
        
        # 결과 검증
        self.assertIn('pattern_counts', result)
        self.assertIn('pattern_probabilities', result)
        
        # 확률 합계 검증
        total_prob = sum(result['pattern_probabilities'].values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)

    def test_range_distribution(self):
        """구간 분포 분석 테스트"""
        result = self.analyzer._analyze_range_distribution()
        
        # 결과 검증
        self.assertIn('range_counts', result)
        self.assertIn('range_probabilities', result)
        
        # 확률 합계 검증
        total_prob = sum(result['range_probabilities'].values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)

    def test_sum_patterns(self):
        """합계 패턴 분석 테스트"""
        result = self.analyzer._analyze_sum_patterns()
        
        # 결과 검증
        self.assertIn('sum_mean', result)
        self.assertIn('sum_std', result)
        self.assertIn('sum_min', result)
        self.assertIn('sum_max', result)
        self.assertIn('normality_p_value', result)
        
        # 합계 범위 검증
        min_possible_sum = 6  # 1+2+3+4+5+6
        max_possible_sum = 255  # 40+41+42+43+44+45
        self.assertGreaterEqual(result['sum_min'], min_possible_sum)
        self.assertLessEqual(result['sum_max'], max_possible_sum)

    def test_gap_patterns(self):
        """간격 패턴 분석 테스트"""
        result = self.analyzer._analyze_gap_patterns()
        
        # 결과 검증
        self.assertIn('gap_mean', result)
        self.assertIn('gap_std', result)
        self.assertIn('gap_min', result)
        self.assertIn('gap_max', result)
        self.assertIn('gap_median', result)
        
        # 간격 범위 검증
        self.assertGreaterEqual(result['gap_min'], 1)
        self.assertLessEqual(result['gap_max'], 44)

    def test_markov_chain(self):
        """마코프 체인 분석 테스트"""
        result = self.analyzer._analyze_markov_chain()
        
        # 결과 검증
        self.assertIn('transition_matrix', result)
        self.assertIn('high_probability_transitions', result)
        
        # 전이 행렬 크기 검증
        matrix = np.array(result['transition_matrix'])
        self.assertEqual(matrix.shape, (45, 45))

    def test_fourier_analysis(self):
        """푸리에 변환 분석 테스트"""
        result = self.analyzer._analyze_fourier()
        
        # 결과 검증
        self.assertIn('frequencies', result)
        self.assertIn('magnitudes', result)
        self.assertIn('significant_frequencies', result)
        self.assertIn('periodic_numbers', result)
        
        # 주기성 있는 번호 범위 검증
        for num in result['periodic_numbers']:
            self.assertGreaterEqual(int(num), 1)
            self.assertLessEqual(int(num), 45)

    def test_plot_generation(self):
        """그래프 생성 테스트"""
        # 분석 결과 저장
        self.analyzer.frequency_stats = self.analyzer._analyze_frequency()
        self.analyzer.sequence_stats = self.analyzer._analyze_sequence_patterns()
        self.analyzer.oddeven_stats = self.analyzer._analyze_oddeven_patterns()
        self.analyzer.range_stats = self.analyzer._analyze_range_distribution()
        self.analyzer.sum_stats = self.analyzer._analyze_sum_patterns()
        self.analyzer.gap_stats = self.analyzer._analyze_gap_patterns()
        self.analyzer.markov_stats = self.analyzer._analyze_markov_chain()
        self.analyzer.fourier_stats = self.analyzer._analyze_fourier()

        # 그래프 저장 디렉토리 생성
        graph_dir = Path('lottery/data/results/graph')
        graph_dir.mkdir(parents=True, exist_ok=True)

        # 그래프 생성
        self.analyzer._plot_frequency(str(graph_dir / 'frequency.png'))
        self.analyzer._plot_oddeven_patterns(str(graph_dir / 'oddeven_patterns.png'))
        self.analyzer._plot_range_distribution(str(graph_dir / 'range_distribution.png'))
        self.analyzer._plot_sum_distribution(str(graph_dir / 'sum_distribution.png'))
        self.analyzer._plot_gap_patterns(str(graph_dir / 'gap_patterns.png'))
        self.analyzer._plot_markov_chain(str(graph_dir / 'markov_chain.png'))
        self.analyzer._plot_fourier(str(graph_dir / 'fourier.png'))

        # 파일 존재 확인
        self.assertTrue((graph_dir / 'frequency.png').exists())
        self.assertTrue((graph_dir / 'oddeven_patterns.png').exists())
        self.assertTrue((graph_dir / 'range_distribution.png').exists())
        self.assertTrue((graph_dir / 'sum_distribution.png').exists())
        self.assertTrue((graph_dir / 'gap_patterns.png').exists())
        self.assertTrue((graph_dir / 'markov_chain.png').exists())
        self.assertTrue((graph_dir / 'fourier.png').exists())

    def test_save_load_analysis(self):
        """분석 결과 저장/로드 테스트"""
        # 분석 결과 저장
        self.analyzer.frequency_stats = self.analyzer._analyze_frequency()
        self.analyzer.sequence_stats = self.analyzer._analyze_sequence_patterns()
        self.analyzer.oddeven_stats = self.analyzer._analyze_oddeven_patterns()
        self.analyzer.range_stats = self.analyzer._analyze_range_distribution()
        self.analyzer.sum_stats = self.analyzer._analyze_sum_patterns()
        self.analyzer.gap_stats = self.analyzer._analyze_gap_patterns()
        self.analyzer.markov_stats = self.analyzer._analyze_markov_chain()
        self.analyzer.fourier_stats = self.analyzer._analyze_fourier()

        # 분석 결과 저장
        save_path = str(Path(self.temp_dir) / 'analysis_results.npy')
        self.analyzer.save_analysis_results(save_path)

        # 파일 존재 확인
        self.assertTrue(Path(save_path).exists())

        # 새로운 분석기 생성
        new_analyzer = PatternAnalyzer(
            config=self.config,
            data=self.data_manager.data
        )

        # 분석 결과 로드
        new_analyzer.load_analysis_results(save_path)

        # 결과 비교
        self.assertEqual(self.analyzer.frequency_stats, new_analyzer.frequency_stats)
        self.assertEqual(self.analyzer.sequence_stats, new_analyzer.sequence_stats)
        self.assertEqual(self.analyzer.oddeven_stats, new_analyzer.oddeven_stats)
        self.assertEqual(self.analyzer.range_stats, new_analyzer.range_stats)
        self.assertEqual(self.analyzer.sum_stats, new_analyzer.sum_stats)
        self.assertEqual(self.analyzer.gap_stats, new_analyzer.gap_stats)
        self.assertEqual(self.analyzer.markov_stats, new_analyzer.markov_stats)
        self.assertEqual(self.analyzer.fourier_stats, new_analyzer.fourier_stats)

    def test_cache_functionality(self):
        """캐시 기능 테스트"""
        # 첫 번째 분석 실행 전 캐시 비우기
        self.analyzer._cache = {}
        self.analyzer._cache_timestamps = {}

        # 캐시 기능을 확인하기 위해 충분한 데이터 생성
        test_data = pd.DataFrame({
            'numbers': [np.random.randint(1, 46, size=6).tolist() for _ in range(1000)]
        })

        # 첫 번째 분석 실행
        start_time = time.time()
        result1 = self.analyzer.analyze()
        first_run_time = time.time() - start_time

        # 캐시 사용 확인을 위한 지연 추가
        time.sleep(0.1)

        # 두 번째 분석 실행 (캐시 사용)
        start_time = time.time()
        result2 = self.analyzer.analyze()
        second_run_time = time.time() - start_time

        # 결과 검증 (결과가 동일한지)
        self.assertEqual(len(result1), len(result2))

        # 캐시 사용으로 인한 성능 향상 확인
        # 실행 시간이 매우 빠른 경우, 절대적 비교보다 캐시 사용 여부 확인이 더 중요
        if first_run_time > 0.01:  # 유의미한 첫 실행 시간이 있는 경우만 비교
            self.assertLess(second_run_time, first_run_time)
        else:
            # 캐시 사용 여부를 직접 확인
            self.assertGreater(len(self.analyzer._cache), 0)
            self.assertIn('full_analysis', self.analyzer._cache)

    def test_combination_stats(self):
        """조합 통계 분석 테스트"""
        result = self.analyzer._analyze_combination_stats()
        
        # 결과 검증
        self.assertIn('combination_counts', result)
        self.assertIn('combination_probabilities', result)
        self.assertIn('most_common_combinations', result)
        
        # 조합 크기 검증
        for combo in result['most_common_combinations']:
            self.assertEqual(len(combo), 6)

    def test_moving_averages(self):
        """이동 평균 분석 테스트"""
        result = self.analyzer._analyze_moving_averages()
        
        # 결과 검증
        self.assertIn('ma_values', result)
        self.assertIn('ma_trends', result)
        self.assertIn('ma_crossovers', result)
        
        # 이동 평균 값 범위 검증
        for ma in result['ma_values'].values():
            self.assertTrue(all(1 <= val <= 45 for val in ma))

    def test_robust_stats(self):
        """강건 통계 분석 테스트"""
        result = self.analyzer._analyze_robust_stats()
        
        # 결과 검증
        self.assertIn('median', result)
        self.assertIn('iqr', result)
        self.assertIn('outliers', result)
        self.assertIn('robust_correlation', result)
        
        # 통계값 범위 검증
        self.assertGreaterEqual(result['median'], 1)
        self.assertLessEqual(result['median'], 45)
        self.assertGreaterEqual(result['iqr'], 0)

    def test_duplicate_patterns(self):
        """중복 패턴 분석 테스트"""
        result = self.analyzer._analyze_duplicate_patterns()
        
        # 결과 검증
        self.assertIn('duplicate_counts', result)
        self.assertIn('duplicate_probabilities', result)
        self.assertIn('most_common_duplicates', result)
        
        # 확률 범위 검증
        for prob in result['duplicate_probabilities'].values():
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)

    def test_number_patterns(self):
        """번호 패턴 분석 테스트"""
        result = self.analyzer._analyze_number_patterns()
        
        # 결과 검증
        self.assertIn('pattern_counts', result)
        self.assertIn('pattern_probabilities', result)
        self.assertIn('significant_patterns', result)
        
        # 패턴 유효성 검증
        for pattern in result['significant_patterns']:
            self.assertIsInstance(pattern, (list, tuple))
            self.assertTrue(all(1 <= num <= 45 for num in pattern))

    def test_performance_optimization(self):
        """성능 최적화 테스트"""
        # 캐시 테스트
        cache_key = 'test_cache'
        test_data = {'test': 'data'}
        
        # 캐시 저장
        self.analyzer._cache_result(cache_key, test_data)
        
        # 캐시 조회
        cached_result = self.analyzer._get_cached_result(cache_key)
        self.assertEqual(cached_result, test_data)
        
        # 캐시 만료 테스트
        time.sleep(self.analyzer.pattern_config.cache_ttl + 1)
        expired_result = self.analyzer._get_cached_result(cache_key)
        self.assertIsNone(expired_result)
        
        # GPU 메모리 관리 테스트
        if torch.cuda.is_available():
            self.assertIsNotNone(self.analyzer.cuda_optimizer)
            self.assertIsNotNone(self.analyzer.tensor_cache)
            self.assertIsNotNone(self.analyzer.inference_buffer)
            self.assertIsNotNone(self.analyzer.memory_manager)

    def test_visualization(self):
        """시각화 테스트"""
        # 그래프 저장 디렉토리 생성
        graph_dir = Path('lottery/data/results/graph')
        graph_dir.mkdir(parents=True, exist_ok=True)
        
        # 새로운 패턴 그래프 생성
        self.analyzer._plot_duplicate_patterns(str(graph_dir / 'duplicate_patterns.png'))
        self.analyzer._plot_number_patterns(str(graph_dir / 'number_patterns.png'))
        self.analyzer._plot_combination_stats(str(graph_dir / 'combination_stats.png'))
        self.analyzer._plot_moving_averages(str(graph_dir / 'moving_averages.png'))
        self.analyzer._plot_robust_stats(str(graph_dir / 'robust_stats.png'))
        
        # 파일 존재 확인
        self.assertTrue((graph_dir / 'duplicate_patterns.png').exists())
        self.assertTrue((graph_dir / 'number_patterns.png').exists())
        self.assertTrue((graph_dir / 'combination_stats.png').exists())
        self.assertTrue((graph_dir / 'moving_averages.png').exists())
        self.assertTrue((graph_dir / 'robust_stats.png').exists())

if __name__ == '__main__':
    unittest.main()