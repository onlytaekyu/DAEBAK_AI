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
from pathlib import Path
import time

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
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
                'processed_data_path': f'{cls.temp_dir}/processed_data.pkl',
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
        cls.config = Config(config_dict)

        # 데이터 매니저 초기화 및 데이터 로드
        cls.data_manager = DataManager(cls.config)
        cls.data_manager.load_data()

        # 패턴 분석기 초기화
        cls.analyzer = PatternAnalyzer(
            config=cls.config,
            data=cls.data_manager.data
        )

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
        self.assertIn('duplicate_patterns', results)
        self.assertIn('number_patterns', results)
        
        # 새로운 분석 결과 테스트
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
        self.assertEqual(len(result['frequency']), 45)  # 1-45 번호
        self.assertEqual(len(result['probabilities']), 45)
        self.assertTrue(np.allclose(sum(result['probabilities'].values()), 1.0, rtol=1e-5))

    def test_sequence_patterns(self):
        """연속 패턴 분석 테스트"""
        result = self.analyzer._analyze_sequence_patterns()

        # 결과 검증
        self.assertIn('sequence_counts', result)
        self.assertIn('consecutive_probability', result)
        self.assertGreaterEqual(len(result['sequence_counts']), 0)

    def test_oddeven_patterns(self):
        """홀짝 패턴 분석 테스트"""
        result = self.analyzer._analyze_oddeven_patterns()

        # 결과 검증
        self.assertIn('pattern_counts', result)
        self.assertIn('pattern_probabilities', result)
        self.assertTrue(np.allclose(sum(result['pattern_probabilities'].values()), 1.0, rtol=1e-5))

    def test_range_distribution(self):
        """구간 분포 분석 테스트"""
        result = self.analyzer._analyze_range_distribution()

        # 결과 검증
        self.assertIn('range_counts', result)
        self.assertIn('range_probabilities', result)
        self.assertEqual(len(result['range_counts']), 5)  # 5개 구간
        self.assertEqual(len(result['range_probabilities']), 5)
        self.assertTrue(np.allclose(sum(result['range_probabilities'].values()), 1.0, rtol=1e-5))

    def test_sum_patterns(self):
        """합계 패턴 분석 테스트"""
        result = self.analyzer._analyze_sum_patterns()

        # 결과 검증
        self.assertIn('sum_mean', result)
        self.assertIn('sum_std', result)
        self.assertIn('sum_min', result)
        self.assertIn('sum_max', result)
        self.assertIn('normality_p_value', result)
        self.assertGreater(result['sum_mean'], 0)

    def test_gap_patterns(self):
        """간격 패턴 분석 테스트"""
        result = self.analyzer._analyze_gap_patterns()

        # 결과 검증
        self.assertIn('gap_mean', result)
        self.assertIn('gap_std', result)
        self.assertIn('gap_min', result)
        self.assertIn('gap_max', result)
        self.assertIn('gap_median', result)
        self.assertGreater(result['gap_mean'], 0)

    def test_markov_chain(self):
        """마코프 체인 분석 테스트"""
        result = self.analyzer._analyze_markov_chain()

        # 결과 검증
        self.assertIn('transition_matrix', result)
        self.assertIn('high_probability_transitions', result)
        self.assertEqual(len(result['transition_matrix']), 45)
        self.assertEqual(len(result['transition_matrix'][0]), 45)

    def test_fourier_analysis(self):
        """푸리에 분석 테스트"""
        result = self.analyzer._analyze_fourier()

        # 결과 검증
        self.assertIn('frequencies', result)
        self.assertIn('magnitudes', result)
        self.assertIn('significant_frequencies', result)
        self.assertIn('periodic_numbers', result)
        self.assertGreater(len(result['frequencies']), 0)
        self.assertGreater(len(result['magnitudes']), 0)

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
        """번호 조합 통계 분석 테스트"""
        results = self.analyzer._analyze_combination_stats()
        
        # 기본 통계량 테스트
        self.assertIn('basic_stats', results)
        basic_stats = results['basic_stats']
        self.assertIn('variance', basic_stats)
        self.assertIn('std_dev', basic_stats)
        self.assertIn('range', basic_stats)
        self.assertIn('median', basic_stats)
        self.assertIn('skewness', basic_stats)
        self.assertIn('kurtosis', basic_stats)
        
        # 시간 시계열 통계 테스트
        self.assertIn('time_series_stats', results)
        time_stats = results['time_series_stats']
        self.assertIn('variance', time_stats)
        self.assertIn('std_dev', time_stats)
        self.assertIn('range', time_stats)
        self.assertIn('median', time_stats)
        self.assertIn('skewness', time_stats)
        self.assertIn('kurtosis', time_stats)
        
        # 데이터 타입 및 범위 검증
        for stat in basic_stats.values():
            self.assertIsInstance(stat, (int, float))
        
        for stat_list in time_stats.values():
            self.assertIsInstance(stat_list, list)
            self.assertTrue(all(isinstance(x, (int, float)) for x in stat_list))

    def test_moving_averages(self):
        """이동 평균 분석 테스트"""
        results = self.analyzer._analyze_moving_averages()
        
        # 이동 평균 결과 테스트
        self.assertIn('moving_averages', results)
        ma_results = results['moving_averages']
        
        # 각 기간별 결과 테스트
        for period in [5, 10, 20]:
            period_key = f'ma_{period}'
            self.assertIn(period_key, ma_results)
            period_data = ma_results[period_key]
            
            self.assertIn('moving_averages', period_data)
            self.assertIn('cross_points', period_data)
            self.assertIn('trend_strength', period_data)
            
            # 데이터 타입 및 범위 검증
            self.assertIsInstance(period_data['moving_averages'], list)
            self.assertIsInstance(period_data['cross_points'], list)
            self.assertIsInstance(period_data['trend_strength'], list)
            
            # 이동 평균 길이 검증
            self.assertEqual(len(period_data['moving_averages']), len(self.test_data))
        
        # 트렌드 방향성 테스트
        self.assertIn('trend_direction', results)
        trend_direction = results['trend_direction']
        for period in [5, 10, 20]:
            period_key = f'ma_{period}'
            self.assertIn(period_key, trend_direction)
            self.assertIsInstance(trend_direction[period_key], list)
            self.assertEqual(len(trend_direction[period_key]), len(self.test_data))

    def test_robust_stats(self):
        """로버스트 통계 분석 테스트"""
        results = self.analyzer._analyze_robust_stats()
        
        # 로버스트 통계량 테스트
        self.assertIn('robust_stats', results)
        robust_stats = results['robust_stats']
        self.assertIn('median', robust_stats)
        self.assertIn('q1', robust_stats)
        self.assertIn('q3', robust_stats)
        self.assertIn('iqr', robust_stats)
        self.assertIn('mad', robust_stats)
        
        # 윈저화된 통계량 테스트
        self.assertIn('winsorized_stats', results)
        winsorized_stats = results['winsorized_stats']
        self.assertIn('mean', winsorized_stats)
        self.assertIn('std', winsorized_stats)
        self.assertIn('median', winsorized_stats)
        self.assertIn('q1', winsorized_stats)
        self.assertIn('q3', winsorized_stats)
        
        # 극단값 제거 후 통계량 테스트
        self.assertIn('clean_stats', results)
        clean_stats = results['clean_stats']
        self.assertIn('mean', clean_stats)
        self.assertIn('std', clean_stats)
        self.assertIn('median', clean_stats)
        self.assertIn('q1', clean_stats)
        self.assertIn('q3', clean_stats)
        
        # 극단값 개수 테스트
        self.assertIn('outlier_count', results)
        self.assertIsInstance(results['outlier_count'], int)
        
        # 데이터 타입 및 범위 검증
        for stat_dict in [robust_stats, winsorized_stats, clean_stats]:
            for stat in stat_dict.values():
                self.assertIsInstance(stat, (int, float))
        
        # 통계적 관계 검증
        self.assertGreaterEqual(robust_stats['q3'], robust_stats['median'])
        self.assertGreaterEqual(robust_stats['median'], robust_stats['q1'])
        self.assertGreaterEqual(robust_stats['iqr'], 0)
        self.assertGreaterEqual(robust_stats['mad'], 0)

if __name__ == '__main__':
    unittest.main()