"""
베이지안 추론 테스트 모듈

이 모듈은 베이지안 추론과 베이지안 최적화의 기능을 테스트합니다.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from lottery.src.utils.config import Config
from lottery.src.bayesian.bayesian_inference import BayesianInference, BayesianOptimizer

class TestBayesianInference(unittest.TestCase):
    """베이지안 추론 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 임시 디렉토리 생성
        cls.temp_dir = tempfile.mkdtemp()
        
        # 설정 객체 생성
        config_dict = {
            'bayesian': {
                'prior_mean': 0.0,
                'prior_std': 1.0,
                'likelihood_std': 0.1,
                'proposal_std': 0.1,
                'n_initial_points': 10,
                'n_iterations': 50,
                'noise_level': 0.1
            },
            'cuda': {
                'use_gpu': torch.cuda.is_available()
            }
        }
        cls.config = Config(config_dict)
        
        # 테스트 데이터 생성
        np.random.seed(42)
        cls.X = torch.randn(100, 5)  # 5차원 입력
        cls.true_params = torch.tensor([1.0, -0.5, 0.2, 0.8, -0.3])
        cls.y = cls.X @ cls.true_params + torch.randn(100) * 0.1  # 노이즈 추가
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """각 테스트 메서드 실행 전 초기화"""
        self.inference = BayesianInference(self.config)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.inference)
        self.assertEqual(self.inference.config, self.config)
        self.assertIsNone(self.inference.posterior_samples)
        self.assertIsNone(self.inference.posterior_mean)
        self.assertIsNone(self.inference.posterior_std)
    
    def test_update_posterior(self):
        """사후 분포 업데이트 테스트"""
        posterior_samples, posterior_mean, posterior_std = self.inference.update_posterior(
            self.X, self.y
        )
        
        # 결과 검증
        self.assertIsNotNone(posterior_samples)
        self.assertIsNotNone(posterior_mean)
        self.assertIsNotNone(posterior_std)
        
        # 차원 검증
        self.assertEqual(posterior_samples.shape[1], 5)  # 5차원 매개변수
        self.assertEqual(posterior_mean.shape[0], 5)
        self.assertEqual(posterior_std.shape[0], 5)
        
        # 실제 매개변수와 추정값 비교
        for i in range(5):
            self.assertAlmostEqual(
                posterior_mean[i].item(),
                self.true_params[i].item(),
                places=1
            )
    
    def test_predict(self):
        """예측 테스트"""
        # 사후 분포 업데이트
        self.inference.update_posterior(self.X, self.y)
        
        # 테스트 데이터 생성
        X_test = torch.randn(10, 5)
        
        # 예측 수행
        pred_mean, pred_std = self.inference.predict(X_test)
        
        # 결과 검증
        self.assertEqual(pred_mean.shape[0], 10)
        self.assertEqual(pred_std.shape[0], 10)
        self.assertTrue(torch.all(pred_std >= 0))  # 표준편차는 항상 양수
    
    def test_save_load_results(self):
        """결과 저장/로드 테스트"""
        # 사후 분포 업데이트
        self.inference.update_posterior(self.X, self.y)
        
        # 결과 저장
        save_path = str(Path(self.temp_dir) / 'inference_results.pt')
        self.inference.save_inference_results(save_path)
        
        # 파일 존재 확인
        self.assertTrue(Path(save_path).exists())
        
        # 새로운 추론 객체 생성
        new_inference = BayesianInference(self.config)
        
        # 결과 로드
        new_inference.load_inference_results(save_path)
        
        # 결과 비교
        torch.testing.assert_close(
            self.inference.posterior_samples,
            new_inference.posterior_samples
        )
        torch.testing.assert_close(
            self.inference.posterior_mean,
            new_inference.posterior_mean
        )
        torch.testing.assert_close(
            self.inference.posterior_std,
            new_inference.posterior_std
        )

class TestBayesianOptimizer(unittest.TestCase):
    """베이지안 최적화 테스트"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 초기화"""
        # 설정 객체 생성
        config_dict = {
            'bayesian': {
                'n_initial_points': 5,
                'n_iterations': 20,
                'noise_level': 0.1
            },
            'cuda': {
                'use_gpu': torch.cuda.is_available()
            }
        }
        cls.config = Config(config_dict)
    
    def setUp(self):
        """각 테스트 메서드 실행 전 초기화"""
        self.optimizer = BayesianOptimizer(self.config)
    
    def test_initialization(self):
        """초기화 테스트"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.config, self.config)
        self.assertEqual(len(self.optimizer.X_samples), 0)
        self.assertEqual(len(self.optimizer.y_samples), 0)
        self.assertIsNone(self.optimizer.best_params)
        self.assertEqual(self.optimizer.best_value, float('-inf'))
    
    def test_optimization(self):
        """최적화 테스트"""
        # 목적 함수 정의
        def objective(x):
            return -(x ** 2).sum()  # 단순한 2차 함수
        
        # 최적화 범위 설정
        bounds = torch.stack([
            torch.ones(5) * -5,
            torch.ones(5) * 5
        ]).t()
        
        # 최적화 수행
        best_params, best_value = self.optimizer.optimize(objective, bounds)
        
        # 결과 검증
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(best_value)
        self.assertEqual(best_params.shape[0], 5)
        self.assertGreater(len(self.optimizer.X_samples), 0)
        self.assertGreater(len(self.optimizer.y_samples), 0)
        
        # 최적값이 0에 가까운지 확인 (2차 함수의 최대값)
        self.assertAlmostEqual(best_value.item(), 0.0, places=1)
    
    def test_initial_sampling(self):
        """초기 샘플링 테스트"""
        bounds = torch.stack([
            torch.ones(5) * -5,
            torch.ones(5) * 5
        ]).t()
        
        n_samples = 10
        samples = self.optimizer._initial_sampling(bounds, n_samples)
        
        # 결과 검증
        self.assertEqual(samples.shape, (n_samples, 5))
        self.assertTrue(torch.all(samples >= bounds[:, 0]))
        self.assertTrue(torch.all(samples <= bounds[:, 1]))

if __name__ == '__main__':
    unittest.main() 