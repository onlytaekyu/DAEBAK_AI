"""
로또 번호 예측을 위한 데이터 로더

이 모듈은 로또 번호 데이터를 로드하고 전처리하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

from .config import Config

# 로거 설정
logger = logging.getLogger(__name__)

class LotteryDataset(Dataset):
    """로또 번호 데이터셋"""

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        transform: Optional[Any] = None
    ):
        """
        데이터셋 초기화

        Args:
            data: 입력 데이터
            targets: 타겟 데이터
            transform: 데이터 변환 함수
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

class DataManager:
    """데이터 관리자"""

    def __init__(self, config: Config):
        """
        데이터 관리자 초기화

        Args:
            config: 설정 객체
        """
        self.config = config
        self.data_config = config.data
        self.scaler = MinMaxScaler()
        self.data = None
        self.targets = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self) -> None:
        """데이터 로드"""
        try:
            data_path = Path(self.data_config.historical_data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

            self.data = pd.read_csv(data_path)
            self._validate_data(self.data)
            self._preprocess_data()
            logger.info(f"데이터 로드 완료: {len(self.data)} 행")
        except Exception as e:
            logger.error(f"데이터 로드 실패: {str(e)}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        데이터 유효성 검사

        Args:
            df: 검사할 데이터프레임
        """
        required_columns = ['seqNum', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

        # 숫자 범위 검사
        number_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        for col in number_columns:
            if not df[col].between(self.data_config.min_number, self.data_config.max_number).all():
                raise ValueError(f"숫자 범위가 잘못되었습니다: {col}")

        # 중복 번호 검사
        for idx, row in df.iterrows():
            numbers = [row[col] for col in number_columns]
            if len(set(numbers)) != len(numbers):
                raise ValueError(f"중복된 번호가 있습니다: {numbers} (회차: {row['seqNum']})")

    def _preprocess_data(self) -> None:
        """데이터 전처리"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")

        # 숫자 컬럼을 리스트로 변환
        number_columns = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6']
        self.processed_data = self.data.copy()
        self.processed_data['numbers'] = self.processed_data[number_columns].values.tolist()

        # 날짜 컬럼 추가 (회차 번호를 기준으로)
        self.processed_data['date'] = pd.to_datetime('2002-12-07') + pd.to_timedelta(self.processed_data['seqNum'] * 7, unit='D')

        # 필요한 컬럼만 선택
        self.processed_data = self.processed_data[['date', 'numbers']]

        # 전처리된 데이터를 self.data에도 저장
        self.data = self.processed_data.copy()

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터 반환

        Returns:
            입력 데이터와 타겟 데이터의 튜플
        """
        if self.processed_data is None:
            raise ValueError("전처리된 데이터가 없습니다.")

        X = np.array(self.processed_data['numbers'].tolist())
        y = np.array(self.processed_data['numbers'].tolist()[1:] + [self.processed_data['numbers'].iloc[-1]])

        return X, y

    def get_latest_numbers(self) -> List[int]:
        """
        최신 당첨 번호 반환

        Returns:
            최신 당첨 번호 리스트
        """
        if self.processed_data is None:
            raise ValueError("전처리된 데이터가 없습니다.")

        return self.processed_data['numbers'].iloc[-1]

    def save_processed_data(self) -> None:
        """전처리된 데이터 저장"""
        if self.processed_data is None:
            raise ValueError("전처리된 데이터가 없습니다.")

        save_path = Path(self.data_config.processed_data_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_data.to_pickle(save_path)
        logger.info(f"전처리된 데이터 저장 완료: {save_path}")

    def load_processed_data(self) -> None:
        """전처리된 데이터 로드"""
        try:
            save_path = Path(self.data_config.processed_data_path)
            if not save_path.exists():
                raise FileNotFoundError(f"전처리된 데이터 파일을 찾을 수 없습니다: {save_path}")

            self.processed_data = pd.read_pickle(save_path)
            logger.info(f"전처리된 데이터 로드 완료: {len(self.processed_data)} 행")
        except Exception as e:
            logger.error(f"전처리된 데이터 로드 실패: {str(e)}")
            raise

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        데이터 로더 반환

        Returns:
            (train_loader, val_loader, test_loader) 튜플
        """
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise ValueError("데이터 로더가 초기화되지 않았습니다")

        return self.train_loader, self.val_loader, self.test_loader

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        정규화된 데이터를 원래 스케일로 변환

        Args:
            data: 변환할 데이터 배열

        Returns:
            변환된 데이터 배열
        """
        try:
            return self.scaler.inverse_transform(data)
        except Exception as e:
            logger.error(f"데이터 역변환 중 오류 발생: {str(e)}")
            raise