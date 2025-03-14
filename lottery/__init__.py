"""
로또 번호 예측 시스템

이 패키지는 로또 번호 예측을 위한 다양한 분석 및 예측 기능을 제공합니다.
"""

from pathlib import Path
from .src.utils.config import Config
from .src.analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisConfig
from .src.utils.data_loader import DataManager

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).parent

# 버전
__version__ = "1.0.0"

__all__ = ['Config', 'PatternAnalyzer', 'PatternAnalysisConfig', 'DataManager']