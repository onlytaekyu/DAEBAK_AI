"""
오류 처리 및 로깅 유틸리티

콘솔 로그는 상세하게 표시하고, ERROR 레벨 이상만 파일에 기록합니다.
"""

import logging
import traceback
import os
import sys
from functools import wraps
from pathlib import Path
import datetime
import functools
import time
from typing import Any, Callable, Optional, TypeVar, Union
import torch

# 로깅 설정
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 컬러 로깅 설정 (Windows 콘솔에서도 작동)
if sys.platform == 'win32':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ANSI 컬러 코드
COLORS = {
    'DEBUG': '\033[94m',  # 파란색
    'INFO': '\033[92m',   # 녹색
    'WARNING': '\033[93m', # 노란색
    'ERROR': '\033[91m',  # 빨간색
    'CRITICAL': '\033[41m\033[97m', # 배경 빨간색, 글자 흰색
    'RESET': '\033[0m'    # 리셋
}

class ColoredFormatter(logging.Formatter):
    """컬러 로그 포매터"""
    
    def format(self, record):
        levelname = record.levelname
        message = super().format(record)
        
        if levelname in COLORS:
            return f"{COLORS[levelname]}{message}{COLORS['RESET']}"
        return message

# 현재 시간 기반 로그 파일명
log_filename = LOG_DIR / f"lottery_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 콘솔 핸들러 (모든 레벨)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = ColoredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# 파일 핸들러 (ERROR 이상만)
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.ERROR)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# 로깅 설정 초기화
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # 루트 로거는 모든 메시지를 받음
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

def get_logger(name):
    """모듈별 로거 생성"""
    logger = logging.getLogger(name)
    return logger

def safe_execute(default_return=None, reraise=False):
    """
    함수 실행을 안전하게 처리하는 데코레이터
    
    Args:
        default_return: 오류 발생 시 반환할 기본값
        reraise: 예외를 다시 발생시킬지 여부
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 상세 오류 정보 로깅
                error_message = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_message)
                
                # 스택 트레이스 로깅
                stack_trace = traceback.format_exc()
                logger.error(f"Stack trace:\n{stack_trace}")
                
                # 함수 인자 정보 로깅 (민감 정보 주의)
                arg_info = f"Function arguments: args={args}, kwargs={kwargs}"
                logger.error(arg_info)
                
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator

# 성능 로깅 데코레이터
def log_performance(func):
    """함수 실행 시간을 로깅하는 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Starting {func.__name__}")
        
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"Finished {func.__name__} in {execution_time:.4f} seconds")
        
        return result
    return wrapper 

# 로깅 설정
def setup_logger(name: str) -> logging.Logger:
    """로거 설정
    
    Args:
        name: 로거 이름
        
    Returns:
        설정된 로거
    """
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 (WARNING 이상만 표시)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # INFO에서 WARNING으로 변경
    
    # 파일 핸들러 (모든 로그 기록)
    file_handler = logging.FileHandler('lottery/logs/app.log')
    file_handler.setLevel(logging.DEBUG)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 성능 측정 데코레이터
def log_performance(func: Callable) -> Callable:
    """
    성능 측정 데코레이터
    
    특징:
    1. 실행 시간 측정
    2. 메모리 사용량 측정
    3. 성능 메트릭 기록
    4. 예외 처리
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = None
        
        try:
            # 메모리 사용량 측정 시작
            if torch.cuda.is_available():
                start_memory = torch.cuda.memory_allocated()
            
            # 함수 실행
            result = func(*args, **kwargs)
            
            # 성능 메트릭 계산
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 메모리 사용량 계산
            memory_usage = None
            if start_memory is not None:
                end_memory = torch.cuda.memory_allocated()
                memory_usage = (end_memory - start_memory) / 1024**2  # MB 단위
            
            # 성능 메트릭 로깅
            logger = logging.getLogger(func.__module__)
            logger.info(
                f"함수 {func.__name__} 실행 완료: "
                f"시간={execution_time:.3f}초, "
                f"메모리={memory_usage:.2f}MB"
            )
            
            return result
            
        except Exception as e:
            # 예외 발생 시 성능 메트릭 기록
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger = logging.getLogger(func.__module__)
            logger.error(
                f"함수 {func.__name__} 실행 실패: "
                f"시간={execution_time:.3f}초, "
                f"오류={str(e)}"
            )
            raise
            
    return wrapper

# 안전한 실행 데코레이터
T = TypeVar('T')

def safe_execute(default_return: Optional[T] = None) -> Callable:
    """
    안전한 실행 데코레이터
    
    특징:
    1. 예외 처리
    2. 기본값 반환
    3. 스택 트레이스 로깅
    4. 메모리 정리
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 예외 정보 로깅
                logger = logging.getLogger(func.__module__)
                logger.error(
                    f"함수 {func.__name__} 실행 중 오류 발생:\n"
                    f"오류: {str(e)}\n"
                    f"스택 트레이스:\n{traceback.format_exc()}"
                )
                
                # 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return default_return
        return wrapper
    return decorator

# 로거 인스턴스 생성
logger = setup_logger('daebak_ai')

# 전역 예외 핸들러
def global_exception_handler(exctype, value, traceback_obj):
    """
    전역 예외 핸들러
    
    특징:
    1. 예외 정보 로깅
    2. 스택 트레이스 기록
    3. 메모리 정리
    4. 시스템 상태 기록
    """
    logger.error(
        f"예외 발생:\n"
        f"타입: {exctype.__name__}\n"
        f"메시지: {str(value)}\n"
        f"스택 트레이스:\n{''.join(traceback.format_tb(traceback_obj))}"
    )
    
    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 시스템 상태 기록
    if torch.cuda.is_available():
        logger.info(
            f"GPU 메모리 상태:\n"
            f"할당: {torch.cuda.memory_allocated() / 1024**2:.2f}MB\n"
            f"캐시: {torch.cuda.memory_reserved() / 1024**2:.2f}MB"
        )

# 전역 예외 핸들러 등록
sys.excepthook = global_exception_handler 