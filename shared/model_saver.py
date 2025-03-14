"""
모델 저장 및 로드 유틸리티

이 모듈은 모델의 저장과 로드를 관리하며, 버전 관리와 호환성을 제공합니다.
"""

import os
import torch
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from packaging import version

# 로거 설정
logger = logging.getLogger(__name__)

class ModelSaver:
    """모델 저장 및 로드 관리자"""
    
    CURRENT_VERSION = "1.0.0"
    
    @staticmethod
    def save_model(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        모델 저장 (버전 관리 강화)
        
        Args:
            model: PyTorch 모델
            optimizer: 옵티마이저
            config: 모델 설정
            filepath: 저장 경로
            metadata: 추가 메타데이터
        """
        save_dict = {
            'version': ModelSaver.CURRENT_VERSION,
            'created_at': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'metadata': metadata or {}
        }
        
        try:
            # 저장 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 모델 저장
            torch.save(save_dict, filepath)
            
            # 설정 파일 별도 저장
            config_path = Path(filepath).with_suffix('.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'version': ModelSaver.CURRENT_VERSION,
                    'config': config,
                    'metadata': metadata or {}
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"모델 저장 완료: {filepath}")
            
        except Exception as e:
            logger.error(f"모델 저장 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def load_model(
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        filepath: str
    ) -> Dict[str, Any]:
        """
        모델 로드 (하위 호환성 관리)
        
        Args:
            model: PyTorch 모델
            optimizer: 옵티마이저 (선택사항)
            filepath: 로드 경로
            
        Returns:
            로드된 설정 및 메타데이터
        """
        try:
            # 체크포인트 로드
            checkpoint = torch.load(filepath)
            
            # 버전 확인 및 호환성 처리
            saved_version = checkpoint.get('version', '0.0.0')
            if version.parse(saved_version) < version.parse('1.0.0'):
                logger.warning(f"이전 버전({saved_version})의 모델 파일입니다. 호환성 변환을 시도합니다.")
                checkpoint = ModelSaver._convert_old_checkpoint(checkpoint)
            
            # 모델 상태 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 옵티마이저 상태 로드 (있는 경우)
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"모델 로드 완료: {filepath}")
            
            return {
                'config': checkpoint['config'],
                'metadata': checkpoint.get('metadata', {}),
                'version': checkpoint['version'],
                'created_at': checkpoint['created_at']
            }
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def _convert_old_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        이전 버전 체크포인트 변환
        
        Args:
            checkpoint: 이전 버전 체크포인트
            
        Returns:
            변환된 체크포인트
        """
        # 버전별 변환 로직
        if 'version' not in checkpoint:
            # 가장 오래된 형식
            converted = {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'model_state_dict': checkpoint.get('state_dict') or checkpoint.get('model_state_dict'),
                'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None),
                'config': checkpoint.get('config', {}),
                'metadata': {}
            }
        else:
            # 이미 버전 정보가 있는 경우
            converted = checkpoint.copy()
            converted['version'] = '1.0.0'
            if 'created_at' not in converted:
                converted['created_at'] = datetime.now().isoformat()
            if 'metadata' not in converted:
                converted['metadata'] = {}
        
        return converted
    
    @staticmethod
    def get_checkpoint_info(filepath: str) -> Dict[str, Any]:
        """
        체크포인트 정보 조회
        
        Args:
            filepath: 체크포인트 파일 경로
            
        Returns:
            체크포인트 정보
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            return {
                'version': checkpoint.get('version', 'unknown'),
                'created_at': checkpoint.get('created_at', 'unknown'),
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
                'config': checkpoint.get('config', {}),
                'metadata': checkpoint.get('metadata', {})
            }
            
        except Exception as e:
            logger.error(f"체크포인트 정보 조회 중 오류 발생: {e}")
            return {} 