"""
설정 관리자 - 로컬/Firebase 추상화 계층
"""
import os
import json
from typing import Any
from firebase_db import get_firebase_db
from config import config


class SettingsManager:
    """사용자별 설정 관리 클래스"""

    DEFAULT_SETTINGS = {
        "tickers_input": "BBAI, VRT, TSLL, 132030.KS, 476800.KS, 144600.KS",
        "start_date": "2020-01-01",
        "end_date": "2025-01-01",
        "initial_capital": 100000,
        "esg_scores": {},
        "rebalance_freq": "M",
        "tx_cost": 0.1,
        "layer_weights": {"regime": 20, "ml": 40, "risk": 25, "cross_asset": 15},
        "holdings": {},
    }

    def __init__(self, username: str):
        """
        Args:
            username: 사용자 ID (로컬 모드인 경우 'local_user')
        """
        self.username = username
        self.db = get_firebase_db()
        self.is_local_mode = (username == "local_user") or not self.db.is_connected()

        # 로컬 설정 파일 경로
        self.local_settings_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config.LOCAL_SETTINGS_FILE
        )

    def load_settings(self) -> dict:
        """설정 로드

        Returns:
            dict: 사용자 설정 (기본값과 병합됨)
        """
        settings = self.DEFAULT_SETTINGS.copy()

        if self.is_local_mode:
            saved = self._load_local_settings()
        else:
            saved = self._load_firebase_settings()

        if saved:
            settings.update(saved)

        return settings

    def save_settings(self, data: dict) -> bool:
        """설정 저장

        Args:
            data: 저장할 설정 딕셔너리

        Returns:
            bool: 성공 여부
        """
        if self.is_local_mode:
            return self._save_local_settings(data)
        else:
            return self._save_firebase_settings(data)

    def _load_local_settings(self) -> dict | None:
        """로컬 파일에서 설정 로드"""
        if os.path.exists(self.local_settings_file):
            try:
                with open(self.local_settings_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"로컬 설정 로드 오류: {e}")
        return None

    def _save_local_settings(self, data: dict) -> bool:
        """로컬 파일에 설정 저장"""
        try:
            with open(self.local_settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"로컬 설정 저장 오류: {e}")
            return False

    def _load_firebase_settings(self) -> dict | None:
        """Firebase에서 설정 로드"""
        return self.db.get_user_settings(self.username)

    def _save_firebase_settings(self, data: dict) -> bool:
        """Firebase에 설정 저장"""
        return self.db.save_user_settings(self.username, data)

    def reset_to_defaults(self) -> bool:
        """설정을 기본값으로 리셋"""
        return self.save_settings(self.DEFAULT_SETTINGS.copy())

    def get_storage_info(self) -> str:
        """현재 저장소 정보 반환"""
        if self.is_local_mode:
            return "로컬 저장소"
        return "Firebase Cloud"


def get_settings_manager(username: str) -> SettingsManager:
    """설정 관리자 인스턴스 생성

    Args:
        username: 사용자 ID

    Returns:
        SettingsManager 인스턴스
    """
    return SettingsManager(username)
