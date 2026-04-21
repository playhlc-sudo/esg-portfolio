"""
환경 설정 및 Firebase 구성 관리
"""
import os
import json
import streamlit as st
from dotenv import load_dotenv

# .env 파일 로드 (로컬 개발용)
load_dotenv()


class Config:
    """앱 설정 관리 클래스"""

    # Firebase 설정
    FIREBASE_CREDENTIALS_FILE = "firebase_credentials.json"

    # 앱 설정
    APP_NAME = "ESG 포트폴리오 대시보드"
    COOKIE_NAME = "esg_portfolio_auth"
    COOKIE_KEY = os.getenv("COOKIE_KEY", "esg_portfolio_secret_key_2024")
    COOKIE_EXPIRY_DAYS = 30

    # 설정 파일
    LOCAL_SETTINGS_FILE = "user_settings.json"

    @classmethod
    def get_firebase_credentials(cls) -> dict | None:
        """Firebase 인증 정보 가져오기

        우선순위:
        1. Streamlit Cloud Secrets (st.secrets)
        2. 환경변수 (FIREBASE_CREDENTIALS JSON 문자열)
        3. 로컬 파일 (firebase_credentials.json)
        """
        # 1. Streamlit Cloud Secrets
        try:
            if hasattr(st, 'secrets') and 'firebase' in st.secrets:
                return dict(st.secrets['firebase'])
        except Exception:
            pass

        # 2. 환경변수 (JSON 문자열)
        firebase_json = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_json:
            try:
                return json.loads(firebase_json)
            except json.JSONDecodeError:
                pass

        # 3. 로컬 파일
        cred_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            cls.FIREBASE_CREDENTIALS_FILE
        )
        if os.path.exists(cred_path):
            try:
                with open(cred_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    @classmethod
    def is_firebase_available(cls) -> bool:
        """Firebase 사용 가능 여부 확인"""
        return cls.get_firebase_credentials() is not None

    @classmethod
    def get_auth_config(cls) -> dict:
        """streamlit-authenticator 설정 반환"""
        return {
            "cookie": {
                "name": cls.COOKIE_NAME,
                "key": cls.COOKIE_KEY,
                "expiry_days": cls.COOKIE_EXPIRY_DAYS,
            },
            "preauthorized": {
                "emails": []  # 필요시 사전 승인 이메일 추가
            }
        }


# 전역 설정 인스턴스
config = Config()
