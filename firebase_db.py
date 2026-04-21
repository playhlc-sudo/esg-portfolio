"""
Firebase Firestore 데이터베이스 연동
"""
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Any
from config import config


class FirebaseDB:
    """Firebase Firestore CRUD 클래스"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not FirebaseDB._initialized:
            self._init_firebase()
            FirebaseDB._initialized = True

    def _init_firebase(self):
        """Firebase 초기화"""
        self.db = None

        if firebase_admin._apps:
            # 이미 초기화된 경우
            self.db = firestore.client()
            return

        cred_dict = config.get_firebase_credentials()
        if cred_dict:
            try:
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                self.db = firestore.client()
            except Exception as e:
                print(f"Firebase 초기화 오류: {e}")
                self.db = None

    def is_connected(self) -> bool:
        """Firebase 연결 상태 확인"""
        return self.db is not None

    # ==================== 사용자 관리 ====================

    def get_user(self, username: str) -> dict | None:
        """사용자 정보 조회"""
        if not self.is_connected():
            return None

        try:
            doc = self.db.collection("users").document(username).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            print(f"사용자 조회 오류: {e}")
            return None

    def create_user(self, username: str, name: str, email: str, password_hash: str) -> bool:
        """새 사용자 생성"""
        if not self.is_connected():
            return False

        try:
            user_data = {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.now(),
                "last_login": None,
                "is_active": True,
            }
            self.db.collection("users").document(username).set(user_data)
            return True
        except Exception as e:
            print(f"사용자 생성 오류: {e}")
            return False

    def update_last_login(self, username: str) -> bool:
        """마지막 로그인 시간 업데이트"""
        if not self.is_connected():
            return False

        try:
            self.db.collection("users").document(username).update({
                "last_login": datetime.now()
            })
            return True
        except Exception as e:
            print(f"로그인 시간 업데이트 오류: {e}")
            return False

    def get_all_users(self) -> dict:
        """모든 사용자 정보 조회 (인증용)"""
        if not self.is_connected():
            return {}

        try:
            users = {}
            docs = self.db.collection("users").stream()
            for doc in docs:
                data = doc.to_dict()
                users[doc.id] = {
                    "name": data.get("name", doc.id),
                    "email": data.get("email", ""),
                    "password": data.get("password_hash", ""),
                }
            return users
        except Exception as e:
            print(f"사용자 목록 조회 오류: {e}")
            return {}

    def user_exists(self, username: str) -> bool:
        """사용자 존재 여부 확인"""
        return self.get_user(username) is not None

    def email_exists(self, email: str) -> bool:
        """이메일 중복 확인"""
        if not self.is_connected():
            return False

        try:
            docs = self.db.collection("users").where("email", "==", email).limit(1).stream()
            return any(True for _ in docs)
        except Exception as e:
            print(f"이메일 확인 오류: {e}")
            return False

    # ==================== 설정 관리 ====================

    def get_user_settings(self, username: str) -> dict | None:
        """사용자 설정 조회"""
        if not self.is_connected():
            return None

        try:
            doc = self.db.collection("user_settings").document(username).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get("settings", {})
            return None
        except Exception as e:
            print(f"설정 조회 오류: {e}")
            return None

    def save_user_settings(self, username: str, settings: dict) -> bool:
        """사용자 설정 저장"""
        if not self.is_connected():
            return False

        try:
            doc_data = {
                "settings": settings,
                "updated_at": datetime.now(),
            }
            self.db.collection("user_settings").document(username).set(doc_data)
            return True
        except Exception as e:
            print(f"설정 저장 오류: {e}")
            return False

    def delete_user_settings(self, username: str) -> bool:
        """사용자 설정 삭제"""
        if not self.is_connected():
            return False

        try:
            self.db.collection("user_settings").document(username).delete()
            return True
        except Exception as e:
            print(f"설정 삭제 오류: {e}")
            return False


# 싱글톤 인스턴스
def get_firebase_db() -> FirebaseDB:
    """Firebase DB 인스턴스 반환"""
    return FirebaseDB()
