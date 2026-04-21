"""
인증 모듈 - 로그인/회원가입 UI 및 인증 로직
"""
import streamlit as st
import streamlit_authenticator as stauth
import bcrypt
from config import config
from firebase_db import get_firebase_db


def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """비밀번호 검증"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def init_session_state():
    """세션 상태 초기화"""
    if "authentication_status" not in st.session_state:
        st.session_state.authentication_status = None
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None


def show_signup_form() -> bool:
    """회원가입 폼 표시"""
    db = get_firebase_db()

    st.subheader("회원가입")

    with st.form("signup_form"):
        new_username = st.text_input("사용자 ID", placeholder="영문, 숫자 조합")
        new_name = st.text_input("이름", placeholder="표시될 이름")
        new_email = st.text_input("이메일", placeholder="example@email.com")
        new_password = st.text_input("비밀번호", type="password", placeholder="8자 이상")
        new_password_confirm = st.text_input("비밀번호 확인", type="password")

        submitted = st.form_submit_button("가입하기", type="primary", use_container_width=True)

        if submitted:
            # 유효성 검사
            if not all([new_username, new_name, new_email, new_password]):
                st.error("모든 필드를 입력해주세요.")
                return False

            if len(new_username) < 3:
                st.error("사용자 ID는 3자 이상이어야 합니다.")
                return False

            if len(new_password) < 8:
                st.error("비밀번호는 8자 이상이어야 합니다.")
                return False

            if new_password != new_password_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
                return False

            if not db.is_connected():
                st.error("데이터베이스 연결 실패. 잠시 후 다시 시도해주세요.")
                return False

            # 중복 확인
            if db.user_exists(new_username):
                st.error("이미 사용 중인 ID입니다.")
                return False

            if db.email_exists(new_email):
                st.error("이미 등록된 이메일입니다.")
                return False

            # 사용자 생성
            password_hash = hash_password(new_password)
            if db.create_user(new_username, new_name, new_email, password_hash):
                st.success("회원가입이 완료되었습니다! 로그인해주세요.")
                return True
            else:
                st.error("회원가입 중 오류가 발생했습니다.")
                return False

    return False


def show_login_form() -> bool:
    """로그인 폼 표시"""
    db = get_firebase_db()

    st.subheader("로그인")

    with st.form("login_form"):
        username = st.text_input("사용자 ID")
        password = st.text_input("비밀번호", type="password")

        submitted = st.form_submit_button("로그인", type="primary", use_container_width=True)

        if submitted:
            if not username or not password:
                st.error("ID와 비밀번호를 입력해주세요.")
                return False

            if not db.is_connected():
                st.error("데이터베이스 연결 실패. 잠시 후 다시 시도해주세요.")
                return False

            user = db.get_user(username)
            if user is None:
                st.error("존재하지 않는 사용자입니다.")
                return False

            if not user.get("is_active", True):
                st.error("비활성화된 계정입니다.")
                return False

            if verify_password(password, user.get("password_hash", "")):
                # 로그인 성공
                st.session_state.authentication_status = True
                st.session_state.current_user = username
                st.session_state.user_name = user.get("name", username)

                # 마지막 로그인 시간 업데이트
                db.update_last_login(username)

                st.success(f"환영합니다, {user.get('name', username)}님!")
                st.rerun()
                return True
            else:
                st.error("비밀번호가 올바르지 않습니다.")
                return False

    return False


def logout():
    """로그아웃 처리"""
    st.session_state.authentication_status = None
    st.session_state.current_user = None
    st.session_state.user_name = None

    # 다른 세션 상태도 초기화 (선택적)
    keys_to_clear = [
        "backtest_result", "prediction_result", "explore_result",
        "auto_weights"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def check_authentication() -> bool:
    """인증 상태 확인 및 로그인/회원가입 UI 표시

    Returns:
        bool: 인증된 경우 True, 아니면 False
    """
    init_session_state()
    db = get_firebase_db()

    # 이미 인증된 경우
    if st.session_state.authentication_status:
        return True

    # Firebase 연결 확인
    if not db.is_connected():
        st.warning("Firebase가 설정되지 않았습니다. 로컬 모드로 실행됩니다.")
        st.session_state.authentication_status = True
        st.session_state.current_user = "local_user"
        st.session_state.user_name = "로컬 사용자"
        return True

    # 로그인/회원가입 UI
    st.title("ESG 포트폴리오 대시보드")
    st.markdown("---")

    tab_login, tab_signup = st.tabs(["로그인", "회원가입"])

    with tab_login:
        show_login_form()

    with tab_signup:
        show_signup_form()

    return False


def show_user_info_sidebar():
    """사이드바에 사용자 정보 및 로그아웃 버튼 표시"""
    if st.session_state.get("authentication_status"):
        user_name = st.session_state.get("user_name", "사용자")
        current_user = st.session_state.get("current_user", "")

        st.sidebar.markdown(f"**{user_name}** 님")
        if current_user != "local_user":
            st.sidebar.caption(f"@{current_user}")

        if st.sidebar.button("로그아웃", use_container_width=True):
            logout()
            st.rerun()

        st.sidebar.divider()
