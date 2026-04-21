"""
주식 종목 유틸리티 - 종목명 ↔ 종목코드 변환
"""
import streamlit as st
from functools import lru_cache


@st.cache_data(ttl=86400)  # 24시간 캐시
def get_krx_stock_list() -> dict:
    """KRX 전체 종목 리스트 가져오기 (종목명 → 코드)

    Returns:
        dict: {"삼성전자": "005930", "SK하이닉스": "000660", ...}
    """
    try:
        from pykrx import stock
        from datetime import datetime, timedelta

        # 최근 거래일 기준
        today = datetime.now()
        for i in range(7):
            date = (today - timedelta(days=i)).strftime("%Y%m%d")
            try:
                # KOSPI + KOSDAQ 종목
                kospi = stock.get_market_ticker_list(date, market="KOSPI")
                kosdaq = stock.get_market_ticker_list(date, market="KOSDAQ")

                if kospi or kosdaq:
                    name_to_code = {}
                    for ticker in kospi + kosdaq:
                        name = stock.get_market_ticker_name(ticker)
                        if name:
                            name_to_code[name] = ticker
                    return name_to_code
            except Exception:
                continue
        return {}
    except ImportError:
        return {}
    except Exception as e:
        print(f"KRX 종목 리스트 로드 오류: {e}")
        return {}


@st.cache_data(ttl=86400)
def get_code_to_name_map() -> dict:
    """종목코드 → 종목명 매핑

    Returns:
        dict: {"005930": "삼성전자", "000660": "SK하이닉스", ...}
    """
    name_to_code = get_krx_stock_list()
    return {code: name for name, code in name_to_code.items()}


def resolve_ticker(input_text: str) -> tuple[str, str]:
    """입력값을 종목코드로 변환

    Args:
        input_text: 사용자 입력 (종목명 또는 종목코드)

    Returns:
        tuple: (종목코드, 종목명) - 예: ("005930.KS", "삼성전자")
               해외주식인 경우: ("AAPL", "AAPL")
    """
    text = input_text.strip()

    # 이미 .KS 또는 .KQ가 붙어있는 경우
    if text.endswith(".KS") or text.endswith(".KQ"):
        code = text.replace(".KS", "").replace(".KQ", "")
        code_to_name = get_code_to_name_map()
        name = code_to_name.get(code, text)
        return (text, name)

    # 숫자로만 구성된 경우 (한국 종목코드)
    if text.isdigit() and len(text) == 6:
        code_to_name = get_code_to_name_map()
        name = code_to_name.get(text, text)
        # KOSPI(.KS) 또는 KOSDAQ(.KQ) 판별
        suffix = ".KS"  # 기본값 KOSPI
        name_to_code = get_krx_stock_list()
        # 실제로는 pykrx에서 시장 구분 가능하지만 단순화를 위해 .KS 사용
        return (f"{text}{suffix}", name)

    # 종목명으로 검색
    name_to_code = get_krx_stock_list()
    if text in name_to_code:
        code = name_to_code[text]
        return (f"{code}.KS", text)

    # 부분 매칭 시도
    for name, code in name_to_code.items():
        if text in name or name in text:
            return (f"{code}.KS", name)

    # 해외 주식으로 간주 (영문 알파벳)
    return (text.upper(), text.upper())


def parse_tickers_input(input_text: str) -> list[dict]:
    """쉼표로 구분된 종목 입력을 파싱

    Args:
        input_text: "삼성전자, AAPL, SK하이닉스, 005930.KS"

    Returns:
        list: [{"code": "005930.KS", "name": "삼성전자"}, ...]
    """
    results = []
    seen_codes = set()

    for item in input_text.split(","):
        item = item.strip()
        if not item:
            continue

        code, name = resolve_ticker(item)

        # 중복 제거
        if code not in seen_codes:
            seen_codes.add(code)
            results.append({"code": code, "name": name})

    return results


def format_ticker_display(code: str) -> str:
    """종목코드를 표시용 문자열로 변환

    Args:
        code: "005930.KS" 또는 "AAPL"

    Returns:
        str: "삼성전자(005930)" 또는 "AAPL"
    """
    if code.endswith(".KS") or code.endswith(".KQ"):
        base_code = code.replace(".KS", "").replace(".KQ", "")
        code_to_name = get_code_to_name_map()
        name = code_to_name.get(base_code, "")
        if name:
            return f"{name}({base_code})"
        return code
    return code


def get_ticker_name(code: str) -> str:
    """종목코드로 종목명 조회

    Args:
        code: "005930.KS" 또는 "AAPL"

    Returns:
        str: "삼성전자" 또는 "AAPL"
    """
    if code.endswith(".KS") or code.endswith(".KQ"):
        base_code = code.replace(".KS", "").replace(".KQ", "")
        code_to_name = get_code_to_name_map()
        return code_to_name.get(base_code, code)
    return code
