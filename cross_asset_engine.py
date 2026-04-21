"""
크로스 에셋 시그널 엔진 (시타델 스타일)
- VIX 기반 공포/안도 시그널
- 달러 강세/약세 시그널
- 채권/주식 비율 Risk-On/Off
- 팩터 노출도 분석
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import logging
import warnings


@st.cache_data(ttl=600, show_spinner=False)
def fetch_cross_asset_data(period_days=120):
    """
    크로스 에셋 데이터 수집 (캐싱 적용)

    Returns:
        dict: {ticker: pd.DataFrame}
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days + 50)

    # 크로스 에셋 티커
    cross_tickers = ['SPY', 'TLT', 'GLD', 'UUP', '^VIX']

    # 로그 억제
    loggers = ['yfinance', 'urllib3', 'requests', 'peewee']
    original_levels = {}
    for name in loggers:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(logging.CRITICAL)

    data = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ticker in cross_tickers:
                try:
                    df = yf.download(
                        ticker,
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False
                    )
                    if not df.empty and len(df) > 20:
                        data[ticker] = df
                except Exception:
                    pass
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)

    return data


def analyze_vix_signal(cross_data):
    """
    VIX 기반 공포/안도 시그널

    Returns:
        dict: signal, score, interpretation
    """
    if '^VIX' not in cross_data or cross_data['^VIX'].empty:
        return {"signal": "NEUTRAL", "score": 50, "interpretation": "VIX 데이터 없음", "value": 0}

    vix = cross_data['^VIX']['Close']
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]

    current_vix = vix.iloc[-1]
    vix_ma20 = vix.rolling(20).mean().iloc[-1]
    vix_ma60 = vix.rolling(60).mean().iloc[-1] if len(vix) >= 60 else vix_ma20

    # VIX 비율
    vix_ratio = current_vix / vix_ma20 if vix_ma20 > 0 else 1

    # 시그널 판단
    if current_vix > 30 or vix_ratio > 1.3:
        signal = "HIGH_FEAR"
        score = 25  # 공포 = 위험자산 매수에 불리
        interpretation = f"극단적 공포 (VIX: {current_vix:.1f})"
    elif current_vix > 20 or vix_ratio > 1.1:
        signal = "ELEVATED"
        score = 40
        interpretation = f"경계 수준 (VIX: {current_vix:.1f})"
    elif current_vix < 15 and vix_ratio < 0.9:
        signal = "COMPLACENT"
        score = 55  # 지나친 안도 = 역추세 주의
        interpretation = f"과도한 안도 (VIX: {current_vix:.1f})"
    else:
        signal = "NEUTRAL"
        score = 50
        interpretation = f"정상 범위 (VIX: {current_vix:.1f})"

    return {
        "signal": signal,
        "score": round(score, 1),
        "interpretation": interpretation,
        "value": round(current_vix, 2),
        "ratio": round(vix_ratio, 3)
    }


def analyze_dollar_signal(cross_data):
    """
    달러 강세/약세 시그널 (UUP 기반)

    Returns:
        dict: signal, score, interpretation
    """
    if 'UUP' not in cross_data or cross_data['UUP'].empty:
        return {"signal": "NEUTRAL", "score": 50, "interpretation": "달러 데이터 없음", "change_20d": 0}

    uup = cross_data['UUP']['Close']
    if isinstance(uup, pd.DataFrame):
        uup = uup.iloc[:, 0]

    # 20일 변화율
    change_20d = (uup.iloc[-1] / uup.iloc[-20] - 1) if len(uup) >= 20 else 0
    change_5d = (uup.iloc[-1] / uup.iloc[-5] - 1) if len(uup) >= 5 else 0

    if change_20d > 0.03:
        signal = "STRONG_DOLLAR"
        score = 35  # 달러 강세 = 위험자산(주식) 약세 경향
        interpretation = f"달러 강세 (+{change_20d:.1%})"
    elif change_20d > 0.01:
        signal = "MILD_STRONG"
        score = 45
        interpretation = f"달러 소폭 강세 (+{change_20d:.1%})"
    elif change_20d < -0.03:
        signal = "WEAK_DOLLAR"
        score = 65  # 달러 약세 = 위험자산 강세 경향
        interpretation = f"달러 약세 ({change_20d:.1%})"
    elif change_20d < -0.01:
        signal = "MILD_WEAK"
        score = 55
        interpretation = f"달러 소폭 약세 ({change_20d:.1%})"
    else:
        signal = "NEUTRAL"
        score = 50
        interpretation = f"달러 중립 ({change_20d:+.1%})"

    return {
        "signal": signal,
        "score": round(score, 1),
        "interpretation": interpretation,
        "change_20d": round(change_20d * 100, 2),
        "change_5d": round(change_5d * 100, 2)
    }


def analyze_bond_equity_signal(cross_data):
    """
    채권/주식 비율 Risk-On/Off 시그널

    Returns:
        dict: signal, score, interpretation
    """
    if 'TLT' not in cross_data or 'SPY' not in cross_data:
        return {"signal": "NEUTRAL", "score": 50, "interpretation": "데이터 없음", "risk_mode": "UNKNOWN"}

    tlt = cross_data['TLT']['Close']
    spy = cross_data['SPY']['Close']

    if isinstance(tlt, pd.DataFrame):
        tlt = tlt.iloc[:, 0]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]

    # 공통 인덱스
    common = tlt.index.intersection(spy.index)
    if len(common) < 20:
        return {"signal": "NEUTRAL", "score": 50, "interpretation": "데이터 부족", "risk_mode": "UNKNOWN"}

    tlt_common = tlt.loc[common]
    spy_common = spy.loc[common]

    # TLT/SPY 비율
    ratio = tlt_common / spy_common
    ratio_current = ratio.iloc[-1]
    ratio_ma20 = ratio.rolling(20).mean().iloc[-1]
    ratio_change = (ratio_current / ratio_ma20 - 1) if ratio_ma20 > 0 else 0

    # 20일 수익률 비교
    tlt_ret = (tlt_common.iloc[-1] / tlt_common.iloc[-20] - 1)
    spy_ret = (spy_common.iloc[-1] / spy_common.iloc[-20] - 1)

    if ratio_change > 0.03 or (tlt_ret > spy_ret + 0.03):
        signal = "RISK_OFF"
        score = 35  # 채권 선호 = 주식 약세 환경
        risk_mode = "Risk-Off"
        interpretation = f"안전자산 선호 (TLT/SPY +{ratio_change:.1%})"
    elif ratio_change < -0.03 or (spy_ret > tlt_ret + 0.03):
        signal = "RISK_ON"
        score = 65  # 주식 선호 = 강세 환경
        risk_mode = "Risk-On"
        interpretation = f"위험자산 선호 (TLT/SPY {ratio_change:.1%})"
    else:
        signal = "NEUTRAL"
        score = 50
        risk_mode = "중립"
        interpretation = f"균형 (TLT/SPY {ratio_change:+.1%})"

    return {
        "signal": signal,
        "score": round(score, 1),
        "interpretation": interpretation,
        "risk_mode": risk_mode,
        "tlt_return_20d": round(tlt_ret * 100, 2),
        "spy_return_20d": round(spy_ret * 100, 2),
        "ratio_change": round(ratio_change * 100, 2)
    }


def analyze_gold_signal(cross_data):
    """
    금 가격 시그널 (안전자산 수요)

    Returns:
        dict: signal, score, interpretation
    """
    if 'GLD' not in cross_data or cross_data['GLD'].empty:
        return {"signal": "NEUTRAL", "score": 50, "interpretation": "금 데이터 없음"}

    gld = cross_data['GLD']['Close']
    if isinstance(gld, pd.DataFrame):
        gld = gld.iloc[:, 0]

    change_20d = (gld.iloc[-1] / gld.iloc[-20] - 1) if len(gld) >= 20 else 0
    ma50 = gld.rolling(50).mean().iloc[-1] if len(gld) >= 50 else gld.mean()
    above_ma50 = gld.iloc[-1] > ma50

    if change_20d > 0.05 and above_ma50:
        signal = "STRONG_DEMAND"
        score = 40  # 금 강세 = 불확실성 증가
        interpretation = f"금 강세 (+{change_20d:.1%}) - 불확실성 증가"
    elif change_20d > 0.02:
        signal = "MILD_DEMAND"
        score = 45
        interpretation = f"금 소폭 상승 (+{change_20d:.1%})"
    elif change_20d < -0.03:
        signal = "WEAK_DEMAND"
        score = 55  # 금 약세 = 위험선호
        interpretation = f"금 약세 ({change_20d:.1%}) - 위험선호"
    else:
        signal = "NEUTRAL"
        score = 50
        interpretation = f"금 중립 ({change_20d:+.1%})"

    return {
        "signal": signal,
        "score": round(score, 1),
        "interpretation": interpretation,
        "change_20d": round(change_20d * 100, 2)
    }


def calculate_cross_asset_score(cross_data):
    """
    크로스 에셋 종합 점수 계산

    Returns:
        dict: 모든 시그널 + 종합 점수
    """
    vix = analyze_vix_signal(cross_data)
    dollar = analyze_dollar_signal(cross_data)
    bond_equity = analyze_bond_equity_signal(cross_data)
    gold = analyze_gold_signal(cross_data)

    # 가중 평균 (VIX가 가장 중요)
    weights = {
        'vix': 0.35,
        'dollar': 0.20,
        'bond_equity': 0.30,
        'gold': 0.15
    }

    total_score = (
        vix['score'] * weights['vix'] +
        dollar['score'] * weights['dollar'] +
        bond_equity['score'] * weights['bond_equity'] +
        gold['score'] * weights['gold']
    )

    # 종합 시그널
    if total_score >= 60:
        overall_signal = "FAVORABLE"
        overall_interpretation = "크로스 에셋 환경 우호적 (Risk-On)"
    elif total_score <= 40:
        overall_signal = "UNFAVORABLE"
        overall_interpretation = "크로스 에셋 환경 비우호적 (Risk-Off)"
    else:
        overall_signal = "NEUTRAL"
        overall_interpretation = "크로스 에셋 환경 중립"

    return {
        "vix": vix,
        "dollar": dollar,
        "bond_equity": bond_equity,
        "gold": gold,
        "cross_asset_score": round(total_score, 1),
        "overall_signal": overall_signal,
        "overall_interpretation": overall_interpretation,
        "weights": weights
    }


def find_pair_opportunities(tickers_data, min_corr=0.6):
    """
    페어 트레이딩 기회 탐색

    Args:
        tickers_data: dict {ticker: pd.Series(prices)}
        min_corr: float (최소 상관계수)

    Returns:
        list: 페어 기회 목록
    """
    tickers = list(tickers_data.keys())
    pairs = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            p1 = tickers_data[t1]
            p2 = tickers_data[t2]

            # 공통 인덱스
            common = p1.index.intersection(p2.index)
            if len(common) < 60:
                continue

            p1_common = p1.loc[common]
            p2_common = p2.loc[common]

            # 상관계수
            corr = p1_common.corr(p2_common)
            if abs(corr) < min_corr:
                continue

            # 스프레드 Z-score
            ratio = p1_common / p2_common
            z_score = (ratio.iloc[-1] - ratio.mean()) / (ratio.std() + 1e-10)

            # 시그널 판단
            if z_score < -2:
                signal = "LONG_1_SHORT_2"
                interpretation = f"{t1} 매수 / {t2} 매도"
            elif z_score > 2:
                signal = "SHORT_1_LONG_2"
                interpretation = f"{t1} 매도 / {t2} 매수"
            else:
                signal = "NO_SIGNAL"
                interpretation = "시그널 없음"

            pairs.append({
                "pair": f"{t1} / {t2}",
                "ticker1": t1,
                "ticker2": t2,
                "correlation": round(corr, 3),
                "z_score": round(z_score, 3),
                "signal": signal,
                "interpretation": interpretation
            })

    # Z-score 절대값 기준 정렬
    pairs.sort(key=lambda x: abs(x['z_score']), reverse=True)
    return pairs[:10]  # 상위 10개
