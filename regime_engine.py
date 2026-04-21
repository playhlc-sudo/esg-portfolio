"""
시장 레짐 판단 엔진 (Stage 1)
- HMM 기반 시장 국면 탐지 (상승/횡보/하락)
- Hurst 지수로 추세추종 vs 평균회귀 판단
- 1.txt RenaissanceModule 참고
"""
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def _hurst_exponent(ts, max_lag=20):
    """Hurst 지수 근사 계산"""
    if len(ts) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if min(tau) <= 0:
        return 0.5
    reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return float(reg[0])


def detect_regime(prices, n_states=3):
    """
    HMM 기반 시장 국면 탐지

    Returns:
        dict with keys:
            regime: int (0=약세, 1=중립, 2=강세)
            regime_name: str
            regime_history: pd.Series
            confidence: float (0~1)
    """
    returns = prices.pct_change().dropna()

    if HMM_AVAILABLE and len(returns) > 100:
        try:
            ret_clean = returns.values.reshape(-1, 1)
            model = GaussianHMM(
                n_components=n_states, covariance_type="full",
                n_iter=200, random_state=42
            )
            model.fit(ret_clean)
            states = model.predict(ret_clean)

            # 각 상태의 평균 수익률로 정렬 (0=약세, 1=중립, 2=강세)
            means = [ret_clean[states == i].mean() for i in range(n_states)]
            sorted_idx = np.argsort(means)
            state_map = {sorted_idx[i]: i for i in range(n_states)}
            states = np.array([state_map[s] for s in states])

            regime_series = pd.Series(states, index=returns.index)
            current_regime = int(regime_series.iloc[-1])

            # 최근 10일 중 현재 레짐과 같은 비율 = 신뢰도
            recent = regime_series.iloc[-10:]
            confidence = float((recent == current_regime).mean())

            return {
                "regime": current_regime,
                "regime_name": {0: "약세", 1: "중립", 2: "강세"}.get(current_regime, "중립"),
                "regime_history": regime_series,
                "confidence": confidence,
                "method": "HMM",
            }
        except Exception:
            pass

    # Fallback: 변동성 + 이동평균 기반 간이 판단
    vol = returns.rolling(20).std()
    ret_ma = returns.rolling(20).mean()

    states = pd.Series(1, index=returns.index)
    states[ret_ma > ret_ma.quantile(0.6)] = 2
    states[ret_ma < ret_ma.quantile(0.4)] = 0
    states[vol > vol.quantile(0.8)] = 0

    states = states.dropna()
    current_regime = int(states.iloc[-1]) if len(states) > 0 else 1
    recent = states.iloc[-10:]
    confidence = float((recent == current_regime).mean()) if len(recent) > 0 else 0.5

    return {
        "regime": current_regime,
        "regime_name": {0: "약세", 1: "중립", 2: "강세"}.get(current_regime, "중립"),
        "regime_history": states,
        "confidence": confidence,
        "method": "Fallback",
    }


def compute_hurst(prices, window=60):
    """
    Hurst 지수 계산

    Returns:
        dict with keys:
            hurst: float
            interpretation: str
            strategy_hint: str
    """
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        return {"hurst": 0.5, "interpretation": "데이터 부족", "strategy_hint": "판단 불가"}

    hurst_series = returns.rolling(window).apply(
        lambda x: _hurst_exponent(x.values), raw=False
    )
    latest_hurst = hurst_series.iloc[-1]

    if pd.isna(latest_hurst):
        latest_hurst = 0.5

    if latest_hurst < 0.45:
        interp = "평균회귀 성향"
        hint = "역추세 전략 유리 (과매도 매수, 과매수 매도)"
    elif latest_hurst > 0.55:
        interp = "추세추종 성향"
        hint = "모멘텀 전략 유리 (추세 방향 따라가기)"
    else:
        interp = "랜덤워크 (추세 없음)"
        hint = "방향성 베팅 위험, 관망 또는 변동성 매매"

    return {
        "hurst": round(float(latest_hurst), 4),
        "hurst_history": hurst_series,
        "interpretation": interp,
        "strategy_hint": hint,
    }


def analyze_regime(prices):
    """
    Stage 1 통합 분석: 레짐 + Hurst

    Returns:
        dict: regime_result + hurst_result + regime_score (0~100)
    """
    regime_result = detect_regime(prices)
    hurst_result = compute_hurst(prices)

    # 레짐 점수 산출 (0~100)
    regime = regime_result["regime"]
    base_score = {0: 25, 1: 50, 2: 75}.get(regime, 50)

    # Hurst로 점수 조정: 강세+추세추종이면 보너스, 약세+추세추종이면 페널티
    hurst = hurst_result["hurst"]
    if regime == 2 and hurst > 0.55:
        # 강세장 + 추세 지속 → 보너스
        base_score += 10
    elif regime == 0 and hurst > 0.55:
        # 약세장 + 추세 지속 → 추가 페널티
        base_score -= 10
    elif regime == 0 and hurst < 0.45:
        # 약세장이지만 평균회귀 성향 → 반등 가능성
        base_score += 5

    regime_score = max(0, min(100, base_score))

    # 신뢰도로 가중
    confidence = regime_result["confidence"]
    regime_score = regime_score * confidence + 50 * (1 - confidence)

    return {
        **regime_result,
        **hurst_result,
        "regime_score": round(regime_score, 1),
    }
