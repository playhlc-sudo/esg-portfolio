"""
리스크 필터링 엔진 (Stage 3)
- VaR / CVaR (Historical & Parametric)
- 최대낙폭, Sharpe/Sortino/Calmar 비율
- Kelly Criterion + 변동성 기반 포지션 사이징
- 최종 투자 신호 결정
- 1.txt DEShawModule 참고
"""
import numpy as np
import pandas as pd
from scipy import stats


def calculate_risk_metrics(prices, confidence=0.95):
    """
    종합 리스크 메트릭스 계산

    Args:
        prices: pd.Series (Close)
        confidence: float (VaR 신뢰수준)

    Returns:
        dict: 모든 리스크 지표
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]

    ret = prices.pct_change().dropna()

    if len(ret) < 30:
        return _empty_risk_metrics()

    metrics = {}

    # VaR (Historical)
    metrics["var_historical"] = round(
        np.percentile(ret, (1 - confidence) * 100) * 100, 3
    )

    # VaR (Parametric - 정규분포 가정)
    metrics["var_parametric"] = round(
        (ret.mean() - stats.norm.ppf(confidence) * ret.std()) * 100, 3
    )

    # CVaR (Expected Shortfall)
    var_threshold = np.percentile(ret, (1 - confidence) * 100)
    tail_losses = ret[ret <= var_threshold]
    metrics["cvar"] = round(tail_losses.mean() * 100, 3) if len(tail_losses) > 0 else 0

    # 최대 낙폭 (Maximum Drawdown)
    cumulative = (1 + ret).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics["max_drawdown"] = round(drawdown.min() * 100, 2)
    metrics["current_drawdown"] = round(drawdown.iloc[-1] * 100, 2)

    # 연간 수익률
    annual_return = ret.mean() * 252
    metrics["annual_return"] = round(annual_return * 100, 2)

    # 연간 변동성
    annual_vol = ret.std() * np.sqrt(252)
    metrics["annual_volatility"] = round(annual_vol * 100, 2)

    # Sharpe Ratio (무위험이자율 2% 가정)
    risk_free = 0.02
    sharpe = (annual_return - risk_free) / (annual_vol + 1e-10)
    metrics["sharpe_ratio"] = round(sharpe, 3)

    # Sortino Ratio (하방 변동성만 사용)
    downside_ret = ret[ret < 0]
    downside_std = downside_ret.std() * np.sqrt(252) if len(downside_ret) > 0 else annual_vol
    sortino = (annual_return - risk_free) / (downside_std + 1e-10)
    metrics["sortino_ratio"] = round(sortino, 3)

    # Calmar Ratio
    calmar = annual_return / (abs(drawdown.min()) + 1e-10)
    metrics["calmar_ratio"] = round(calmar, 3)

    # 꼬리 리스크
    metrics["skewness"] = round(ret.skew(), 3)
    metrics["kurtosis"] = round(ret.kurtosis(), 3)

    # 승률 (양수 수익률 비율)
    metrics["win_rate"] = round((ret > 0).mean() * 100, 1)

    return metrics


def calculate_position_size(risk_metrics, max_risk_pct=2, vol_target=15):
    """
    Kelly Criterion + 리스크 기반 포지션 사이징

    Args:
        risk_metrics: dict (calculate_risk_metrics 결과)
        max_risk_pct: float (최대 손실 허용 %)
        vol_target: float (목표 변동성 %)

    Returns:
        dict: position_pct, risk_based, vol_based
    """
    if not risk_metrics:
        return {"position_pct": 10.0, "risk_based": 10.0, "vol_based": 10.0}

    # VaR 기반 포지션 사이징
    var = abs(risk_metrics.get("var_historical", -2))
    risk_based_pct = min(max_risk_pct / (var + 0.01) * 100, 100) if var > 0 else 10

    # 변동성 기반 조정
    vol = risk_metrics.get("annual_volatility", 20)
    vol_based_pct = min(vol_target / (vol + 1e-10) * 100, 100)

    # 최종 포지션 (보수적: 둘 중 작은 값)
    position_pct = min(risk_based_pct, vol_based_pct, 100)

    return {
        "position_pct": round(position_pct, 1),
        "risk_based": round(risk_based_pct, 1),
        "vol_based": round(vol_based_pct, 1),
    }


def compute_risk_score(risk_metrics):
    """
    리스크 메트릭스 → 0~100 점수로 변환

    높은 점수 = 리스크 관리 관점에서 투자하기 좋은 상태
    """
    if not risk_metrics:
        return 50.0

    score = 50.0

    # Sharpe Ratio 반영
    sharpe = risk_metrics.get("sharpe_ratio", 0)
    if sharpe > 1.5:
        score += 20
    elif sharpe > 1.0:
        score += 15
    elif sharpe > 0.5:
        score += 10
    elif sharpe > 0:
        score += 5
    elif sharpe < -0.5:
        score -= 20
    elif sharpe < 0:
        score -= 10

    # 현재 낙폭 반영
    dd = abs(risk_metrics.get("current_drawdown", 0))
    if dd > 30:
        score -= 25
    elif dd > 20:
        score -= 15
    elif dd > 10:
        score -= 10
    elif dd < 3:
        score += 5  # 낙폭이 작으면 보너스

    # 변동성 반영
    vol = risk_metrics.get("annual_volatility", 20)
    if vol > 60:
        score -= 15
    elif vol > 40:
        score -= 10
    elif vol < 15:
        score += 5

    # 승률 반영
    win_rate = risk_metrics.get("win_rate", 50)
    if win_rate > 55:
        score += 5
    elif win_rate < 45:
        score -= 5

    # 왜도 반영 (양의 왜도 = 큰 상승 가능성)
    skew = risk_metrics.get("skewness", 0)
    if skew > 0.5:
        score += 5
    elif skew < -0.5:
        score -= 5

    return round(max(0, min(100, score)), 1)


def make_final_decision(regime_score, ml_score, risk_score,
                        weights=None, risk_metrics=None):
    """
    3단계 파이프라인 최종 투자 결정

    Args:
        regime_score: float (0~100, Stage 1)
        ml_score: float (0~100, Stage 2)
        risk_score: float (0~100, Stage 3)
        weights: dict (optional, 각 레이어 가중치)
        risk_metrics: dict (optional, 절대적 리스크 차단용)

    Returns:
        dict: final_score, signal, action, confidence
    """
    if weights is None:
        weights = {"regime": 0.25, "ml": 0.45, "risk": 0.30}

    # 가중 평균 최종 점수
    final_score = (
        regime_score * weights["regime"]
        + ml_score * weights["ml"]
        + risk_score * weights["risk"]
    )
    final_score = round(max(0, min(100, final_score)), 1)

    # 리스크 절대 차단: 극단적 리스크 상황에서는 강제 하향
    if risk_metrics:
        dd = abs(risk_metrics.get("current_drawdown", 0))
        var = abs(risk_metrics.get("var_historical", 0))

        # 낙폭 30% 이상이면 최종 점수 강제 하향
        if dd > 30:
            final_score = min(final_score, 35)
        # 일일 VaR이 5% 이상이면 경고
        elif var > 5:
            final_score = min(final_score, 45)

    # 투자 신호 결정
    if final_score >= 70:
        signal = "STRONG_BUY"
        action = "매수 유리"
    elif final_score >= 58:
        signal = "BUY"
        action = "매수 고려"
    elif final_score <= 30:
        signal = "STRONG_SELL"
        action = "매도 유리"
    elif final_score <= 42:
        signal = "SELL"
        action = "매도 고려"
    else:
        signal = "HOLD"
        action = "관망"

    # 방향 결정
    if final_score > 55:
        direction = "상승"
    elif final_score < 45:
        direction = "하락"
    else:
        direction = "보합"

    # 신뢰도: 중심(50)에서 떨어질수록 높은 신뢰
    confidence = min(100, int(abs(final_score - 50) * 2))

    return {
        "final_score": final_score,
        "signal": signal,
        "action": action,
        "direction": direction,
        "confidence": confidence,
        "layer_scores": {
            "regime": round(regime_score, 1),
            "ml": round(ml_score, 1),
            "risk": round(risk_score, 1),
        },
        "weights": weights,
    }


def analyze_risk(prices, regime_score=50, ml_score=50, ml_result=None):
    """
    Stage 3 통합 분석: 리스크 계산 + 최종 결정

    Args:
        prices: pd.Series or DataFrame
        regime_score: float (Stage 1 점수)
        ml_score: float (Stage 2 점수)
        ml_result: dict (Stage 2 전체 결과, optional)

    Returns:
        dict: 리스크 메트릭스 + 포지션 사이징 + 최종 투자 결정
    """
    risk_metrics = calculate_risk_metrics(prices)
    risk_score = compute_risk_score(risk_metrics)
    position = calculate_position_size(risk_metrics)

    decision = make_final_decision(
        regime_score, ml_score, risk_score,
        risk_metrics=risk_metrics
    )

    return {
        "risk_metrics": risk_metrics,
        "risk_score": risk_score,
        "position": position,
        "decision": decision,
    }


def _empty_risk_metrics():
    """데이터 부족 시 기본 리스크 메트릭스"""
    return {
        "var_historical": 0, "var_parametric": 0, "cvar": 0,
        "max_drawdown": 0, "current_drawdown": 0,
        "annual_return": 0, "annual_volatility": 0,
        "sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0,
        "skewness": 0, "kurtosis": 0, "win_rate": 50,
    }
