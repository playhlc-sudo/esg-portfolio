"""
리밸런싱 엔진 (고도화)
- 4단계 파이프라인 결과를 종합하여 최적 비중 계산
- 액션 신호, 신뢰도, 리스크, 레짐 모두 반영
- Kelly Criterion 기반 포지션 사이징
"""
import numpy as np
import pandas as pd


def calculate_optimal_weights(predictions, tickers, esg_scores,
                               risk_tolerance="moderate",
                               min_weight=0.02, max_weight=0.40):
    """
    4단계 파이프라인 결과를 종합하여 최적 비중 계산

    Args:
        predictions: dict (predict_future 결과)
        tickers: list
        esg_scores: dict
        risk_tolerance: str ("conservative", "moderate", "aggressive")
        min_weight: float (최소 비중)
        max_weight: float (최대 비중)

    Returns:
        dict: {ticker: weight, ...} + 메타 정보
    """
    # 리스크 허용도별 파라미터
    risk_params = {
        "conservative": {"score_power": 1.5, "action_penalty": 0.8, "regime_factor": 0.7},
        "moderate": {"score_power": 2.0, "action_penalty": 0.6, "regime_factor": 0.85},
        "aggressive": {"score_power": 2.5, "action_penalty": 0.4, "regime_factor": 1.0},
    }
    params = risk_params.get(risk_tolerance, risk_params["moderate"])

    results = {}
    raw_scores = {}
    details = {}

    # 크로스 에셋 시장 환경 점수
    meta = predictions.get("__meta__", {})
    cross_asset = meta.get("cross_asset", {})
    market_env_score = cross_asset.get("cross_asset_score", 50)

    # 시장 환경 조정 계수 (약세장에서 전체 노출 축소)
    if market_env_score < 40:
        market_multiplier = 0.7  # Risk-Off 환경
    elif market_env_score > 60:
        market_multiplier = 1.1  # Risk-On 환경
    else:
        market_multiplier = 1.0

    for ticker in tickers:
        if ticker not in predictions or ticker == "__meta__":
            raw_scores[ticker] = 0
            details[ticker] = {"reason": "데이터 없음"}
            continue

        pred = predictions[ticker]
        layers = pred.get("layers", {})

        # 1. 기본 점수 (최종 점수 기반)
        final_score = float(pred.get("signals", {}).get("최종 점수", "50/100").split("/")[0])

        # 점수를 0~100 범위에서 가중치로 변환
        # 50점 = 기준, 70점 이상 = 높은 비중, 30점 이하 = 낮은 비중
        score_centered = (final_score - 50) / 50  # -1 ~ +1
        base_weight = max(0, (score_centered + 1) / 2)  # 0 ~ 1

        # 점수 거듭제곱 (높은 점수 종목에 더 많은 비중)
        base_weight = base_weight ** params["score_power"]

        # 2. 액션 신호 조정
        action = pred.get("action", "관망")
        action_multipliers = {
            "매수 유리": 1.3,
            "매수 고려": 1.1,
            "관망": 0.8,
            "매도 고려": params["action_penalty"],
            "매도 유리": params["action_penalty"] * 0.5,
            "판단 불가": 0.3,
        }
        action_mult = action_multipliers.get(action, 0.8)

        # 3. 신뢰도 조정 (확신 없으면 보수적으로)
        confidence = pred.get("confidence", 50) / 100
        confidence_mult = 0.5 + (confidence * 0.5)  # 50% ~ 100%

        # 4. 리스크 기반 포지션 사이징
        risk_layer = layers.get("risk", {})
        position_info = risk_layer.get("position", {})
        risk_position = position_info.get("position_pct", 50) / 100
        risk_mult = 0.5 + (risk_position * 0.5)  # 50% ~ 100%

        # 5. 레짐 조정 (약세장 종목은 축소)
        regime_layer = layers.get("regime", {})
        regime = regime_layer.get("regime", 1)
        regime_multipliers = {0: params["regime_factor"], 1: 1.0, 2: 1.15}
        regime_mult = regime_multipliers.get(regime, 1.0)

        # 6. ESG 보너스 (0~20% 추가)
        esg = esg_scores.get(ticker, 50) / 100
        esg_mult = 0.9 + (esg * 0.2)

        # 7. 변동성 조정 (고변동성 종목은 축소)
        risk_metrics = risk_layer.get("metrics", {})
        annual_vol = risk_metrics.get("annual_volatility", 30)
        vol_mult = min(1.0, 30 / (annual_vol + 1))  # 30% 변동성 기준

        # 종합 점수 계산
        combined_score = (
            base_weight *
            action_mult *
            confidence_mult *
            risk_mult *
            regime_mult *
            esg_mult *
            vol_mult *
            market_multiplier
        )

        raw_scores[ticker] = combined_score
        details[ticker] = {
            "final_score": final_score,
            "action": action,
            "action_mult": action_mult,
            "confidence": pred.get("confidence", 50),
            "confidence_mult": round(confidence_mult, 3),
            "risk_position": position_info.get("position_pct", 50),
            "risk_mult": round(risk_mult, 3),
            "regime": regime_layer.get("name", "중립"),
            "regime_mult": regime_mult,
            "esg": esg_scores.get(ticker, 50),
            "esg_mult": round(esg_mult, 3),
            "volatility": annual_vol,
            "vol_mult": round(vol_mult, 3),
            "market_env": market_env_score,
            "market_mult": market_multiplier,
            "raw_score": round(combined_score, 4),
        }

    # 정규화
    total_raw = sum(raw_scores.values())
    if total_raw > 0:
        weights = {t: raw_scores[t] / total_raw for t in tickers}
    else:
        weights = {t: 1.0 / len(tickers) for t in tickers}

    # 최소/최대 비중 제한 적용
    weights = _apply_weight_constraints(weights, min_weight, max_weight)

    # 결과 조합
    for ticker in tickers:
        details[ticker]["final_weight"] = round(weights[ticker], 4)
        details[ticker]["final_weight_pct"] = round(weights[ticker] * 100, 2)

    return {
        "weights": weights,
        "details": details,
        "market_env_score": market_env_score,
        "market_multiplier": market_multiplier,
        "risk_tolerance": risk_tolerance,
    }


def _apply_weight_constraints(weights, min_weight, max_weight):
    """최소/최대 비중 제한 적용 (반복 정규화)"""
    tickers = list(weights.keys())
    constrained = weights.copy()

    for _ in range(10):  # 최대 10회 반복
        adjusted = False

        # 최소 비중 미만 → 최소로
        for t in tickers:
            if 0 < constrained[t] < min_weight:
                constrained[t] = min_weight
                adjusted = True

        # 최대 비중 초과 → 최대로
        for t in tickers:
            if constrained[t] > max_weight:
                constrained[t] = max_weight
                adjusted = True

        # 재정규화
        total = sum(constrained.values())
        if total > 0:
            constrained = {t: constrained[t] / total for t in tickers}

        if not adjusted:
            break

    return constrained


def generate_rebalance_orders(current_holdings, target_weights, total_value,
                               predictions, min_trade_pct=0.02, min_trade_amount=50000):
    """
    리밸런싱 매수/매도 주문 생성

    Args:
        current_holdings: dict {ticker: 현재 보유액}
        target_weights: dict {ticker: 목표 비중}
        total_value: float (총 투자금)
        predictions: dict (예측 결과)
        min_trade_pct: float (최소 거래 비율)
        min_trade_amount: float (최소 거래 금액)

    Returns:
        list: 주문 목록 [{ticker, action, amount, reason, priority, ...}, ...]
    """
    orders = []

    for ticker in target_weights:
        current_amount = current_holdings.get(ticker, 0)
        current_weight = current_amount / total_value if total_value > 0 else 0
        target_weight = target_weights.get(ticker, 0)
        target_amount = total_value * target_weight

        diff = target_amount - current_amount
        diff_pct = abs(diff) / total_value if total_value > 0 else 0

        # 최소 거래 기준 확인
        if abs(diff) < min_trade_amount or diff_pct < min_trade_pct:
            orders.append({
                "ticker": ticker,
                "action": "유지",
                "amount": 0,
                "current_amount": current_amount,
                "target_amount": target_amount,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "reason": "최소 거래 기준 미달",
                "priority": 0,
                "signal": "⚪",
            })
            continue

        # 예측 정보 가져오기
        pred = predictions.get(ticker, {}) if ticker != "__meta__" else {}
        action_signal = pred.get("action", "관망")
        confidence = pred.get("confidence", 50)
        final_score = float(pred.get("signals", {}).get("최종 점수", "50/100").split("/")[0])

        if diff > 0:
            # 매수
            priority = _calculate_priority(final_score, confidence, "buy")
            reason = _generate_buy_reason(pred, diff_pct)
            signal = "🟢" if action_signal in ["매수 유리", "매수 고려"] else "🟡"

            orders.append({
                "ticker": ticker,
                "action": "매수",
                "amount": diff,
                "current_amount": current_amount,
                "target_amount": target_amount,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "reason": reason,
                "priority": priority,
                "signal": signal,
                "action_signal": action_signal,
                "confidence": confidence,
                "final_score": final_score,
            })
        else:
            # 매도
            priority = _calculate_priority(final_score, confidence, "sell")
            reason = _generate_sell_reason(pred, diff_pct)
            signal = "🔴" if action_signal in ["매도 유리", "매도 고려"] else "🟡"

            orders.append({
                "ticker": ticker,
                "action": "매도",
                "amount": abs(diff),
                "current_amount": current_amount,
                "target_amount": target_amount,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "reason": reason,
                "priority": priority,
                "signal": signal,
                "action_signal": action_signal,
                "confidence": confidence,
                "final_score": final_score,
            })

    # 우선순위 정렬 (매도 먼저, 그 다음 매수)
    sell_orders = sorted([o for o in orders if o["action"] == "매도"],
                         key=lambda x: x["priority"], reverse=True)
    buy_orders = sorted([o for o in orders if o["action"] == "매수"],
                        key=lambda x: x["priority"], reverse=True)
    hold_orders = [o for o in orders if o["action"] == "유지"]

    return sell_orders + buy_orders + hold_orders


def _calculate_priority(final_score, confidence, action_type):
    """주문 우선순위 계산 (0~100)"""
    if action_type == "buy":
        # 점수 높고 신뢰도 높으면 우선 매수
        return (final_score - 50) * (confidence / 100)
    else:
        # 점수 낮고 신뢰도 높으면 우선 매도
        return (50 - final_score) * (confidence / 100)


def _generate_buy_reason(pred, diff_pct):
    """매수 이유 생성"""
    reasons = []

    action = pred.get("action", "관망")
    if action == "매수 유리":
        reasons.append("강력한 매수 신호")
    elif action == "매수 고려":
        reasons.append("매수 신호")

    layers = pred.get("layers", {})

    # 레짐
    regime = layers.get("regime", {})
    if regime.get("regime") == 2:
        reasons.append(f"강세장 ({regime.get('name', '')})")

    # ML
    ml = layers.get("ml", {})
    if ml.get("prob_up", 0) > 0.6:
        reasons.append(f"ML 상승확률 {ml.get('prob_up', 0):.0%}")

    # 크로스 에셋
    cross = layers.get("cross_asset", {})
    if cross.get("overall_signal") == "FAVORABLE":
        reasons.append("시장환경 우호적")

    if not reasons:
        reasons.append(f"비중 조정 (+{diff_pct:.1%})")

    return " | ".join(reasons)


def _generate_sell_reason(pred, diff_pct):
    """매도 이유 생성"""
    reasons = []

    action = pred.get("action", "관망")
    if action == "매도 유리":
        reasons.append("강력한 매도 신호")
    elif action == "매도 고려":
        reasons.append("매도 신호")

    layers = pred.get("layers", {})

    # 레짐
    regime = layers.get("regime", {})
    if regime.get("regime") == 0:
        reasons.append(f"약세장 ({regime.get('name', '')})")

    # ML
    ml = layers.get("ml", {})
    if ml.get("prob_down", 0) > 0.5:
        reasons.append(f"ML 하락확률 {ml.get('prob_down', 0):.0%}")

    # 리스크
    risk = layers.get("risk", {})
    metrics = risk.get("metrics", {})
    if metrics.get("max_drawdown", 0) < -25:
        reasons.append(f"높은 낙폭 위험 ({metrics.get('max_drawdown', 0):.0f}%)")

    if not reasons:
        reasons.append(f"비중 조정 (-{diff_pct:.1%})")

    return " | ".join(reasons)


def calculate_expected_performance(orders, predictions):
    """
    리밸런싱 후 예상 성과 계산

    Returns:
        dict: 예상 수익률, 리스크 등
    """
    total_weight = 0
    weighted_return = 0
    weighted_confidence = 0
    weighted_risk = 0

    for order in orders:
        ticker = order["ticker"]
        weight = order["target_weight"]

        if ticker in predictions and ticker != "__meta__":
            pred = predictions[ticker]
            pred_return = pred.get("predicted_return", 0)
            confidence = pred.get("confidence", 50)

            layers = pred.get("layers", {})
            risk = layers.get("risk", {}).get("metrics", {})
            sharpe = risk.get("sharpe_ratio", 0)

            weighted_return += weight * pred_return
            weighted_confidence += weight * confidence
            weighted_risk += weight * sharpe
            total_weight += weight

    if total_weight > 0:
        avg_return = weighted_return / total_weight
        avg_confidence = weighted_confidence / total_weight
        avg_sharpe = weighted_risk / total_weight
    else:
        avg_return = avg_confidence = avg_sharpe = 0

    # 매수/매도 요약
    buy_count = sum(1 for o in orders if o["action"] == "매수")
    sell_count = sum(1 for o in orders if o["action"] == "매도")
    buy_amount = sum(o["amount"] for o in orders if o["action"] == "매수")
    sell_amount = sum(o["amount"] for o in orders if o["action"] == "매도")

    return {
        "expected_return": round(avg_return * 100, 2),
        "avg_confidence": round(avg_confidence, 1),
        "avg_sharpe": round(avg_sharpe, 3),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "net_flow": buy_amount - sell_amount,
    }
