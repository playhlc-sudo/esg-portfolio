"""
ML 예측 엔진 (Stage 2)
- Random Forest + Gradient Boosting 앙상블
- 레짐 정보를 피처로 활용
- 40+ 기술적 피처 자동 생성
- 1.txt TwoSigmaModule 참고
"""
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


def build_features(prices_df, regime_history=None):
    """
    기술적 피처 매트릭스 생성 (40+ 피처)

    Args:
        prices_df: DataFrame with columns Close, High, Low, Volume (or just Close as Series)
        regime_history: pd.Series from regime_engine (optional)

    Returns:
        pd.DataFrame: 피처 매트릭스
    """
    if isinstance(prices_df, pd.Series):
        close = prices_df
        has_ohlcv = False
    else:
        close = prices_df["Close"] if "Close" in prices_df.columns else prices_df.iloc[:, 0]
        has_ohlcv = all(c in prices_df.columns for c in ["High", "Low", "Volume"])

    feat = pd.DataFrame(index=close.index)

    # 수익률 피처 (다중 시차)
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"ret_{lag}d"] = close.pct_change(lag)

    # 변동성 피처
    ret = close.pct_change()
    for window in [5, 10, 20, 60]:
        feat[f"vol_{window}d"] = ret.rolling(window).std()

    # 변동성 비율 (단기/장기)
    feat["vol_ratio_5_20"] = feat["vol_5d"] / (feat["vol_20d"] + 1e-10)
    feat["vol_ratio_20_60"] = feat["vol_20d"] / (feat["vol_60d"] + 1e-10)

    # 모멘텀 피처
    for window in [5, 10, 20, 60]:
        feat[f"mom_{window}d"] = close / close.shift(window) - 1

    # 이동평균 비율
    for window in [5, 10, 20, 50, 200]:
        ma = close.rolling(window).mean()
        feat[f"ma_ratio_{window}d"] = close / ma - 1

    # RSI (14일)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    feat["macd"] = ema12 - ema26
    feat["macd_signal"] = feat["macd"].ewm(span=9).mean()
    feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

    # 볼린저 밴드 위치
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    feat["bb_position"] = (close - bb_mid) / (bb_std * 2 + 1e-10)

    # 통계적 피처 (왜도, 첨도)
    for window in [20, 60]:
        feat[f"skew_{window}d"] = ret.rolling(window).skew()
        feat[f"kurt_{window}d"] = ret.rolling(window).kurt()

    # 자기상관
    for lag in [1, 5]:
        feat[f"autocorr_{lag}"] = ret.rolling(60).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    # OHLCV 피처 (가능한 경우)
    if has_ohlcv:
        feat["high_low_range"] = (prices_df["High"] - prices_df["Low"]) / close
        feat["close_position"] = (close - prices_df["Low"]) / (prices_df["High"] - prices_df["Low"] + 1e-10)
        feat["vol_ma_ratio"] = prices_df["Volume"] / prices_df["Volume"].rolling(20).mean()

    # 레짐 피처 (Stage 1 연동)
    if regime_history is not None:
        common = feat.index.intersection(regime_history.index)
        regime_aligned = regime_history.reindex(feat.index, method="ffill")
        feat["regime"] = regime_aligned
        feat["regime_is_bull"] = (regime_aligned == 2).astype(float)
        feat["regime_is_bear"] = (regime_aligned == 0).astype(float)

    return feat.replace([np.inf, -np.inf], np.nan)


def generate_target(close, forward_days=5, threshold=0.01):
    """
    예측 타겟 생성: N일 후 수익률 기반 3분류

    Returns:
        pd.Series: 0=하락, 1=중립, 2=상승
    """
    future_ret = close.pct_change(forward_days).shift(-forward_days)
    target = pd.Series(1, index=close.index)
    target[future_ret > threshold] = 2
    target[future_ret < -threshold] = 0
    return target


def train_and_predict(features, target, test_size=60):
    """
    앙상블 ML 모델 학습 및 예측

    Args:
        features: pd.DataFrame (피처)
        target: pd.Series (타겟)
        test_size: int (테스트 기간 일수)

    Returns:
        dict: ml_score(0~100), probabilities, accuracy, feature_importance 등
    """
    if not ML_AVAILABLE:
        return _fallback_prediction(features)

    combined = pd.concat([features, target.rename("target")], axis=1).dropna()

    if len(combined) < 200:
        return _fallback_prediction(features)

    X = combined.drop("target", axis=1)
    y = combined["target"]

    # 시계열 분리
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    if len(X_train) < 100 or len(X_test) < 10:
        return _fallback_prediction(features)

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 앙상블 모델
    models = {
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ),
        "gb": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, random_state=42
        ),
    }

    model_results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        model_results[name] = {"accuracy": acc, "proba": proba}

        if hasattr(model, "feature_importances_"):
            model_results[name]["importance"] = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)

    # 앙상블 확률 (평균)
    ensemble_proba = np.mean(
        [model_results[m]["proba"] for m in model_results], axis=0
    )
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    # 최신 예측
    latest_proba = ensemble_proba[-1]
    prob_down, prob_neutral, prob_up = latest_proba[0], latest_proba[1], latest_proba[2]

    # ML 점수 산출 (0~100): 상승 확률 기반
    ml_score = round(prob_up * 100, 1)

    # 예측 방향
    if prob_up > 0.5:
        direction = "상승"
    elif prob_down > 0.5:
        direction = "하락"
    else:
        direction = "중립"

    # 신뢰도 = 가장 높은 확률
    confidence = round(float(max(latest_proba)) * 100, 1)

    # 예측 수익률 추정 (확률 가중 기대값)
    predicted_return = (prob_up * 0.03) + (prob_neutral * 0.0) + (prob_down * -0.03)

    return {
        "ml_score": ml_score,
        "direction": direction,
        "confidence": confidence,
        "predicted_return": round(predicted_return, 4),
        "prob_up": round(float(prob_up), 3),
        "prob_neutral": round(float(prob_neutral), 3),
        "prob_down": round(float(prob_down), 3),
        "ensemble_accuracy": round(ensemble_acc, 3),
        "model_accuracies": {m: round(model_results[m]["accuracy"], 3) for m in model_results},
        "top_features": _get_top_features(model_results),
        "method": "ML_Ensemble",
    }


def _get_top_features(model_results, top_n=5):
    """상위 피처 중요도 추출"""
    for name in model_results:
        if "importance" in model_results[name]:
            imp = model_results[name]["importance"]
            return {k: round(float(v), 4) for k, v in imp.head(top_n).items()}
    return {}


def _fallback_prediction(features):
    """ML 불가 시 기술적 지표 기반 폴백 예측"""
    score = 50.0
    direction = "중립"

    if features is not None and len(features) > 0:
        last = features.iloc[-1]

        # RSI 기반
        rsi = last.get("rsi_14", 50)
        if not pd.isna(rsi):
            if rsi < 30:
                score += 15
            elif rsi < 45:
                score += 5
            elif rsi > 70:
                score -= 15
            elif rsi > 55:
                score -= 5

        # 모멘텀 기반
        mom = last.get("mom_20d", 0)
        if not pd.isna(mom):
            score += mom * 100  # 5% 모멘텀 → +5점

        # MACD 기반
        macd_hist = last.get("macd_hist", 0)
        if not pd.isna(macd_hist) and macd_hist > 0:
            score += 5
        elif not pd.isna(macd_hist):
            score -= 5

        score = max(0, min(100, score))

        if score > 60:
            direction = "상승"
        elif score < 40:
            direction = "하락"

    return {
        "ml_score": round(score, 1),
        "direction": direction,
        "confidence": 30.0,
        "predicted_return": (score - 50) * 0.001,
        "prob_up": round(score / 100, 3),
        "prob_neutral": 0.3,
        "prob_down": round(1 - score / 100, 3),
        "ensemble_accuracy": 0.0,
        "model_accuracies": {},
        "top_features": {},
        "method": "Fallback_Technical",
    }


def analyze_ml(prices, regime_history=None, forward_days=5, threshold=0.01):
    """
    Stage 2 통합 분석: 피처 생성 → ML 학습/예측

    Args:
        prices: pd.Series or DataFrame (Close 포함)
        regime_history: pd.Series (Stage 1 결과, optional)
        forward_days: int
        threshold: float

    Returns:
        dict: ML 분석 결과
    """
    if isinstance(prices, pd.Series):
        close = prices
    else:
        close = prices["Close"] if "Close" in prices.columns else prices.iloc[:, 0]

    features = build_features(prices, regime_history)
    target = generate_target(close, forward_days, threshold)

    result = train_and_predict(features, target)
    return result
