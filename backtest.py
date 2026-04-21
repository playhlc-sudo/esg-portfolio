import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
import streamlit as st

# 3단계 파이프라인 엔진 임포트
from regime_engine import analyze_regime
from ml_engine import analyze_ml
from risk_engine import analyze_risk
from cross_asset_engine import fetch_cross_asset_data, calculate_cross_asset_score, find_pair_opportunities
from rebalance_engine import calculate_optimal_weights
from news_engine import fetch_comprehensive_news, get_market_pulse, analyze_news_impact_on_portfolio


# -- ESG score auto-fetch --
def get_esg_scores(tickers):
    results = {}
    # yfinance 및 관련 HTTP 에러 로그 억제
    loggers_to_suppress = ['yfinance', 'urllib3', 'requests', 'peewee']
    original_levels = {}
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.CRITICAL)

    for t in tickers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker_obj = yf.Ticker(t)
                sus = ticker_obj.sustainability
            if sus is not None and not sus.empty:
                total = sus.loc['totalEsg'].values[0] if 'totalEsg' in sus.index else None
                if total is not None and not np.isnan(total):
                    score = max(0, min(100, int(100 - total)))
                    results[t] = score
                    continue
        except Exception:
            pass
        results[t] = 50

    # 로그 레벨 복원
    for logger_name, level in original_levels.items():
        logging.getLogger(logger_name).setLevel(level)
    return results


POSITIVE_KEYWORDS = [
    'surge', 'jump', 'gain', 'rise', 'bull', 'beat', 'record', 'high',
    'upgrade', 'buy', 'strong', 'growth', 'profit', 'revenue', 'outperform',
    'positive', 'optimistic', 'rally', 'boom', 'soar', 'breakthrough',
    'dividend', 'innovation', 'expand', 'partnership', 'deal', 'approval',
]
NEGATIVE_KEYWORDS = [
    'drop', 'fall', 'decline', 'bear', 'miss', 'low', 'downgrade', 'sell',
    'weak', 'loss', 'debt', 'risk', 'crash', 'plunge', 'cut', 'layoff',
    'negative', 'pessimistic', 'slump', 'recession', 'bankruptcy', 'lawsuit',
    'warning', 'investigation', 'fraud', 'recall', 'fine', 'penalty',
]

SECTOR_ETF_MAP = {
    'Technology': ['XLK', 'QQQ'],
    'Communication Services': ['XLC'],
    'Consumer Cyclical': ['XLY'],
    'Consumer Defensive': ['XLP'],
    'Energy': ['XLE'],
    'Financial Services': ['XLF'],
    'Healthcare': ['XLV'],
    'Industrials': ['XLI'],
    'Basic Materials': ['XLB'],
    'Real Estate': ['XLRE', 'VNQ'],
    'Utilities': ['XLU'],
}

MACRO_ETFS = ['SPY', 'TLT', 'GLD', 'UUP', 'USO']
MACRO_LABELS = {
    'SPY': 'US Market',
    'TLT': 'US Bonds',
    'GLD': 'Gold',
    'UUP': 'USD',
    'USO': 'Oil',
}


def _extract_news_from_ticker(ticker_obj):
    try:
        raw_news = ticker_obj.news
    except Exception:
        return []
    news_items = []
    if isinstance(raw_news, list):
        news_items = raw_news
    elif isinstance(raw_news, dict) and 'news' in raw_news:
        news_items = raw_news['news']
    return news_items


def _parse_news_item(item):
    title = ""
    publisher = ""
    link = ""
    if isinstance(item, dict):
        title = item.get('title', item.get('headline', ''))
        publisher = item.get('publisher', item.get('source', ''))
        link = item.get('link', item.get('url', ''))
        if not title and 'content' in item:
            content = item['content']
            if isinstance(content, dict):
                title = content.get('title', '')
                link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else ''
    return title, publisher, link


def _score_title(title):
    title_lower = title.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in title_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in title_lower)
    if pos_count > neg_count:
        return "호재", min(1.0, pos_count * 0.3)
    elif neg_count > pos_count:
        return "악재", max(-1.0, -neg_count * 0.3)
    else:
        return "중립", 0.0


def _analyze_ticker_news(ticker_symbol, tag, max_items=5):
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        news_items = _extract_news_from_ticker(ticker_obj)
    except Exception:
        return [], []
    analyzed = []
    scores = []
    seen_titles = set()
    for item in news_items[:10]:
        title, publisher, link = _parse_news_item(item)
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        sentiment, score = _score_title(title)
        scores.append(score)
        analyzed.append({
            "title": title, "publisher": publisher,
            "sentiment": sentiment, "link": link, "tag": tag,
        })
    return analyzed[:max_items], scores


def analyze_news_sentiment(ticker_symbol):
    """
    능동적 뉴스 분석 (news_engine 사용)
    - 종목 직접 뉴스
    - 경쟁사 뉴스
    - 섹터 뉴스
    - 거시경제 뉴스
    - 영향도 분석
    """
    # 새로운 포괄적 뉴스 엔진 사용
    result = fetch_comprehensive_news(
        ticker_symbol,
        include_competitors=True,
        include_macro=True,
        include_supply_chain=True
    )
    return result


def _empty_prediction(news_result, reason="판단 불가"):
    return {
        "direction": reason, "confidence": 0, "predicted_return": 0.0,
        "signals": {}, "recommended_weight": 0.0,
        "news": news_result, "action": "판단 불가",
        "layers": {
            "regime": {"score": 50, "regime": 1, "name": "중립", "hurst": 0.5,
                       "hurst_interp": "", "strategy_hint": "", "confidence": 0, "method": ""},
            "ml": {"score": 50, "direction": "중립", "prob_up": 0, "prob_neutral": 1,
                   "prob_down": 0, "accuracy": 0, "top_features": {}, "method": ""},
            "risk": {"score": 50, "metrics": {}, "position": {"position_pct": 0, "risk_based": 0, "vol_based": 0}},
        },
    }


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_price_data(tickers_tuple, start_str, end_str):
    """캐싱된 가격 데이터 조회"""
    tickers = list(tickers_tuple)
    loggers = ['yfinance', 'urllib3', 'requests', 'peewee']
    original_levels = {}
    for name in loggers:
        logger = logging.getLogger(name)
        original_levels[name] = logger.level
        logger.setLevel(logging.CRITICAL)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(tickers, start=start_str, end=end_str, progress=False)
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)

    return raw


def predict_future(tickers, esg_scores, lookback_days=120, layer_weights=None, cross_asset_weight=0.15):
    """
    종합 투자 분석 (4단계 파이프라인)

    Args:
        tickers: list
        esg_scores: dict
        lookback_days: int
        layer_weights: dict (regime/ml/risk 가중치)
        cross_asset_weight: float (크로스 에셋 시그널 가중치)
    """
    if layer_weights is None:
        layer_weights = {"regime": 0.20, "ml": 0.40, "risk": 0.25}

    # 크로스 에셋 가중치 반영 후 정규화
    total_layer = sum(layer_weights.values())
    adjusted_weights = {k: v * (1 - cross_asset_weight) / total_layer for k, v in layer_weights.items()}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=max(lookback_days + 200, 500))

    # 캐싱된 데이터 조회
    raw = _fetch_price_data(
        tuple(tickers),
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    prices = raw["Close"].dropna(how="all")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index()

    # 크로스 에셋 시그널 조회 (한 번만)
    cross_data = fetch_cross_asset_data(lookback_days)
    cross_asset_result = calculate_cross_asset_score(cross_data)
    cross_asset_score = cross_asset_result["cross_asset_score"]

    # 페어 트레이딩 기회 탐색
    ticker_prices = {t: prices[t].dropna() for t in tickers if t in prices.columns}
    pair_opportunities = find_pair_opportunities(ticker_prices) if len(ticker_prices) >= 2 else []

    predictions = {}
    for ticker in tickers:
        news_result = analyze_news_sentiment(ticker)
        if ticker not in prices.columns:
            predictions[ticker] = _empty_prediction(news_result, "데이터 없음")
            continue
        p = prices[ticker].dropna()
        if len(p) < 60:
            predictions[ticker] = _empty_prediction(news_result, "데이터 부족")
            continue
        # Stage 1: 시장 레짐 판단
        regime_result = analyze_regime(p)
        regime_score = regime_result["regime_score"]
        # Stage 2: ML 예측
        regime_history = regime_result.get("regime_history", None)
        ml_result = analyze_ml(p, regime_history=regime_history)
        ml_score = ml_result["ml_score"]
        # Stage 3: 리스크 필터링 + 최종 결정
        risk_result = analyze_risk(p, regime_score, ml_score, ml_result)
        decision = risk_result["decision"]
        risk_metrics = risk_result["risk_metrics"]

        # Stage 4: 크로스 에셋 시그널 반영
        base_score = (
            regime_score * adjusted_weights["regime"] +
            ml_score * adjusted_weights["ml"] +
            risk_result["risk_score"] * adjusted_weights["risk"] +
            cross_asset_score * cross_asset_weight
        )

        # 뉴스 감성 미세 조정
        news_signal = news_result["sentiment_score"]
        news_adjustment = news_signal * 3  # 뉴스 영향력 조정
        adjusted_score = max(0, min(100, base_score + news_adjustment))
        # 최종 방향/액션
        if adjusted_score >= 70:
            action = "매수 유리"
        elif adjusted_score >= 58:
            action = "매수 고려"
        elif adjusted_score <= 30:
            action = "매도 유리"
        elif adjusted_score <= 42:
            action = "매도 고려"
        else:
            action = "관망"
        if adjusted_score > 55:
            direction = "상승"
        elif adjusted_score < 45:
            direction = "하락"
        else:
            direction = "보합"
        confidence = min(100, int(abs(adjusted_score - 50) * 2))
        predicted_return = (adjusted_score - 50) * 0.001
        # 기존 기술지표 (UI 호환)
        ma20 = p.rolling(20).mean().iloc[-1]
        ma60 = p.rolling(60).mean().iloc[-1]
        current_price = p.iloc[-1]
        ma_trend = "상승" if current_price > ma20 > ma60 else ("하락" if current_price < ma20 < ma60 else "중립")
        delta = p.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        momentum = (p.iloc[-1] / p.iloc[-20] - 1)
        volatility = p.pct_change().iloc[-20:].std()
        predictions[ticker] = {
            "direction": direction, "confidence": confidence,
            "predicted_return": predicted_return,
            "signals": {
                "시장 레짐": f"{regime_result['regime_name']} ({regime_result.get('method', '')})",
                "레짐 점수": f"{regime_score:.1f}/100",
                "Hurst 지수": f"{regime_result.get('hurst', 0.5):.3f} ({regime_result.get('interpretation', '')})",
                "ML 예측": f"{ml_result['direction']} (정확도: {ml_result.get('ensemble_accuracy', 0):.1%})",
                "상승확률": f"{ml_result.get('prob_up', 0):.1%}",
                "ML 점수": f"{ml_score:.1f}/100",
                "Sharpe": f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
                "VaR(95%)": f"{risk_metrics.get('var_historical', 0):.2f}%",
                "최대낙폭": f"{risk_metrics.get('max_drawdown', 0):.1f}%",
                "리스크 점수": f"{risk_result['risk_score']:.1f}/100",
                "이동평균 추세": ma_trend,
                "RSI": f"{rsi:.1f}",
                "20일 모멘텀": f"{momentum:.1%}",
                "변동성": f"{volatility:.4f}",
                "뉴스 감성": news_result["summary"],
                "뉴스 점수": f"{news_signal:.2f}",
                "최종 점수": f"{adjusted_score:.1f}/100",
            },
            "recommended_weight": 0.0,
            "news": news_result,
            "action": action,
            "layers": {
                "regime": {
                    "score": regime_score,
                    "regime": regime_result["regime"],
                    "name": regime_result["regime_name"],
                    "hurst": regime_result.get("hurst", 0.5),
                    "hurst_interp": regime_result.get("interpretation", ""),
                    "strategy_hint": regime_result.get("strategy_hint", ""),
                    "confidence": regime_result.get("confidence", 0),
                    "method": regime_result.get("method", ""),
                },
                "ml": {
                    "score": ml_score,
                    "direction": ml_result["direction"],
                    "prob_up": ml_result.get("prob_up", 0),
                    "prob_neutral": ml_result.get("prob_neutral", 0),
                    "prob_down": ml_result.get("prob_down", 0),
                    "accuracy": ml_result.get("ensemble_accuracy", 0),
                    "top_features": ml_result.get("top_features", {}),
                    "method": ml_result.get("method", ""),
                },
                "risk": {
                    "score": risk_result["risk_score"],
                    "metrics": risk_metrics,
                    "position": risk_result["position"],
                },
                "cross_asset": {
                    "score": cross_asset_score,
                    "vix": cross_asset_result["vix"],
                    "dollar": cross_asset_result["dollar"],
                    "bond_equity": cross_asset_result["bond_equity"],
                    "gold": cross_asset_result["gold"],
                    "overall_signal": cross_asset_result["overall_signal"],
                },
            },
        }
    # 고도화된 포트폴리오 비중 계산
    weight_result = calculate_optimal_weights(
        predictions, tickers, esg_scores,
        risk_tolerance="moderate",
        min_weight=0.05,
        max_weight=0.40
    )
    optimal_weights = weight_result["weights"]
    weight_details = weight_result["details"]

    for t in tickers:
        predictions[t]["recommended_weight"] = optimal_weights.get(t, 0)
        predictions[t]["weight_detail"] = weight_details.get(t, {})

    # 메타 정보 추가
    if tickers:
        predictions["__meta__"] = {
            "cross_asset": cross_asset_result,
            "pair_opportunities": pair_opportunities,
            "layer_weights": {**adjusted_weights, "cross_asset": cross_asset_weight},
            "weight_calculation": weight_result,
        }

    return predictions


def run_backtest(tickers, start_date, end_date, initial_capital, esg_scores,
                 rebalance_freq='M', transaction_cost_rate=0.001):
    raw = yf.download(tickers, start=start_date, end=end_date)
    prices = raw["Close"].dropna(how="all")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index()
    returns = prices.pct_change().dropna()
    raw_bench = yf.download("^GSPC", start=start_date, end=end_date)
    benchmark = raw_bench["Close"]
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[:, 0]
    benchmark = benchmark.dropna().sort_index()
    benchmark_returns = benchmark.pct_change().dropna()
    freq_map = {'M': 'ME', 'Q': 'QE'}
    rebalance_freq = freq_map.get(rebalance_freq, rebalance_freq)
    period_starts = returns.resample(rebalance_freq).first().index
    period_ends = returns.resample(rebalance_freq).last().index
    if len(period_starts) < 2 or len(period_ends) < 2:
        raise ValueError("Data period too short for rebalancing.")
    esg_array = np.array([esg_scores[t] for t in tickers], dtype=float)
    prev_weights = np.zeros(len(tickers))
    current_capital = initial_capital
    current_benchmark = initial_capital
    portfolio_values = []
    benchmark_values = []
    portfolio_dates = []
    weights_history = []
    daily_portfolio_list = []
    n_periods = min(len(period_starts), len(period_ends)) - 1
    for i in range(n_periods):
        start = period_starts[i]
        end = period_ends[i + 1]
        period_returns = returns.loc[start:end]
        if period_returns.empty:
            continue
        past_returns = returns.loc[:start].iloc[:-1]
        if len(past_returns) < 5:
            predicted_return = np.ones(len(tickers))
        else:
            lookback = past_returns.iloc[-60:]
            predicted_return = lookback.mean().values
        predicted_nonneg = np.maximum(predicted_return, 0.0)
        raw_weights = esg_array * predicted_nonneg
        if raw_weights.sum() == 0:
            raw_weights = esg_array.copy()
        new_weights = raw_weights / raw_weights.sum()
        turnover = np.abs(new_weights - prev_weights).sum() / 2.0
        transaction_cost = turnover * transaction_cost_rate * current_capital
        portfolio_daily = period_returns.dot(new_weights)
        growth_series = (1 + portfolio_daily).cumprod()
        if len(growth_series) > 0:
            current_capital = (current_capital - transaction_cost) * growth_series.iloc[-1]
        else:
            current_capital = current_capital - transaction_cost
        bench_period = benchmark_returns.loc[start:end]
        if not bench_period.empty:
            bench_growth = (1 + bench_period).cumprod()
            current_benchmark = current_benchmark * bench_growth.iloc[-1]
        portfolio_values.append(current_capital)
        benchmark_values.append(current_benchmark)
        portfolio_dates.append(end)
        weights_history.append(dict(zip(tickers, new_weights)))
        daily_portfolio_list.append(portfolio_daily)
        prev_weights = new_weights.copy()
    portfolio_series = pd.Series(portfolio_values, index=portfolio_dates)
    benchmark_series = pd.Series(benchmark_values, index=portfolio_dates)
    final_value = portfolio_series.iloc[-1]
    final_benchmark = benchmark_series.iloc[-1]
    alpha = final_value - final_benchmark
    if daily_portfolio_list:
        portfolio_daily_full = pd.concat(daily_portfolio_list).sort_index()
        mean_return = portfolio_daily_full.mean()
        volatility_val = portfolio_daily_full.std()
        risk_free_rate = 0.02 / 252
        sharpe_ratio = (mean_return - risk_free_rate) / volatility_val if volatility_val != 0 else np.nan
    else:
        mean_return = volatility_val = sharpe_ratio = np.nan
    cum = portfolio_series.cummax()
    max_drawdown = (portfolio_series / cum - 1).min()
    weights_df = pd.DataFrame(weights_history, index=portfolio_dates)
    return {
        "portfolio_series": portfolio_series, "benchmark_series": benchmark_series,
        "final_value": final_value, "final_benchmark": final_benchmark, "alpha": alpha,
        "mean_return": mean_return, "volatility": volatility_val,
        "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown, "weights_df": weights_df,
    }


def optimize_layer_weights(tickers, esg_scores, lookback_days=120, method="grid"):
    """
    레이어 가중치 자동 최적화

    각 레이어별 예측 정확도와 기여도를 분석하여 최적 가중치 계산

    Args:
        tickers: list - 종목 리스트
        esg_scores: dict - ESG 점수
        lookback_days: int - 분석 기간
        method: str - 최적화 방법 ("grid" 또는 "performance")

    Returns:
        dict: 최적 가중치 및 분석 결과
    """
    from itertools import product

    end_date = datetime.now()
    start_date = end_date - timedelta(days=max(lookback_days + 200, 500))

    # 가격 데이터 조회
    raw = _fetch_price_data(
        tuple(tickers),
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    prices = raw["Close"].dropna(how="all")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.sort_index()

    # 크로스 에셋 데이터
    cross_data = fetch_cross_asset_data(lookback_days)
    cross_result = calculate_cross_asset_score(cross_data)
    cross_score = cross_result["cross_asset_score"]

    # 각 레이어별 점수 수집
    layer_scores = {"regime": [], "ml": [], "risk": [], "cross_asset": []}
    actual_returns = []

    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        p = prices[ticker].dropna()
        if len(p) < 60:
            continue

        # 각 레이어 분석
        regime_result = analyze_regime(p)
        regime_history = regime_result.get("regime_history", None)
        ml_result = analyze_ml(p, regime_history=regime_history)
        risk_result = analyze_risk(p, regime_result["regime_score"], ml_result["ml_score"], ml_result)

        layer_scores["regime"].append(regime_result["regime_score"])
        layer_scores["ml"].append(ml_result["ml_score"])
        layer_scores["risk"].append(risk_result["risk_score"])
        layer_scores["cross_asset"].append(cross_score)

        # 실제 최근 수익률 (미래 성과 추정용)
        recent_return = p.pct_change().iloc[-20:].mean() * 100 + 50  # 0-100 스케일
        actual_returns.append(recent_return)

    if not actual_returns:
        return {
            "optimal_weights": {"regime": 20, "ml": 40, "risk": 25, "cross_asset": 15},
            "method": "default",
            "message": "데이터 부족으로 기본 가중치 사용"
        }

    # 각 레이어의 예측력 계산 (상관계수 기반)
    correlations = {}
    for layer, scores in layer_scores.items():
        if len(scores) >= 2:
            corr = np.corrcoef(scores, actual_returns)[0, 1]
            correlations[layer] = max(0, corr) if not np.isnan(corr) else 0
        else:
            correlations[layer] = 0.25

    # 각 레이어의 일관성 (표준편차 기반 - 안정적일수록 높은 점수)
    consistency = {}
    for layer, scores in layer_scores.items():
        if len(scores) >= 2:
            std = np.std(scores)
            consistency[layer] = max(0, 1 - std / 50)  # 변동성이 낮을수록 높은 점수
        else:
            consistency[layer] = 0.5

    # 종합 점수 = 상관계수 * 0.7 + 일관성 * 0.3
    combined_scores = {
        layer: correlations[layer] * 0.7 + consistency[layer] * 0.3
        for layer in layer_scores.keys()
    }

    # 정규화하여 가중치 계산
    total_score = sum(combined_scores.values())
    if total_score > 0:
        raw_weights = {
            layer: (score / total_score) * 100
            for layer, score in combined_scores.items()
        }
    else:
        raw_weights = {"regime": 25, "ml": 25, "risk": 25, "cross_asset": 25}

    # 범위 제한 적용 (각 레이어 최소 5%, 최대 50%)
    min_weight, max_weight = 5, 50
    adjusted_weights = {}
    for layer, weight in raw_weights.items():
        if layer == "cross_asset":
            adjusted_weights[layer] = max(5, min(30, int(round(weight))))
        else:
            adjusted_weights[layer] = max(min_weight, min(max_weight, int(round(weight))))

    # 합계 100으로 정규화
    total = sum(adjusted_weights.values())
    if total != 100:
        diff = 100 - total
        # ML에 차이 적용 (가장 유연한 레이어)
        adjusted_weights["ml"] = max(min_weight, min(max_weight, adjusted_weights["ml"] + diff))

        # 여전히 100이 아니면 다른 레이어에서 조정
        total = sum(adjusted_weights.values())
        if total != 100:
            diff = 100 - total
            for layer in ["regime", "risk"]:
                if adjusted_weights[layer] + diff >= min_weight and adjusted_weights[layer] + diff <= max_weight:
                    adjusted_weights[layer] += diff
                    break

    # 최종 확인 및 강제 정규화
    total = sum(adjusted_weights.values())
    if total != 100:
        scale = 100 / total
        adjusted_weights = {k: int(round(v * scale)) for k, v in adjusted_weights.items()}
        diff = 100 - sum(adjusted_weights.values())
        if diff != 0:
            adjusted_weights["ml"] += diff

    return {
        "optimal_weights": adjusted_weights,
        "method": method,
        "correlations": {k: round(v, 3) for k, v in correlations.items()},
        "consistency": {k: round(v, 3) for k, v in consistency.items()},
        "combined_scores": {k: round(v, 3) for k, v in combined_scores.items()},
        "message": "레이어별 예측 정확도 및 일관성 기반 최적화 완료",
        "details": {
            "regime": f"상관계수: {correlations['regime']:.2f}, 일관성: {consistency['regime']:.2f}",
            "ml": f"상관계수: {correlations['ml']:.2f}, 일관성: {consistency['ml']:.2f}",
            "risk": f"상관계수: {correlations['risk']:.2f}, 일관성: {consistency['risk']:.2f}",
            "cross_asset": f"상관계수: {correlations['cross_asset']:.2f}, 일관성: {consistency['cross_asset']:.2f}",
        }
    }
