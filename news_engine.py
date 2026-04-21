"""
능동적 뉴스 수집 엔진
- 종목 직접 뉴스
- 업종/산업 뉴스
- 경쟁사 뉴스
- 공급망 뉴스
- 정책/규제 뉴스
- 거시경제 뉴스
- 뉴스 영향도 분석
"""
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import logging
import warnings
import re

# ========== 산업별 키워드 매핑 ==========
INDUSTRY_KEYWORDS = {
    "Technology": ["AI", "artificial intelligence", "semiconductor", "chip", "cloud", "data center",
                   "software", "SaaS", "cybersecurity", "quantum computing", "5G", "metaverse"],
    "Semiconductors": ["chip", "semiconductor", "foundry", "wafer", "TSMC", "memory", "DRAM", "NAND",
                       "GPU", "CPU", "AI chip", "advanced packaging", "EUV", "lithography"],
    "Consumer Cyclical": ["retail", "e-commerce", "consumer spending", "holiday sales",
                          "consumer confidence", "discretionary spending"],
    "Financial Services": ["interest rate", "Fed", "Federal Reserve", "banking", "credit",
                           "loan", "mortgage", "inflation", "treasury yield"],
    "Healthcare": ["FDA", "drug approval", "clinical trial", "pharma", "biotech",
                   "healthcare policy", "Medicare", "insurance"],
    "Energy": ["oil price", "OPEC", "crude oil", "natural gas", "renewable", "solar",
               "wind energy", "EV", "battery", "energy transition"],
    "Industrials": ["manufacturing", "supply chain", "logistics", "infrastructure",
                    "construction", "aerospace", "defense spending"],
    "Communication Services": ["streaming", "advertising", "social media", "telecom",
                               "content", "media", "subscriber"],
    "Consumer Defensive": ["grocery", "food prices", "consumer staples", "inflation",
                           "essential goods"],
    "Real Estate": ["housing", "mortgage rates", "commercial real estate", "REIT",
                    "property", "rent"],
    "Utilities": ["electricity", "power grid", "renewable energy", "utility rates",
                  "energy demand"],
    "Basic Materials": ["commodity", "mining", "steel", "copper", "lithium",
                        "raw materials", "supply shortage"],
}

# ========== 주요 경쟁사 매핑 ==========
COMPETITOR_MAP = {
    # 반도체
    "NVDA": ["AMD", "INTC", "TSM", "AVGO", "QCOM", "MU"],
    "AMD": ["NVDA", "INTC", "TSM", "QCOM"],
    "INTC": ["AMD", "NVDA", "TSM", "QCOM"],
    "TSM": ["INTC", "SSNLF", "NVDA", "AMD"],
    "SOXL": ["SMH", "SOXX", "NVDA", "AMD", "TSM"],
    "SMH": ["SOXL", "SOXX", "NVDA", "AMD"],

    # 빅테크
    "AAPL": ["MSFT", "GOOGL", "AMZN", "META", "SSNLF"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "CRM"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL"],
    "AMZN": ["WMT", "MSFT", "GOOGL", "BABA"],
    "META": ["GOOGL", "SNAP", "PINS", "TWTR"],

    # 전기차/에너지
    "TSLA": ["RIVN", "LCID", "F", "GM", "NIO", "BYD"],
    "RIVN": ["TSLA", "LCID", "F", "GM"],
    "NIO": ["TSLA", "LI", "XPEV", "BYD"],

    # 금융
    "JPM": ["BAC", "GS", "MS", "C", "WFC"],
    "GS": ["MS", "JPM", "BAC"],

    # 스트리밍/미디어
    "NFLX": ["DIS", "WBD", "PARA", "AMZN"],
    "DIS": ["NFLX", "WBD", "CMCSA"],
}

# ========== 공급망 키워드 ==========
SUPPLY_CHAIN_KEYWORDS = {
    "Technology": ["supply chain", "chip shortage", "component", "manufacturing delay",
                   "logistics", "inventory", "production"],
    "Semiconductors": ["wafer shortage", "foundry capacity", "lead time", "chip supply",
                       "equipment", "ASML", "Applied Materials", "Lam Research"],
    "Consumer Cyclical": ["inventory levels", "shipping costs", "port congestion",
                          "delivery delays", "warehouse"],
    "Healthcare": ["drug shortage", "API supply", "manufacturing capacity", "distribution"],
    "Energy": ["pipeline", "refinery", "storage", "transport", "infrastructure"],
}

# ========== 정책/규제 키워드 ==========
POLICY_KEYWORDS = {
    "Technology": ["antitrust", "regulation", "privacy law", "data protection", "GDPR",
                   "Section 230", "tech regulation", "EU Digital"],
    "Semiconductors": ["CHIPS Act", "export control", "China ban", "tariff",
                       "trade restriction", "national security"],
    "Financial Services": ["banking regulation", "Basel", "Dodd-Frank", "stress test",
                           "capital requirement", "CFPB"],
    "Healthcare": ["FDA", "drug pricing", "Medicare", "ACA", "healthcare reform",
                   "patent expiry", "generic"],
    "Energy": ["carbon tax", "emissions", "EPA", "climate policy", "green energy subsidy",
               "drilling permit", "pipeline approval"],
}

# ========== 감성 분석 키워드 (확장) ==========
POSITIVE_KEYWORDS = [
    "surge", "soar", "jump", "rally", "beat", "exceed", "record", "breakthrough",
    "upgrade", "outperform", "bullish", "strong", "growth", "profit", "gain",
    "partnership", "deal", "acquisition", "expansion", "innovation", "launch",
    "approval", "win", "success", "milestone", "optimistic", "positive",
    "buy rating", "price target raised", "upside", "momentum", "recovery",
    "dividend increase", "buyback", "repurchase", "guidance raise"
]

NEGATIVE_KEYWORDS = [
    "crash", "plunge", "drop", "fall", "miss", "below", "decline", "loss",
    "downgrade", "underperform", "bearish", "weak", "cut", "slash", "warning",
    "lawsuit", "investigation", "scandal", "recall", "delay", "cancel",
    "reject", "fail", "concern", "risk", "threat", "pressure", "selloff",
    "sell rating", "price target cut", "downside", "layoff", "restructure",
    "guidance cut", "disappointing", "bankruptcy", "default"
]

# ========== 거시경제 뉴스 소스 ==========
MACRO_TICKERS = {
    "SPY": "S&P 500 시장",
    "QQQ": "나스닥/기술주",
    "DIA": "다우/산업주",
    "IWM": "소형주",
    "TLT": "미국채 장기",
    "HYG": "하이일드 채권",
    "GLD": "금/안전자산",
    "UUP": "달러 강세",
    "USO": "유가/에너지",
    "VIX": "변동성/공포지수",
    "FXI": "중국 시장",
    "EEM": "신흥국 시장",
}


def _suppress_logs():
    """yfinance 로그 억제"""
    loggers = ['yfinance', 'urllib3', 'requests', 'peewee']
    original = {}
    for name in loggers:
        logger = logging.getLogger(name)
        original[name] = logger.level
        logger.setLevel(logging.CRITICAL)
    return original


def _restore_logs(original):
    """로그 레벨 복원"""
    for name, level in original.items():
        logging.getLogger(name).setLevel(level)


def _extract_news(ticker_obj):
    """뉴스 추출"""
    try:
        raw = ticker_obj.news
    except:
        return []
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict) and 'news' in raw:
        return raw['news']
    return []


def _parse_news_item(item):
    """뉴스 아이템 파싱"""
    title = ""
    publisher = ""
    link = ""
    pub_time = None

    if isinstance(item, dict):
        title = item.get('title', item.get('headline', ''))
        publisher = item.get('publisher', item.get('source', ''))
        link = item.get('link', item.get('url', ''))

        # 발행 시간
        if 'providerPublishTime' in item:
            try:
                pub_time = datetime.fromtimestamp(item['providerPublishTime'])
            except:
                pass

        # content 구조 처리
        if not title and 'content' in item:
            content = item['content']
            if isinstance(content, dict):
                title = content.get('title', '')
                link = content.get('canonicalUrl', {}).get('url', '') if isinstance(content.get('canonicalUrl'), dict) else ''
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else ''

    return title, publisher, link, pub_time


def _analyze_sentiment(title):
    """감성 분석 (확장)"""
    title_lower = title.lower()

    pos_matches = [kw for kw in POSITIVE_KEYWORDS if kw.lower() in title_lower]
    neg_matches = [kw for kw in NEGATIVE_KEYWORDS if kw.lower() in title_lower]

    pos_count = len(pos_matches)
    neg_count = len(neg_matches)

    if pos_count > neg_count:
        sentiment = "호재"
        score = min(1.0, pos_count * 0.25)
        keywords = pos_matches[:3]
    elif neg_count > pos_count:
        sentiment = "악재"
        score = max(-1.0, -neg_count * 0.25)
        keywords = neg_matches[:3]
    else:
        sentiment = "중립"
        score = 0.0
        keywords = []

    return sentiment, score, keywords


def _calculate_impact(title, category, sector):
    """뉴스 영향도 계산 (1-5)"""
    title_lower = title.lower()
    impact = 2  # 기본 영향도

    # 높은 영향도 키워드
    high_impact = ["earnings", "guidance", "fda", "merger", "acquisition",
                   "bankruptcy", "lawsuit", "fed", "interest rate", "tariff",
                   "ceo", "fraud", "recall", "breakthrough"]

    # 중간 영향도 키워드
    medium_impact = ["upgrade", "downgrade", "price target", "analyst",
                     "revenue", "profit", "sales", "growth", "decline"]

    for kw in high_impact:
        if kw in title_lower:
            impact = 5
            break

    if impact < 5:
        for kw in medium_impact:
            if kw in title_lower:
                impact = 4
                break

    # 카테고리별 조정
    if category == "종목":
        impact = min(5, impact + 1)
    elif category == "경쟁사":
        impact = max(1, impact - 1)

    return impact


def _fetch_ticker_news(ticker_symbol, tag, max_items=5):
    """티커별 뉴스 수집"""
    original = _suppress_logs()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker_obj = yf.Ticker(ticker_symbol)
            news_items = _extract_news(ticker_obj)
    except:
        return []
    finally:
        _restore_logs(original)

    analyzed = []
    seen = set()

    for item in news_items[:15]:
        title, publisher, link, pub_time = _parse_news_item(item)
        if not title or title in seen:
            continue
        seen.add(title)

        sentiment, score, keywords = _analyze_sentiment(title)

        analyzed.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "tag": tag,
            "sentiment": sentiment,
            "score": score,
            "keywords": keywords,
            "pub_time": pub_time.strftime("%Y-%m-%d %H:%M") if pub_time else "",
        })

    return analyzed[:max_items]


def _get_stock_info(ticker_symbol):
    """종목 정보 가져오기"""
    original = _suppress_logs()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ticker_obj = yf.Ticker(ticker_symbol)
            info = ticker_obj.info
            return {
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "name": info.get("shortName", info.get("longName", ticker_symbol)),
                "country": info.get("country", ""),
            }
    except:
        return {"sector": "", "industry": "", "name": ticker_symbol, "country": ""}
    finally:
        _restore_logs(original)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_comprehensive_news(ticker_symbol, include_competitors=True,
                              include_macro=True, include_supply_chain=True):
    """
    포괄적 뉴스 수집

    Args:
        ticker_symbol: str
        include_competitors: bool - 경쟁사 뉴스 포함
        include_macro: bool - 거시경제 뉴스 포함
        include_supply_chain: bool - 공급망 뉴스 포함

    Returns:
        dict: 카테고리별 뉴스 + 종합 분석
    """
    all_news = []
    seen_titles = set()
    category_scores = {}

    # 1. 종목 정보 가져오기
    info = _get_stock_info(ticker_symbol)
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    company_name = info.get("name", "")

    # 2. 종목 직접 뉴스
    stock_news = _fetch_ticker_news(ticker_symbol, "종목", max_items=8)
    stock_scores = []
    for n in stock_news:
        if n["title"] not in seen_titles:
            seen_titles.add(n["title"])
            n["category"] = "종목"
            n["impact"] = _calculate_impact(n["title"], "종목", sector)
            all_news.append(n)
            stock_scores.append(n["score"])
    if stock_scores:
        category_scores["종목"] = sum(stock_scores) / len(stock_scores)

    # 3. 경쟁사 뉴스
    if include_competitors:
        competitors = COMPETITOR_MAP.get(ticker_symbol.upper(), [])[:3]
        comp_scores = []
        for comp in competitors:
            comp_news = _fetch_ticker_news(comp, f"경쟁사:{comp}", max_items=2)
            for n in comp_news:
                if n["title"] not in seen_titles:
                    seen_titles.add(n["title"])
                    n["category"] = "경쟁사"
                    n["impact"] = _calculate_impact(n["title"], "경쟁사", sector)
                    all_news.append(n)
                    comp_scores.append(n["score"])
        if comp_scores:
            category_scores["경쟁사"] = sum(comp_scores) / len(comp_scores)

    # 4. 섹터 ETF 뉴스
    sector_etfs = {
        "Technology": ["XLK", "VGT"],
        "Semiconductors": ["SMH", "SOXX"],
        "Financial Services": ["XLF", "VFH"],
        "Healthcare": ["XLV", "VHT"],
        "Energy": ["XLE", "VDE"],
        "Consumer Cyclical": ["XLY", "VCR"],
        "Communication Services": ["XLC", "VOX"],
        "Industrials": ["XLI", "VIS"],
        "Consumer Defensive": ["XLP", "VDC"],
        "Real Estate": ["XLRE", "VNQ"],
        "Utilities": ["XLU", "VPU"],
        "Basic Materials": ["XLB", "VAW"],
    }.get(sector, [])

    if sector_etfs:
        sector_scores = []
        for etf in sector_etfs[:1]:
            sector_news = _fetch_ticker_news(etf, f"섹터:{sector}", max_items=3)
            for n in sector_news:
                if n["title"] not in seen_titles:
                    seen_titles.add(n["title"])
                    n["category"] = "섹터"
                    n["impact"] = _calculate_impact(n["title"], "섹터", sector)
                    all_news.append(n)
                    sector_scores.append(n["score"])
        if sector_scores:
            category_scores["섹터"] = sum(sector_scores) / len(sector_scores)

    # 5. 거시경제 뉴스
    if include_macro:
        macro_scores = []
        macro_list = ["SPY", "TLT", "GLD"]
        for macro_ticker in macro_list:
            label = MACRO_TICKERS.get(macro_ticker, macro_ticker)
            macro_news = _fetch_ticker_news(macro_ticker, f"거시:{label}", max_items=2)
            for n in macro_news:
                if n["title"] not in seen_titles:
                    seen_titles.add(n["title"])
                    n["category"] = "거시경제"
                    n["impact"] = _calculate_impact(n["title"], "거시경제", sector)
                    all_news.append(n)
                    macro_scores.append(n["score"])
        if macro_scores:
            category_scores["거시경제"] = sum(macro_scores) / len(macro_scores)

    # 6. 종합 분석
    if all_news:
        # 영향도 가중 평균
        weighted_scores = []
        for n in all_news:
            weight = n.get("impact", 2) / 3  # 영향도를 가중치로
            weighted_scores.append(n["score"] * weight)

        avg_score = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0

        # 감성 요약
        if avg_score > 0.15:
            summary = "강한 호재"
            summary_emoji = "🟢"
        elif avg_score > 0.05:
            summary = "호재"
            summary_emoji = "🟢"
        elif avg_score < -0.15:
            summary = "강한 악재"
            summary_emoji = "🔴"
        elif avg_score < -0.05:
            summary = "악재"
            summary_emoji = "🔴"
        else:
            summary = "중립"
            summary_emoji = "🟡"

        # 주요 이슈 추출
        high_impact_news = sorted(all_news, key=lambda x: x.get("impact", 0), reverse=True)[:3]
        key_issues = [n["title"][:50] + "..." if len(n["title"]) > 50 else n["title"]
                      for n in high_impact_news]
    else:
        avg_score = 0
        summary = "정보 없음"
        summary_emoji = "⚪"
        key_issues = []

    # 뉴스 정렬 (영향도 > 최신순)
    all_news.sort(key=lambda x: (x.get("impact", 0), x.get("pub_time", "")), reverse=True)

    return {
        "news_list": all_news,
        "sentiment_score": round(avg_score, 3),
        "summary": summary,
        "summary_emoji": summary_emoji,
        "category_scores": category_scores,
        "key_issues": key_issues,
        "stock_info": info,
        "total_count": len(all_news),
        "industry_keywords": INDUSTRY_KEYWORDS.get(sector, [])[:5],
        "policy_keywords": POLICY_KEYWORDS.get(sector, [])[:3],
    }


def get_market_pulse():
    """
    시장 전반 뉴스 펄스 (메인 대시보드용)

    Returns:
        dict: 시장 전반 뉴스 요약
    """
    pulse_news = []
    seen = set()

    # 주요 지수 뉴스
    key_tickers = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]

    for ticker in key_tickers:
        label = MACRO_TICKERS.get(ticker.replace("^", ""), ticker)
        news = _fetch_ticker_news(ticker, label, max_items=2)
        for n in news:
            if n["title"] not in seen:
                seen.add(n["title"])
                n["category"] = "시장"
                pulse_news.append(n)

    # 감성 점수
    if pulse_news:
        scores = [n["score"] for n in pulse_news]
        avg_score = sum(scores) / len(scores)

        if avg_score > 0.1:
            mood = "Risk-On (낙관)"
            mood_emoji = "🟢"
        elif avg_score < -0.1:
            mood = "Risk-Off (비관)"
            mood_emoji = "🔴"
        else:
            mood = "중립"
            mood_emoji = "🟡"
    else:
        avg_score = 0
        mood = "정보 없음"
        mood_emoji = "⚪"

    return {
        "news": pulse_news[:10],
        "market_mood": mood,
        "mood_emoji": mood_emoji,
        "sentiment_score": round(avg_score, 3),
    }


def analyze_news_impact_on_portfolio(tickers, predictions):
    """
    포트폴리오 전체에 대한 뉴스 영향 분석

    Args:
        tickers: list
        predictions: dict

    Returns:
        dict: 포트폴리오 뉴스 영향 분석
    """
    portfolio_news = []
    sector_sentiments = {}

    for ticker in tickers:
        if ticker == "__meta__":
            continue

        pred = predictions.get(ticker, {})
        news_data = pred.get("news", {})

        if news_data:
            news_list = news_data.get("news_list", [])
            sentiment = news_data.get("sentiment_score", 0)

            # 섹터별 감성 집계
            info = _get_stock_info(ticker)
            sector = info.get("sector", "기타")
            if sector not in sector_sentiments:
                sector_sentiments[sector] = []
            sector_sentiments[sector].append(sentiment)

            # 영향도 높은 뉴스 수집
            for n in news_list:
                if n.get("impact", 0) >= 4:
                    n["ticker"] = ticker
                    portfolio_news.append(n)

    # 섹터별 평균
    sector_summary = {}
    for sector, scores in sector_sentiments.items():
        avg = sum(scores) / len(scores) if scores else 0
        if avg > 0.1:
            status = "호재"
        elif avg < -0.1:
            status = "악재"
        else:
            status = "중립"
        sector_summary[sector] = {"score": round(avg, 3), "status": status}

    # 영향도순 정렬
    portfolio_news.sort(key=lambda x: x.get("impact", 0), reverse=True)

    return {
        "high_impact_news": portfolio_news[:10],
        "sector_sentiment": sector_summary,
        "total_high_impact": len(portfolio_news),
    }
