import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback
from backtest import run_backtest, predict_future, get_esg_scores, optimize_layer_weights
from rebalance_engine import calculate_optimal_weights, generate_rebalance_orders, calculate_expected_performance
from news_engine import get_market_pulse, analyze_news_impact_on_portfolio
from auth import check_authentication, show_user_info_sidebar
from settings_manager import get_settings_manager
from stock_utils import parse_tickers_input, format_ticker_display, get_ticker_name

st.set_page_config(page_title="ESG Portfolio Backtest", layout="wide")

# 인증 체크 (로그인 안 된 경우 여기서 중단)
if not check_authentication():
    st.stop()

# 사이드바에 사용자 정보 표시
show_user_info_sidebar()

st.title("ESG 포트폴리오 백테스트 대시보드")

# 사용자별 설정 관리자
settings_manager = get_settings_manager(st.session_state.current_user)
settings = settings_manager.load_settings()

def save_settings(data):
    """설정 저장 (settings_manager 사용)"""
    settings_manager.save_settings(data)

if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "backtest_tickers" not in st.session_state:
    st.session_state.backtest_tickers = []
if "backtest_capital" not in st.session_state:
    st.session_state.backtest_capital = 100000.0
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "pred_tickers" not in st.session_state:
    st.session_state.pred_tickers = []

st.sidebar.header("설정")
tickers_input = st.sidebar.text_input(
    "종목 (쉼표 구분)",
    settings["tickers_input"],
    help="종목명(삼성전자) 또는 코드(005930.KS) 입력 가능"
)

# 종목 파싱 (종목명 → 코드 자동 변환)
ticker_info_list = parse_tickers_input(tickers_input)
tickers = [t["code"] for t in ticker_info_list]
ticker_names = {t["code"]: t["name"] for t in ticker_info_list}

# 세션에 종목명 매핑 저장 (다른 곳에서 사용)
if "ticker_names" not in st.session_state:
    st.session_state.ticker_names = {}
st.session_state.ticker_names.update(ticker_names)

# 인식된 종목 표시
if ticker_info_list:
    display_tickers = [format_ticker_display(t["code"]) for t in ticker_info_list]
    st.sidebar.caption(f"인식: {', '.join(display_tickers)}")

col_date1, col_date2 = st.sidebar.columns(2)
start_date = col_date1.date_input("시작일", pd.to_datetime(settings["start_date"]))
end_date = col_date2.date_input("종료일", pd.to_datetime(settings["end_date"]))
initial_capital = st.sidebar.number_input("초기 자본 (원)", value=settings["initial_capital"], step=10000)

st.sidebar.subheader("분석 레이어 가중치")
st.sidebar.caption("각 분석 레이어의 비중을 조절하세요")

# 세션 상태로 가중치 관리
if "auto_weights" not in st.session_state:
    st.session_state.auto_weights = None

saved_weights = settings.get("layer_weights", {"regime": 20, "ml": 40, "risk": 25, "cross_asset": 15})

# 자동 최적화 결과가 있으면 적용
if st.session_state.auto_weights is not None:
    saved_weights = st.session_state.auto_weights

# 자동 최적화 버튼
if st.sidebar.button("🎯 가중치 자동 최적화", help="종목 데이터 기반 최적 가중치 계산"):
    if len(tickers) > 0:
        with st.sidebar:
            with st.spinner("가중치 최적화 중..."):
                try:
                    opt_result = optimize_layer_weights(tickers, {t: 50 for t in tickers})
                    opt_weights = opt_result["optimal_weights"]
                    st.session_state.auto_weights = opt_weights
                    st.toast(f"✅ 최적화 완료!", icon="🎯")
                    st.sidebar.success(f"레짐:{opt_weights['regime']}% | ML:{opt_weights['ml']}% | 리스크:{opt_weights['risk']}% | 크로스:{opt_weights['cross_asset']}%")
                    st.rerun()
                except Exception as e:
                    st.error(f"최적화 오류: {e}")
    else:
        st.sidebar.warning("종목을 먼저 입력하세요")

w_regime = st.sidebar.slider("레짐 분석", 0, 50, saved_weights.get("regime", 20), help="HMM + Hurst 기반 시장 국면")
w_ml = st.sidebar.slider("ML 예측", 0, 50, saved_weights.get("ml", 40), help="RF+GB 앙상블 예측")
w_risk = st.sidebar.slider("리스크 필터", 0, 50, saved_weights.get("risk", 25), help="VaR/Sharpe 기반 리스크")
w_cross = st.sidebar.slider("크로스 에셋", 0, 30, saved_weights.get("cross_asset", 15), help="VIX/달러/채권 시그널")

# 자동 최적화 상태 초기화 (슬라이더 수동 변경 시)
if st.session_state.auto_weights is not None:
    current = {"regime": w_regime, "ml": w_ml, "risk": w_risk, "cross_asset": w_cross}
    if current != st.session_state.auto_weights:
        st.session_state.auto_weights = None

total_weight = w_regime + w_ml + w_risk + w_cross
if total_weight != 100:
    st.sidebar.warning(f"가중치 합계: {total_weight}% (100%로 맞춰주세요)")

layer_weights = {"regime": w_regime / 100, "ml": w_ml / 100, "risk": w_risk / 100}
cross_asset_weight = w_cross / 100

st.sidebar.divider()
st.sidebar.subheader("ESG 점수 (0~100)")
saved_esg = settings.get("esg_scores", {})

if st.sidebar.button("ESG 자동 조회", width="stretch"):
    with st.sidebar:
        with st.spinner("ESG 점수 조회 중..."):
            auto_scores = get_esg_scores(tickers)
            for t in tickers:
                if t in auto_scores:
                    st.session_state[f"esg_{t}"] = auto_scores[t]
            st.toast("ESG 점수 조회 완료!", icon="📊")
            st.rerun()

esg_scores = {}
for t in tickers:
    if f"esg_{t}" not in st.session_state:
        if t in saved_esg:
            default = int(saved_esg[t])
        else:
            default = 50
        st.session_state[f"esg_{t}"] = default
    display_name = format_ticker_display(t)
    esg_scores[t] = st.sidebar.slider(display_name, 0, 100, key=f"esg_{t}")

rebalance_freq_options = ["M", "Q"]
rebalance_idx = rebalance_freq_options.index(settings["rebalance_freq"]) if settings["rebalance_freq"] in rebalance_freq_options else 0
rebalance_freq = st.sidebar.selectbox("리밸런싱 주기", rebalance_freq_options, index=rebalance_idx,
                                       format_func=lambda x: "월간" if x == "M" else "분기")
tx_cost_pct = st.sidebar.number_input("거래 비용 (%)", value=settings["tx_cost"], step=0.05, format="%.2f")
tx_cost = tx_cost_pct / 100.0

current_settings = {
    "tickers_input": tickers_input,
    "start_date": str(start_date),
    "end_date": str(end_date),
    "initial_capital": int(initial_capital),
    "esg_scores": esg_scores,
    "rebalance_freq": rebalance_freq,
    "tx_cost": tx_cost_pct,
    "layer_weights": {"regime": w_regime, "ml": w_ml, "risk": w_risk, "cross_asset": w_cross},
}
save_settings(current_settings)

if st.sidebar.button("백테스트 실행", type="primary", width="stretch"):
    with st.spinner("백테스트 실행 중..."):
        try:
            result = run_backtest(
                tickers=tickers, start_date=str(start_date), end_date=str(end_date),
                initial_capital=float(initial_capital), esg_scores=esg_scores,
                rebalance_freq=rebalance_freq, transaction_cost_rate=tx_cost,
            )
            st.session_state.backtest_result = result
            st.session_state.backtest_tickers = list(tickers)
            st.session_state.backtest_capital = float(initial_capital)
            st.toast("백테스트 완료!", icon="✅")
        except Exception as e:
            st.error(f"백테스트 오류: {e}")
            st.error(f"\n```\n{traceback.format_exc()}\n```")
            st.session_state.backtest_result = None

if st.session_state.backtest_result is not None:
    result = st.session_state.backtest_result
    bt_tickers = st.session_state.backtest_tickers
    bt_capital = st.session_state.backtest_capital
    st.header("핵심 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("최종 자산", f"₩{result['final_value']:,.0f}",
              delta=f"₩{result['alpha']:+,.0f} vs 벤치마크")
    m2.metric("벤치마크(S&P500)", f"₩{result['final_benchmark']:,.0f}")
    m3.metric("샤프 지수", f"{result['sharpe_ratio']:.3f}")
    m4.metric("최대 낙폭", f"{result['max_drawdown']:.1%}")
    st.divider()
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("포트폴리오 vs 벤치마크")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result["portfolio_series"].index,
            y=result["portfolio_series"].values,
            name="포트폴리오 (ESG+예측)", line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(
            x=result["benchmark_series"].index,
            y=result["benchmark_series"].values,
            name="S&P 500", line=dict(color="#dc2626", width=2, dash="dash")))
        fig.update_layout(yaxis_title="자산 가치 (원)", xaxis_title="날짜",
            legend=dict(orientation="h", y=1.12), height=420, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
    with col_right:
        st.subheader("종목별 비중 변화")
        wdf = result["weights_df"]
        fig2 = go.Figure()
        colors = ["#2563eb", "#16a34a", "#eab308", "#dc2626", "#8b5cf6", "#ec4899"]
        for idx, col in enumerate(wdf.columns):
            fig2.add_trace(go.Scatter(
                x=wdf.index, y=wdf[col], name=col,
                stackgroup="one", line=dict(color=colors[idx % len(colors)])))
        fig2.update_layout(yaxis_title="비중", yaxis_tickformat=".0%",
            height=420, margin=dict(t=30), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, width="stretch")
    with st.expander("상세 수치"):
        detail = pd.DataFrame({
            "항목": ["최종 자산", "벤치마크", "초과수익(Alpha)", "평균 일간 수익률",
                    "일간 변동성", "샤프 지수", "최대 낙폭"],
            "값": [f"₩{result['final_value']:,.2f}", f"₩{result['final_benchmark']:,.2f}",
                   f"₩{result['alpha']:,.2f}", f"{result['mean_return']:.6f}",
                   f"{result['volatility']:.6f}", f"{result['sharpe_ratio']:.3f}",
                   f"{result['max_drawdown']:.3%}"],
        })
        st.table(detail)
else:
    st.info("왼쪽 사이드바에서 설정을 조정한 뒤 **백테스트 실행** 버튼을 눌러주세요.")

st.divider()
st.header("🎯 리밸런싱 가이드")
st.caption("4단계 파이프라인 분석 결과 + 리스크 조정 + 시장 환경을 종합하여 최적 비중과 매수/매도 가이드를 제공합니다.")

# 리스크 허용도 선택
risk_col1, risk_col2 = st.columns([1, 3])
with risk_col1:
    risk_tolerance = st.selectbox(
        "리스크 허용도",
        ["conservative", "moderate", "aggressive"],
        index=1,
        format_func=lambda x: {"conservative": "보수적", "moderate": "중립", "aggressive": "공격적"}[x]
    )
with risk_col2:
    st.caption({
        "conservative": "💚 보수적: 매도 신호에 민감, 약세장에서 비중 대폭 축소, 낮은 변동성 선호",
        "moderate": "💛 중립: 균형 잡힌 접근, 신호에 따른 적절한 비중 조정",
        "aggressive": "🧡 공격적: 매수 신호에 집중, 약세장에서도 비중 유지, 높은 수익 추구"
    }[risk_tolerance])

saved_holdings = settings.get("holdings", {})

# 보유 현황 입력
st.subheader("현재 보유 현황")
hold_cols = st.columns(len(tickers)) if tickers else [st.container()]
current_holdings = {}
for i, t in enumerate(tickers):
    saved_val = float(saved_holdings.get(t, 0.0))
    display_name = format_ticker_display(t)
    current_holdings[t] = hold_cols[i].number_input(
        display_name, value=saved_val, step=10000.0, format="%.0f", key=f"hold_{t}")

total_invest = sum(current_holdings.values())
current_settings["holdings"] = {t: current_holdings[t] for t in tickers}
save_settings(current_settings)

# 요약 메트릭
sum_col1, sum_col2, sum_col3 = st.columns(3)
sum_col1.metric("현재 총 투자금", f"₩{total_invest:,.0f}")

# 현재 비중 계산
if total_invest > 0:
    current_weights = {t: current_holdings[t] / total_invest for t in tickers}
    max_ticker = max(current_weights, key=current_weights.get)
    max_ticker_display = format_ticker_display(max_ticker)
    sum_col2.metric("최대 비중 종목", f"{max_ticker_display} ({current_weights[max_ticker]:.1%})")
else:
    sum_col2.metric("최대 비중 종목", "-")

# 분석 상태
if st.session_state.prediction_result is not None:
    sum_col3.metric("분석 상태", "✅ 분석 완료")
else:
    sum_col3.metric("분석 상태", "⚠️ 분석 필요", "종합 분석을 먼저 실행하세요")

if st.button("📊 리밸런싱 계산", type="primary", key="rebalance_btn"):
    if total_invest <= 0:
        st.warning("보유액을 입력해주세요.")
    elif st.session_state.prediction_result is None:
        st.warning("먼저 **종합 분석 실행** 버튼을 눌러 분석을 완료해주세요.")
    else:
        preds = st.session_state.prediction_result

        # 최적 비중 재계산 (리스크 허용도 반영)
        weight_result = calculate_optimal_weights(
            preds, tickers, esg_scores,
            risk_tolerance=risk_tolerance,
            min_weight=0.05,
            max_weight=0.40
        )
        target_weights = weight_result["weights"]
        weight_details = weight_result["details"]

        # 주문 생성
        orders = generate_rebalance_orders(
            current_holdings, target_weights, total_invest,
            preds, min_trade_pct=0.02, min_trade_amount=30000
        )

        # 예상 성과
        expected = calculate_expected_performance(orders, preds)

        # 시장 환경 표시
        market_env = weight_result.get("market_env_score", 50)
        market_mult = weight_result.get("market_multiplier", 1.0)
        env_status = "🟢 우호적" if market_env > 55 else ("🔴 비우호적" if market_env < 45 else "🟡 중립")

        st.info(f"**시장 환경**: {env_status} (점수: {market_env:.0f}/100, 조정계수: {market_mult:.2f}x)")

        # 리밸런싱 요약
        st.subheader("📋 리밸런싱 요약")

        reb_col1, reb_col2, reb_col3, reb_col4 = st.columns(4)
        reb_col1.metric("매수 종목", f"{expected['buy_count']}개", f"₩{expected['buy_amount']:,.0f}")
        reb_col2.metric("매도 종목", f"{expected['sell_count']}개", f"₩{expected['sell_amount']:,.0f}")
        reb_col3.metric("순 현금흐름", f"₩{expected['net_flow']:,.0f}")
        reb_col4.metric("평균 신뢰도", f"{expected['avg_confidence']:.0f}%")

        # 상세 주문 목록
        st.subheader("📝 상세 주문 가이드")

        for order in orders:
            ticker = order["ticker"]
            ticker_display = format_ticker_display(ticker)
            action = order["action"]
            amount = order["amount"]
            signal = order["signal"]
            reason = order.get("reason", "")
            current_w = order["current_weight"]
            target_w = order["target_weight"]
            action_signal = order.get("action_signal", "")
            confidence = order.get("confidence", 0)
            final_score = order.get("final_score", 50)

            if action == "매수":
                with st.container():
                    st.success(f"""
                    {signal} **{ticker_display}** — **₩{amount:,.0f} 매수**
                    - 비중: {current_w:.1%} → {target_w:.1%} (+{target_w - current_w:.1%})
                    - 모델 신호: {action_signal} | 신뢰도: {confidence}% | 점수: {final_score:.0f}/100
                    - 이유: {reason}
                    """)
            elif action == "매도":
                with st.container():
                    st.error(f"""
                    {signal} **{ticker_display}** — **₩{amount:,.0f} 매도**
                    - 비중: {current_w:.1%} → {target_w:.1%} ({target_w - current_w:.1%})
                    - 모델 신호: {action_signal} | 신뢰도: {confidence}% | 점수: {final_score:.0f}/100
                    - 이유: {reason}
                    """)
            else:
                st.info(f"⚪ **{ticker_display}** — 현재 비중 유지 ({current_w:.1%}, 거래 불필요)")

        # 비중 계산 상세 (접기)
        with st.expander("🔍 비중 계산 상세 (디버그)"):
            detail_data = []
            for t in tickers:
                d = weight_details.get(t, {})
                detail_data.append({
                    "종목": format_ticker_display(t),
                    "최종점수": d.get("final_score", 50),
                    "액션": d.get("action", "-"),
                    "액션배수": d.get("action_mult", 1),
                    "신뢰도배수": d.get("confidence_mult", 1),
                    "리스크배수": d.get("risk_mult", 1),
                    "레짐": d.get("regime", "-"),
                    "레짐배수": d.get("regime_mult", 1),
                    "변동성": d.get("volatility", 0),
                    "변동성배수": d.get("vol_mult", 1),
                    "ESG배수": d.get("esg_mult", 1),
                    "원점수": d.get("raw_score", 0),
                    "최종비중": f"{d.get('final_weight_pct', 0):.1f}%",
                })
            st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

            st.caption("""
            **배수 설명**:
            - 액션배수: 매수유리=1.3, 매수고려=1.1, 관망=0.8, 매도고려=0.6, 매도유리=0.3
            - 신뢰도배수: 신뢰도 0%=0.5, 100%=1.0
            - 리스크배수: 포지션사이징 0%=0.5, 100%=1.0
            - 레짐배수: 약세장=0.7~0.85, 중립=1.0, 강세장=1.15
            - 변동성배수: 연변동성 30% 기준, 높으면 축소
            - 시장배수: Risk-Off=0.7, 중립=1.0, Risk-On=1.1
            """)

        st.warning("⚠️ 이 가이드는 퀀트 모델 기반 참고 자료입니다. 실제 투자 판단은 본인의 책임이며, 미래 수익을 보장하지 않습니다.")

st.divider()
st.header("종합 투자 분석")
st.caption("4단계 다중 레이어 파이프라인: 시장 레짐(HMM) → ML 예측(앙상블) → 리스크 필터링(VaR) → 크로스 에셋 시그널 + 뉴스 감성분석")

if st.button("종합 분석 실행", type="primary", width="stretch"):
    with st.spinner("4단계 파이프라인 분석 중... (레짐→ML→리스크→크로스에셋)"):
        try:
            predictions = predict_future(
                tickers, esg_scores,
                layer_weights=layer_weights,
                cross_asset_weight=cross_asset_weight
            )
            st.session_state.prediction_result = predictions
            st.session_state.pred_tickers = list(tickers)
            st.toast("종합 분석 완료!", icon="✅")
        except Exception as e:
            st.error(f"분석 오류: {e}")
            st.session_state.prediction_result = None

if st.session_state.prediction_result is not None:
    predictions = st.session_state.prediction_result
    pred_tickers = [t for t in st.session_state.pred_tickers if t != "__meta__"]

    # 고정 컬럼 수 사용 (DOM 충돌 방지)
    num_cols = min(len(pred_tickers), 6)
    pred_cols = st.columns(max(num_cols, 1))
    for i, t in enumerate(pred_tickers[:6]):  # 최대 6개
        if t not in predictions:
            continue
        pred = predictions[t]
        with pred_cols[i % num_cols]:
            action = pred.get("action", "판단 불가")
            if action in ["매수 유리", "매수 고려"]:
                action_icon = "🟢"
            elif action in ["매도 유리", "매도 고려"]:
                action_icon = "🔴"
            else:
                action_icon = "🟡"
            st.subheader(format_ticker_display(t))
            st.markdown(f"### {action_icon} {action}")
            st.metric("예측 방향", pred["direction"])
            st.metric("신뢰도", f"{pred['confidence']}%")
            st.metric("예측 수익률", f"{pred['predicted_return']:+.2%}")
            st.metric("추천 비중", f"{pred['recommended_weight']:.1%}")
            news = pred.get("news", {})
            ns = news.get("summary", "없음")
            if ns == "호재":
                st.success(f"뉴스 분위기: {ns}")
            elif ns == "악재":
                st.error(f"뉴스 분위기: {ns}")
            else:
                st.info(f"뉴스 분위기: {ns}")

    # 4단계 다중 레이어 분석 상세
    st.divider()
    st.subheader("다중 레이어 분석 (레짐 → ML → 리스크)")

    for t in pred_tickers:
        if t not in predictions:
            continue
        pred = predictions[t]
        layers = pred.get("layers", {})
        if not layers:
            continue

        with st.expander(f"🔎 {format_ticker_display(t)} — 3단계 파이프라인 상세", expanded=(t == pred_tickers[0])):
            layer_tab1, layer_tab2, layer_tab3, layer_tab4 = st.tabs([
                "📊 Stage 1: 시장 레짐", "🤖 Stage 2: ML 예측", "🛡️ Stage 3: 리스크", "🌐 Stage 4: 크로스 에셋"
            ])

            with layer_tab1:
                regime = layers.get("regime", {})
                r_col1, r_col2, r_col3 = st.columns(3)
                regime_icon = {0: "🔴", 1: "🟡", 2: "🟢"}.get(regime.get("regime", 1), "🟡")
                r_col1.metric("현재 레짐", f"{regime_icon} {regime.get('name', '중립')}")
                r_col2.metric("레짐 점수", f"{regime.get('score', 50):.1f}/100")
                r_col3.metric("레짐 신뢰도", f"{regime.get('confidence', 0):.0%}")
                h_col1, h_col2 = st.columns(2)
                h_col1.metric("Hurst 지수", f"{regime.get('hurst', 0.5):.4f}")
                h_col2.metric("해석", regime.get("hurst_interp", ""))
                st.info(f"💡 전략 힌트: {regime.get('strategy_hint', '')}")
                st.caption(f"분석 방법: {regime.get('method', '')}")

            with layer_tab2:
                ml = layers.get("ml", {})
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("ML 예측 방향", ml.get("direction", "중립"))
                m_col2.metric("ML 점수", f"{ml.get('score', 50):.1f}/100")
                m_col3.metric("모델 정확도", f"{ml.get('accuracy', 0):.1%}")
                p_col1, p_col2, p_col3 = st.columns(3)
                prob_up = ml.get("prob_up", 0)
                prob_neutral = ml.get("prob_neutral", 0)
                prob_down = ml.get("prob_down", 0)
                p_col1.metric("상승 확률", f"{prob_up:.1%}")
                p_col2.metric("중립 확률", f"{prob_neutral:.1%}")
                p_col3.metric("하락 확률", f"{prob_down:.1%}")
                fig_prob = go.Figure(data=[go.Bar(
                    x=["하락", "중립", "상승"],
                    y=[prob_down * 100, prob_neutral * 100, prob_up * 100],
                    marker_color=["#dc2626", "#eab308", "#16a34a"],
                    text=[f"{prob_down:.1%}", f"{prob_neutral:.1%}", f"{prob_up:.1%}"],
                    textposition="auto",
                )])
                fig_prob.update_layout(height=250, yaxis_title="확률 (%)", margin=dict(t=10, b=30))
                st.plotly_chart(fig_prob, width="stretch")
                top_features = ml.get("top_features", {})
                if top_features:
                    st.caption("주요 예측 피처:")
                    feat_items = [f"`{k}`: {v:.4f}" for k, v in top_features.items()]
                    st.markdown(" | ".join(feat_items))
                st.caption(f"분석 방법: {ml.get('method', '')}")

            with layer_tab3:
                risk = layers.get("risk", {})
                metrics = risk.get("metrics", {})
                position = risk.get("position", {})
                rk_col1, rk_col2, rk_col3, rk_col4 = st.columns(4)
                rk_col1.metric("리스크 점수", f"{risk.get('score', 50):.1f}/100")
                rk_col2.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
                rk_col3.metric("Sortino", f"{metrics.get('sortino_ratio', 0):.3f}")
                rk_col4.metric("Calmar", f"{metrics.get('calmar_ratio', 0):.3f}")
                rk2_col1, rk2_col2, rk2_col3, rk2_col4 = st.columns(4)
                rk2_col1.metric("VaR(95%)", f"{metrics.get('var_historical', 0):.2f}%")
                rk2_col2.metric("CVaR", f"{metrics.get('cvar', 0):.2f}%")
                rk2_col3.metric("최대낙폭", f"{metrics.get('max_drawdown', 0):.1f}%")
                rk2_col4.metric("연간변동성", f"{metrics.get('annual_volatility', 0):.1f}%")
                st.markdown("---")
                pos_col1, pos_col2, pos_col3 = st.columns(3)
                pos_col1.metric("추천 포지션", f"{position.get('position_pct', 0):.1f}%")
                pos_col2.metric("리스크 기반", f"{position.get('risk_based', 0):.1f}%")
                pos_col3.metric("변동성 기반", f"{position.get('vol_based', 0):.1f}%")

            with layer_tab4:
                cross = layers.get("cross_asset", {})
                ca_col1, ca_col2, ca_col3, ca_col4 = st.columns(4)
                vix_data = cross.get("vix", {})
                dollar_data = cross.get("dollar", {})
                bond_data = cross.get("bond_equity", {})
                gold_data = cross.get("gold", {})

                ca_col1.metric(
                    "VIX 시그널",
                    vix_data.get("signal", "N/A"),
                    vix_data.get("interpretation", "")
                )
                ca_col2.metric(
                    "달러 시그널",
                    dollar_data.get("signal", "N/A"),
                    dollar_data.get("interpretation", "")
                )
                ca_col3.metric(
                    "채권/주식",
                    bond_data.get("risk_mode", "N/A"),
                    bond_data.get("interpretation", "")
                )
                ca_col4.metric(
                    "금 시그널",
                    gold_data.get("signal", "N/A"),
                    gold_data.get("interpretation", "")
                )

                st.markdown("---")
                ca2_col1, ca2_col2, ca2_col3, ca2_col4 = st.columns(4)
                ca2_col1.metric("VIX 점수", f"{vix_data.get('score', 50):.0f}/100")
                ca2_col2.metric("달러 점수", f"{dollar_data.get('score', 50):.0f}/100")
                ca2_col3.metric("채권/주식 점수", f"{bond_data.get('score', 50):.0f}/100")
                ca2_col4.metric("금 점수", f"{gold_data.get('score', 50):.0f}/100")

                overall = cross.get("overall_signal", "NEUTRAL")
                overall_color = "green" if overall == "FAVORABLE" else ("red" if overall == "UNFAVORABLE" else "gray")
                st.markdown(f"**크로스 에셋 종합: {cross.get('score', 50):.1f}/100** — :{overall_color}[{overall}]")

    # 크로스 에셋 종합 (메타 정보에서)
    meta = predictions.get("__meta__", {})
    cross_asset_meta = meta.get("cross_asset", {})

    if cross_asset_meta:
        st.divider()
        st.subheader("🌐 크로스 에셋 시장 환경")
        st.caption("VIX, 달러, 채권/주식, 금 시장 시그널 종합")

        env_col1, env_col2, env_col3, env_col4, env_col5 = st.columns(5)

        vix_m = cross_asset_meta.get("vix", {})
        dollar_m = cross_asset_meta.get("dollar", {})
        bond_m = cross_asset_meta.get("bond_equity", {})
        gold_m = cross_asset_meta.get("gold", {})

        with env_col1:
            vix_val = vix_m.get("value", 0)
            vix_color = "🔴" if vix_val > 25 else ("🟡" if vix_val > 18 else "🟢")
            st.metric(f"{vix_color} VIX", f"{vix_val:.1f}", vix_m.get("signal", ""))

        with env_col2:
            dollar_chg = dollar_m.get("change_20d", 0)
            dollar_color = "🔴" if dollar_chg > 2 else ("🟢" if dollar_chg < -2 else "⚪")
            st.metric(f"{dollar_color} 달러(UUP)", f"{dollar_chg:+.1f}%", dollar_m.get("signal", ""))

        with env_col3:
            risk_mode = bond_m.get("risk_mode", "중립")
            risk_color = "🟢" if risk_mode == "Risk-On" else ("🔴" if risk_mode == "Risk-Off" else "⚪")
            st.metric(f"{risk_color} 리스크 모드", risk_mode)

        with env_col4:
            gold_chg = gold_m.get("change_20d", 0)
            gold_color = "🟡" if gold_chg > 3 else ("⚪" if gold_chg > -2 else "🟢")
            st.metric(f"{gold_color} 금(GLD)", f"{gold_chg:+.1f}%", gold_m.get("signal", ""))

        with env_col5:
            total_cross = cross_asset_meta.get("cross_asset_score", 50)
            overall = cross_asset_meta.get("overall_signal", "NEUTRAL")
            overall_kr = {"FAVORABLE": "우호적", "UNFAVORABLE": "비우호적", "NEUTRAL": "중립"}.get(overall, "중립")
            st.metric("종합 환경", f"{total_cross:.0f}/100", overall_kr)

    # 페어 트레이딩 기회
    pair_opps = meta.get("pair_opportunities", [])
    active_pairs = [p for p in pair_opps if p.get("signal") != "NO_SIGNAL"]

    if active_pairs:
        st.divider()
        st.subheader("🔗 페어 트레이딩 기회")
        st.caption("고상관 종목 간 Z-score 기반 차익거래 시그널")

        pair_df = pd.DataFrame(active_pairs)[["pair", "correlation", "z_score", "interpretation"]]
        pair_df.columns = ["페어", "상관계수", "Z-Score", "시그널"]
        st.dataframe(pair_df, use_container_width=True, hide_index=True)

    # 포괄적 뉴스 분석 섹션
    st.divider()
    st.subheader("📰 능동적 뉴스 분석")
    st.caption("종목 직접 뉴스 + 경쟁사 + 섹터 + 거시경제 뉴스를 종합 분석")

    # 뉴스 카테고리별 탭
    news_tab1, news_tab2, news_tab3 = st.tabs(["🎯 핵심 뉴스", "📊 카테고리별", "🌍 시장 펄스"])

    with news_tab1:
        # 영향도 높은 뉴스 우선 표시
        for t in pred_tickers:
            if t not in predictions:
                continue
            pred = predictions[t]
            news = pred.get("news", {})
            news_list = news.get("news_list", [])
            summary = news.get("summary", "중립")
            summary_emoji = news.get("summary_emoji", "🟡")
            key_issues = news.get("key_issues", [])
            stock_info = news.get("stock_info", {})

            with st.expander(f"{summary_emoji} **{format_ticker_display(t)}** — {summary} ({len(news_list)}건)", expanded=True):
                # 종목 정보
                sector = stock_info.get("sector", "")
                industry = stock_info.get("industry", "")
                if sector or industry:
                    st.caption(f"섹터: {sector} | 산업: {industry}")

                # 카테고리별 감성 점수
                cat_scores = news.get("category_scores", {})
                if cat_scores:
                    score_cols = st.columns(len(cat_scores))
                    for i, (cat, score) in enumerate(cat_scores.items()):
                        with score_cols[i]:
                            if score > 0.05:
                                delta_color = "normal"
                            elif score < -0.05:
                                delta_color = "inverse"
                            else:
                                delta_color = "off"
                            st.metric(cat, f"{score:+.2f}", delta_color=delta_color)

                # 주요 이슈
                if key_issues:
                    st.markdown("**주요 이슈:**")
                    for issue in key_issues[:3]:
                        st.markdown(f"- {issue}")

                # 뉴스 목록 (영향도순)
                st.markdown("**뉴스 목록:**")
                for article in news_list[:8]:
                    sentiment = article.get("sentiment", "중립")
                    impact = article.get("impact", 2)
                    category = article.get("category", "")
                    tag = article.get("tag", "종목")
                    pub_time = article.get("pub_time", "")

                    # 감성 이모지
                    if sentiment == "호재":
                        emoji = "🟢"
                    elif sentiment == "악재":
                        emoji = "🔴"
                    else:
                        emoji = "⚪"

                    # 영향도 표시
                    impact_stars = "⭐" * min(impact, 5)

                    title = article.get("title", "")
                    publisher = article.get("publisher", "")
                    link = article.get("link", "")

                    # 카테고리 색상
                    cat_colors = {
                        "종목": "blue", "경쟁사": "orange",
                        "섹터": "green", "거시경제": "violet"
                    }
                    cat_color = cat_colors.get(category, "gray")
                    cat_badge = f":{cat_color}[{tag}]"

                    if link:
                        st.markdown(f"{emoji} {impact_stars} {cat_badge} [{title}]({link})")
                    else:
                        st.markdown(f"{emoji} {impact_stars} {cat_badge} {title}")

                    if pub_time:
                        st.caption(f"  └ {publisher} | {pub_time}")

    with news_tab2:
        # 카테고리별 뉴스 분류
        all_news_by_cat = {"종목": [], "경쟁사": [], "섹터": [], "거시경제": []}

        for t in pred_tickers:
            if t not in predictions:
                continue
            pred = predictions[t]
            news = pred.get("news", {})
            news_list = news.get("news_list", [])
            for n in news_list:
                cat = n.get("category", "종목")
                n["source_ticker"] = t
                if cat in all_news_by_cat:
                    all_news_by_cat[cat].append(n)

        for cat, news_list in all_news_by_cat.items():
            if news_list:
                # 중복 제거
                seen = set()
                unique_news = []
                for n in news_list:
                    if n["title"] not in seen:
                        seen.add(n["title"])
                        unique_news.append(n)

                cat_emoji = {"종목": "🎯", "경쟁사": "🏢", "섹터": "📊", "거시경제": "🌍"}
                st.markdown(f"### {cat_emoji.get(cat, '📰')} {cat} 뉴스 ({len(unique_news)}건)")

                for article in unique_news[:5]:
                    sentiment = article.get("sentiment", "중립")
                    emoji = "🟢" if sentiment == "호재" else ("🔴" if sentiment == "악재" else "⚪")
                    impact = article.get("impact", 2)
                    title = article.get("title", "")
                    link = article.get("link", "")
                    source_ticker = article.get("source_ticker", "")

                    if link:
                        st.markdown(f"{emoji} {'⭐' * impact} [{title}]({link}) `{source_ticker}`")
                    else:
                        st.markdown(f"{emoji} {'⭐' * impact} {title} `{source_ticker}`")

    with news_tab3:
        # 시장 전반 펄스
        try:
            from news_engine import get_market_pulse
            pulse = get_market_pulse()

            mood = pulse.get("market_mood", "중립")
            mood_emoji = pulse.get("mood_emoji", "🟡")
            pulse_score = pulse.get("sentiment_score", 0)

            st.markdown(f"### {mood_emoji} 시장 분위기: {mood}")
            st.metric("시장 감성 지수", f"{pulse_score:+.3f}")

            pulse_news = pulse.get("news", [])
            if pulse_news:
                st.markdown("**주요 시장 뉴스:**")
                for article in pulse_news[:8]:
                    sentiment = article.get("sentiment", "중립")
                    emoji = "🟢" if sentiment == "호재" else ("🔴" if sentiment == "악재" else "⚪")
                    tag = article.get("tag", "")
                    title = article.get("title", "")
                    link = article.get("link", "")

                    if link:
                        st.markdown(f"{emoji} `{tag}` [{title}]({link})")
                    else:
                        st.markdown(f"{emoji} `{tag}` {title}")
        except Exception as e:
            st.warning(f"시장 펄스 로딩 중: {e}")

    with st.expander("전체 분석 지표 상세"):
        signal_data = []
        for t in pred_tickers:
            if t not in predictions:
                continue
            pred = predictions[t]
            row = {"종목": format_ticker_display(t), "종합 판단": pred.get("action", "")}
            row.update(pred["signals"])
            signal_data.append(row)
        st.table(pd.DataFrame(signal_data))

    st.subheader("종합 분석 기반 추천 포트폴리오 비중")
    valid_tickers = [t for t in pred_tickers if t in predictions and t != "__meta__"]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_pred = go.Figure(data=[go.Pie(
            labels=[format_ticker_display(t) for t in valid_tickers],
            values=[predictions[t]["recommended_weight"] for t in valid_tickers],
            hole=0.4,
            marker_colors=["#2563eb", "#16a34a", "#eab308", "#dc2626", "#8b5cf6", "#ec4899"][:len(valid_tickers)],
        )])
        fig_pred.update_layout(height=350, margin=dict(t=20, b=20), title="추천 비중")
        st.plotly_chart(fig_pred, width="stretch")

    with chart_col2:
        # 레이더 차트: 첫 번째 종목의 4단계 점수
        if valid_tickers:
            first_ticker = valid_tickers[0]
            layers = predictions[first_ticker].get("layers", {})
            categories = ["레짐", "ML", "리스크", "크로스에셋"]
            values = [
                layers.get("regime", {}).get("score", 50),
                layers.get("ml", {}).get("score", 50),
                layers.get("risk", {}).get("score", 50),
                layers.get("cross_asset", {}).get("score", 50),
            ]
            values.append(values[0])  # 닫기
            categories.append(categories[0])

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line_color='#2563eb',
                fillcolor='rgba(37, 99, 235, 0.3)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=350,
                margin=dict(t=30, b=20),
                title=f"{format_ticker_display(first_ticker)} 4단계 점수"
            )
            st.plotly_chart(fig_radar, width="stretch")

    # 종목간 상관관계 히트맵
    if len(valid_tickers) >= 2:
        st.subheader("📊 종목간 상관관계")
        try:
            import yfinance as yf
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                price_df = yf.download(valid_tickers, period="6mo", progress=False)["Close"]

            if not price_df.empty and len(price_df.columns) >= 2:
                returns_df = price_df.pct_change().dropna()
                corr_matrix = returns_df.corr()

                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="수익률 상관관계 (6개월)"
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, width="stretch")
        except Exception:
            pass

    st.warning("⚠️ 이 분석은 다중 레이어 퀀트 모델(레짐+ML+리스크+크로스에셋) 기반의 참고 자료입니다. 미래 수익을 보장하지 않으며, 실제 투자 판단은 본인의 책임입니다.")

st.divider()
st.header("🔍 관심 종목 탐색")
st.caption("포트폴리오와 별개로, 매수를 고려 중인 종목을 검색하여 3단계 파이프라인 분석을 받아보세요.")

if "explore_result" not in st.session_state:
    st.session_state.explore_result = None
if "explore_tickers" not in st.session_state:
    st.session_state.explore_tickers = []

explore_col1, explore_col2 = st.columns([3, 1])
with explore_col1:
    explore_input = st.text_input(
        "종목 코드 입력 (쉼표로 구분)",
        placeholder="ex: NVDA, PLTR, 005930.KS",
        key="explore_input"
    )
with explore_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    explore_btn = st.button("종목 분석", type="primary", width="stretch")

if explore_btn and explore_input.strip():
    explore_tickers = [t.strip().upper() for t in explore_input.split(",") if t.strip()]
    if not explore_tickers:
        st.warning("종목 코드를 입력해주세요.")
    else:
        explore_esg = {t: 50 for t in explore_tickers}
        with st.spinner(f"{', '.join(explore_tickers)} 4단계 분석 중..."):
            try:
                explore_predictions = predict_future(
                    explore_tickers, explore_esg,
                    layer_weights=layer_weights,
                    cross_asset_weight=cross_asset_weight
                )
                st.session_state.explore_result = explore_predictions
                st.session_state.explore_tickers = explore_tickers
                st.toast("관심 종목 분석 완료!", icon="🔍")
            except Exception as e:
                st.error(f"분석 오류: {e}")
                st.session_state.explore_result = None

if st.session_state.explore_result is not None:
    ex_predictions = st.session_state.explore_result
    ex_tickers = st.session_state.explore_tickers

    # 고정 4컬럼 사용 (DOM 충돌 방지)
    ex_cols = st.columns(4)
    for i, t in enumerate(ex_tickers[:4]):  # 최대 4개만 표시
        if t not in ex_predictions or t == "__meta__":
            continue
        pred = ex_predictions[t]
        with ex_cols[i]:
            action = pred.get("action", "판단 불가")
            if action in ["매수 유리", "매수 고려"]:
                action_icon = "🟢"
            elif action in ["매도 유리", "매도 고려"]:
                action_icon = "🔴"
            else:
                action_icon = "🟡"
            st.subheader(format_ticker_display(t))
            st.markdown(f"### {action_icon} {action}")
            st.metric("예측 방향", pred["direction"])
            st.metric("신뢰도", f"{pred['confidence']}%")
            st.metric("예측 수익률", f"{pred['predicted_return']:+.2%}")
            news = pred.get("news", {})
            ns = news.get("summary", "없음")
            summary_emoji = news.get("summary_emoji", "🟡")
            news_count = news.get("total_count", 0)
            if ns in ["호재", "강한 호재"]:
                st.success(f"뉴스: {ns} ({news_count}건)")
            elif ns in ["악재", "강한 악재"]:
                st.error(f"뉴스: {ns} ({news_count}건)")
            else:
                st.info(f"뉴스: {ns} ({news_count}건)")

    with st.expander("📰 능동적 뉴스 분석", expanded=False):
        for t in ex_tickers:
            if t not in ex_predictions or t == "__meta__":
                continue
            pred = ex_predictions[t]
            news = pred.get("news", {})
            news_list = news.get("news_list", [])
            key_issues = news.get("key_issues", [])
            cat_scores = news.get("category_scores", {})
            stock_info = news.get("stock_info", {})

            st.markdown(f"### {format_ticker_display(t)}")

            # 종목 정보
            sector = stock_info.get("sector", "")
            industry = stock_info.get("industry", "")
            if sector:
                st.caption(f"섹터: {sector} | 산업: {industry}")

            # 카테고리별 점수
            if cat_scores:
                score_text = " | ".join([f"{cat}: {score:+.2f}" for cat, score in cat_scores.items()])
                st.markdown(f"**카테고리별 감성:** {score_text}")

            # 주요 이슈
            if key_issues:
                st.markdown("**주요 이슈:**")
                for issue in key_issues[:2]:
                    st.markdown(f"- {issue}")

            # 뉴스 목록
            if news_list:
                for article in news_list[:6]:
                    sentiment = article.get("sentiment", "중립")
                    emoji = "🟢" if sentiment == "호재" else ("🔴" if sentiment == "악재" else "⚪")
                    impact = article.get("impact", 2)
                    category = article.get("category", "")
                    tag = article.get("tag", "종목")
                    title = article.get("title", "")
                    link = article.get("link", "")
                    pub_time = article.get("pub_time", "")

                    impact_str = "⭐" * min(impact, 5)
                    if link:
                        st.markdown(f"{emoji} {impact_str} `{tag}` [{title}]({link})")
                    else:
                        st.markdown(f"{emoji} {impact_str} `{tag}` {title}")
            else:
                st.markdown("관련 뉴스 없음")
            st.markdown("---")

    with st.expander("📊 기술적 지표 상세", expanded=False):
        signal_data = []
        for t in ex_tickers:
            if t not in ex_predictions or t == "__meta__":
                continue
            pred = ex_predictions[t]
            row = {"종목": format_ticker_display(t), "종합 판단": pred.get("action", "")}
            row.update(pred["signals"])
            signal_data.append(row)
        if signal_data:
            st.dataframe(pd.DataFrame(signal_data), use_container_width=True, hide_index=True)

    st.warning("⚠️ 이 분석은 참고용입니다. 실제 투자 판단은 본인의 책임입니다.")
