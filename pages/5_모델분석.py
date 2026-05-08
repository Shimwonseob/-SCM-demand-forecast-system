import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_export_data, load_import_data

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

st.set_page_config(page_title="모델분析", layout="wide", page_icon="🔬")
st.title("🧠 모델 분석")

# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
BACKTEST_TRAIN_END = "2025-02-01"
BACKTEST_START = "2025-03-01"
BACKTEST_END = "2026-02-01"
N_BACKTEST = 12

MODEL_COLORS = {
    "SARIMA": "royalblue",
    "Prophet": "mediumseagreen",
    "XGBoost": "darkorange",
    "앙상블": "crimson",
}
MODEL_ORDER = ["SARIMA", "Prophet", "XGBoost", "앙상블"]

_SARIMA_KW = dict(
    start_p=1, max_p=3, start_q=1, max_q=3, d=1,
    start_P=0, max_P=2, start_Q=0, max_Q=2, D=1,
    seasonal=True, m=12, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    information_criterion="aic",
)

_FEAT = [
    "month", "year",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_3", "rolling_6", "rolling_12",
]


def _xgb_features(series: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"y": series})
    d["month"] = d.index.month
    d["year"] = d.index.year
    for lag in [1, 2, 3, 6, 12]:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in [3, 6, 12]:
        d[f"rolling_{w}"] = d["y"].shift(1).rolling(w).mean()
    return d.dropna()


def _xgb_predict(
    model, history: np.ndarray, n: int, s_month: int, s_year: int
) -> np.ndarray:
    buf = list(history[-12:])
    preds, month, year = [], s_month, s_year
    for _ in range(n):
        row = {
            "month": month, "year": year,
            "lag_1": buf[-1], "lag_2": buf[-2], "lag_3": buf[-3],
            "lag_6": buf[-6], "lag_12": buf[-12],
            "rolling_3": float(np.mean(buf[-3:])),
            "rolling_6": float(np.mean(buf[-6:])),
            "rolling_12": float(np.mean(buf[-12:])),
        }
        p = float(model.predict(pd.DataFrame([row])[_FEAT])[0])
        preds.append(p)
        buf.append(p)
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return np.array(preds)


def _chart_layout(title: str, height: int = 500, **extra) -> dict:
    base = dict(
        title=title, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        hovermode="x unified",
    )
    base.update(extra)
    return base


@st.cache_resource
def run_backtest(target: str) -> dict:
    """백테스트: 2018-01 ~ 2025-02 학습, 2025-03 ~ 2026-02 예측."""
    from pmdarima import auto_arima
    from prophet import Prophet
    from xgboost import XGBRegressor

    df = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    train = df.loc[(df.index >= TRAIN_START) & (df.index <= BACKTEST_TRAIN_END), col]
    actual = df.loc[(df.index >= BACKTEST_START) & (df.index <= BACKTEST_END), col]
    n = len(actual)

    preds: dict[str, np.ndarray | None] = {}
    errors: dict[str, str] = {}
    params: dict[str, dict] = {}

    # SARIMA
    try:
        m = auto_arima(train, **_SARIMA_KW)
        fc, _ = m.predict(n_periods=n, return_conf_int=True, alpha=0.05)
        preds["SARIMA"] = np.array(fc)
        params["SARIMA"] = {"order": m.order, "seasonal_order": m.seasonal_order}
    except Exception as e:
        preds["SARIMA"] = None
        errors["SARIMA"] = str(e)

    # Prophet
    try:
        mp = Prophet(
            yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False, changepoint_prior_scale=0.1, interval_width=0.95,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
        future = mp.make_future_dataframe(periods=n, freq="MS")
        preds["Prophet"] = mp.predict(future).tail(n)["yhat"].values
        params["Prophet"] = {"changepoint_prior_scale": 0.1, "yearly_seasonality": True}
    except Exception as e:
        preds["Prophet"] = None
        errors["Prophet"] = str(e)

    # XGBoost
    try:
        ft = _xgb_features(train)
        mx = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            verbosity=0, random_state=42,
        )
        mx.fit(ft[_FEAT], ft["y"])
        s_month = actual.index[0].month if n > 0 else 3
        s_year = actual.index[0].year if n > 0 else 2025
        preds["XGBoost"] = _xgb_predict(mx, train.values, n, s_month, s_year)
        params["XGBoost"] = {
            "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "features": _FEAT,
        }
    except Exception as e:
        preds["XGBoost"] = None
        errors["XGBoost"] = str(e)

    # 앙상블 (SARIMA 40% / Prophet 40% / XGBoost 20%)
    target_weights = {"SARIMA": 0.4, "Prophet": 0.4, "XGBoost": 0.2}
    avail = [(target_weights[nm], nm)
             for nm in ["SARIMA", "Prophet", "XGBoost"]
             if preds.get(nm) is not None]
    if avail:
        tot = sum(w for w, _ in avail)
        preds["앙상블"] = sum(w * preds[nm] for w, nm in avail) / tot
        params["앙상블"] = {
            "target_weights": target_weights,
            "active_models": [nm for _, nm in avail],
        }
    else:
        preds["앙상블"] = None

    return {"actual": actual, "preds": preds, "errors": errors, "params": params}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("분석 설정")
    target = st.radio("분석 대상", ["수입", "수출"])

# ── 백테스트 실행 ──────────────────────────────────────────────────────────────
with st.spinner("백테스트 실행 중... (약 10~20초 소요)"):
    bt = run_backtest(target)

actual: pd.Series = bt["actual"]
preds: dict = bt["preds"]
errors: dict = bt["errors"]
params: dict = bt["params"]

for name, msg in errors.items():
    st.warning(f"⚠️ {name} 학습 오류 (스킵됨): {msg}")

if len(actual) == 0:
    st.error("백테스트 기간(2025-03 ~ 2026-02)의 실제 데이터가 없습니다.")
    st.stop()

available = [nm for nm in MODEL_ORDER if preds.get(nm) is not None]

# ── 섹션 1: 백테스트 결과 차트 ────────────────────────────────────────────────
st.subheader("📈 섹션 1: 백테스트 결과 차트")

fig1 = go.Figure()

# 실제값 — 흰색 굵은 실선
fig1.add_trace(go.Scatter(
    x=actual.index, y=actual.values,
    mode="lines", name="실제값",
    line=dict(color="white", width=3),
))

# 모델 예측선 — 색상별 점선 + 마지막 포인트에 MAPE 레이블
for name in available:
    p = preds[name]
    mape_val = float(np.mean(np.abs((actual.values - p) / actual.values)) * 100)
    texts = [""] * (len(p) - 1) + [f" {mape_val:.1f}%"]
    fig1.add_trace(go.Scatter(
        x=actual.index, y=p,
        mode="lines+text", name=name,
        text=texts, textposition="middle right",
        textfont=dict(color=MODEL_COLORS[name], size=11),
        line=dict(color=MODEL_COLORS[name], width=2, dash="dash"),
    ))

fig1.update_layout(
    **_chart_layout(
        "백테스트: 모델별 예측 vs 실제값 비교 (2025-03 ~ 2026-02)",
        height=520,
        xaxis_title="날짜", yaxis_title="금액(USD)",
        yaxis=dict(tickformat=",", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
)
st.plotly_chart(fig1, use_container_width=True)

# ── 섹션 2: 오차 분석 ─────────────────────────────────────────────────────────
st.subheader("📊 섹션 2: 오차 분석")

col_l, col_r = st.columns(2)

# 왼쪽: 월별 오차율(%)
with col_l:
    fig_err = go.Figure()
    for name in available:
        err_rate = (preds[name] - actual.values) / actual.values * 100
        fig_err.add_trace(go.Scatter(
            x=actual.index, y=err_rate,
            mode="lines+markers", name=name,
            line=dict(color=MODEL_COLORS[name], width=2),
            marker=dict(size=6),
        ))
    fig_err.add_hline(y=0, line=dict(color="white", dash="dot", width=1.5))
    fig_err.update_layout(
        **_chart_layout(
            "월별 예측 오차율(%)", height=400,
            xaxis_title="날짜", yaxis_title="오차율(%)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", y=-0.25),
        )
    )
    st.plotly_chart(fig_err, use_container_width=True)

# 오른쪽: 잔차 분포 히스토그램
with col_r:
    fig_hist = go.Figure()
    for name in available:
        residuals = actual.values - preds[name]
        fig_hist.add_trace(go.Histogram(
            x=residuals, name=name, opacity=0.6,
            marker_color=MODEL_COLORS[name], nbinsx=8,
        ))
    fig_hist.add_vline(x=0, line=dict(color="white", dash="dash", width=1.5))
    fig_hist.update_layout(
        **_chart_layout(
            "잔차 분포", height=400, barmode="overlay",
            xaxis_title="잔차 (실제값 − 예측값)", yaxis_title="빈도",
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(orientation="h", y=-0.25),
        )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── 섹션 3: 모델 성능 종합 리포트 ────────────────────────────────────────────
st.subheader("📋 섹션 3: 모델 성능 종합 리포트")

rows = []
for name in available:
    p = preds[name]
    res = actual.values - p
    mae_v  = float(np.mean(np.abs(res)))
    mape_v = float(np.mean(np.abs(res / actual.values)) * 100)
    rmse_v = float(np.sqrt(np.mean(res ** 2)))
    max_e  = float(np.max(np.abs(res)))
    min_e  = float(np.min(np.abs(res)))
    bias_v = float(np.mean(res))
    if mape_v < 8:
        evaluation = "우수"
    elif mape_v <= 12:
        evaluation = "양호"
    else:
        evaluation = "개선필요"
    rows.append({
        "모델명": name,
        "MAE": mae_v,
        "MAPE(%)": mape_v,
        "RMSE": rmse_v,
        "최대오차": max_e,
        "최소오차": min_e,
        "편향(Bias)": bias_v,
        "종합평가": evaluation,
    })

if rows:
    perf_df = pd.DataFrame(rows)
    best_idx = int(perf_df["MAPE(%)"].idxmin())

    def _highlight_best(row):
        style = "background-color: rgba(255, 215, 0, 0.25); font-weight: bold"
        return [style if row.name == best_idx else "" for _ in row]

    styled = (
        perf_df.style
        .apply(_highlight_best, axis=1)
        .format({
            "MAE": "{:,.0f}",
            "MAPE(%)": "{:.2f}",
            "RMSE": "{:,.0f}",
            "최대오차": "{:,.0f}",
            "최소오차": "{:,.0f}",
            "편향(Bias)": "{:+,.0f}",
        })
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)
    # ── 모델 선택 가이드 ─────────────────────────────────────────────────────
    mape_dict = {r["모델명"]: r["MAPE(%)"] for r in rows}
    st.markdown("#### 📌 모델 선택 가이드")
    with st.container(border=True):
        _g1, _g2 = st.columns(2)
        with _g1:
            st.markdown(
                f"🎯 **정확도 우선**: XGBoost 추천 "
                f"(MAPE {mape_dict.get('XGBoost', 0):.2f}%)"
            )
            st.markdown(
                f"📖 **해석 가능성 우선**: SARIMA 추천 "
                f"(파라미터 명시적, MAPE {mape_dict.get('SARIMA', 0):.2f}%)"
            )
        with _g2:
            st.markdown(
                f"🔄 **변화점 많은 데이터**: Prophet 추천 "
                f"(변화점 자동 감지, MAPE {mape_dict.get('Prophet', 0):.2f}%)"
            )
            st.markdown(
                f"🛡️ **안정적 예측 우선**: 앙상블 추천 "
                f"(단일 모델 리스크 분산, MAPE {mape_dict.get('앙상블', 0):.2f}%)"
            )

# ── 섹션 4: 모델 파라미터 정보 ───────────────────────────────────────────────
st.subheader("⚙️ 섹션 4: 모델 파라미터 정보")

row1_c1, row1_c2 = st.columns(2)
row2_c1, row2_c2 = st.columns(2)
card_map = {
    "SARIMA": row1_c1,
    "Prophet": row1_c2,
    "XGBoost": row2_c1,
    "앙상블": row2_c2,
}

for name in MODEL_ORDER:
    col = card_map[name]
    p = params.get(name, {})
    color = MODEL_COLORS[name]
    with col:
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{color}; font-size:1.1em; font-weight:bold;'>"
                f"{name}</span>",
                unsafe_allow_html=True,
            )
            if not p:
                st.caption("학습 실패 — 파라미터 없음")
                continue

            if name == "SARIMA":
                order = p.get("order", "N/A")
                seasonal_order = p.get("seasonal_order", "N/A")
                st.write(f"**(p, d, q)** = `{order}`")
                st.write(f"**(P, D, Q, m)** = `{seasonal_order}`")
                st.caption("Auto-ARIMA, stepwise, AIC 기준 선택")

            elif name == "Prophet":
                st.write(f"**changepoint_prior_scale** = `{p.get('changepoint_prior_scale', 0.1)}`")
                st.write(f"**yearly_seasonality** = `{p.get('yearly_seasonality', True)}`")
                st.write(f"**weekly_seasonality** = `False`")
                st.write(f"**interval_width** = `0.95`")

            elif name == "XGBoost":
                st.write(f"**n_estimators** = `{p.get('n_estimators', 200)}`")
                st.write(f"**max_depth** = `{p.get('max_depth', 4)}`")
                st.write(f"**learning_rate** = `{p.get('learning_rate', 0.05)}`")
                feats = p.get("features", _FEAT)
                st.write("**사용 피처:**")
                st.caption(", ".join(feats))

            elif name == "앙상블":
                tw = p.get("target_weights", {"SARIMA": 0.4, "Prophet": 0.4, "XGBoost": 0.2})
                active = p.get("active_models", list(tw.keys()))
                st.write("**적용 가중치:**")
                for model_nm, w in tw.items():
                    status = "" if model_nm in active else " *(비활성)*"
                    st.write(f"- **{model_nm}**: `{w:.0%}`{status}")
                st.caption(f"활성 모델: {len(active)}/3 (가중치 정규화 적용)")
