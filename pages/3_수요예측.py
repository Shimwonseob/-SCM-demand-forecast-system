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

try:
    import kaleido as _kaleido_pkg  # noqa: F401
    _kaleido_ok = True
except ImportError:
    try:
        import subprocess as _sp
        _sp.check_call(
            [sys.executable, "-m", "pip", "install", "kaleido", "-q"],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
        )
        import kaleido as _kaleido_pkg  # noqa: F401
        _kaleido_ok = True
    except Exception:
        _kaleido_ok = False

st.set_page_config(page_title="수요예측", layout="wide", page_icon="📈")
st.title("📈 수요예측")

# ── Constants ─────────────────────────────────────────────────────────────────
N_PERIODS = 10
FC_START = "2026-03-01"
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
ALL_MODELS = ["SARIMA", "Prophet", "XGBoost", "앙상블"]

_SARIMA_KW = dict(
    start_p=1, max_p=3, start_q=1, max_q=3, d=1,
    start_P=0, max_P=2, start_Q=0, max_Q=2, D=1,
    seasonal=True, m=12, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    information_criterion="aic",
)

# ── XGBoost helpers ───────────────────────────────────────────────────────────
_FEAT = [
    "month", "year",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_3", "rolling_6", "rolling_12",
]

def _xgb_features(series: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"y": series})
    d["month"] = d.index.month
    d["year"]  = d.index.year
    for lag in [1, 2, 3, 6, 12]:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in [3, 6, 12]:
        d[f"rolling_{w}"] = d["y"].shift(1).rolling(w).mean()
    return d.dropna()

def _xgb_predict(model, history: np.ndarray, n: int, s_month: int, s_year: int) -> np.ndarray:
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

# ── Metrics ───────────────────────────────────────────────────────────────────
def _metrics(actual: np.ndarray, pred: np.ndarray) -> tuple:
    mae  = float(np.mean(np.abs(actual - pred)))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    return mae, mape, rmse

# ── Cached production models ───────────────────────────────────────────────────
@st.cache_resource
def fit_sarima(target: str, start_year: int):
    from pmdarima import auto_arima
    df  = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    return auto_arima(df.loc[df.index >= f"{start_year}-01-01", col], **_SARIMA_KW)

@st.cache_resource
def fit_prophet(target: str, start_year: int):
    from prophet import Prophet
    df  = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    s   = df.loc[df.index >= f"{start_year}-01-01", col]
    m   = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                  daily_seasonality=False, changepoint_prior_scale=0.1, interval_width=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(pd.DataFrame({"ds": s.index, "y": s.values}))
    return m, s

@st.cache_resource
def fit_xgboost(target: str, start_year: int):
    from xgboost import XGBRegressor
    df  = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    s   = df.loc[df.index >= f"{start_year}-01-01", col]
    ft  = _xgb_features(s)
    m   = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                       verbosity=0, random_state=42)
    m.fit(ft[_FEAT], ft["y"])
    return m, s

@st.cache_resource
def compute_backtest(target: str, start_year: int) -> dict:
    """Train all models on [start_year ~ 2025-02], predict 2025-03 ~ 2026-02."""
    from pmdarima import auto_arima
    from prophet import Prophet
    from xgboost import XGBRegressor

    df  = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    train  = df.loc[(df.index >= f"{start_year}-01-01") & (df.index <= BACKTEST_TRAIN_END), col]
    actual = df.loc[(df.index >= BACKTEST_START) & (df.index <= BACKTEST_END), col].values
    preds  = {}

    try:
        m = auto_arima(train, **_SARIMA_KW)
        fc, _ = m.predict(n_periods=N_BACKTEST, return_conf_int=True, alpha=0.05)
        preds["SARIMA"] = fc
    except Exception:
        preds["SARIMA"] = None

    try:
        mp = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                     daily_seasonality=False, changepoint_prior_scale=0.1, interval_width=0.95)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
        fc_df = mp.predict(mp.make_future_dataframe(periods=N_BACKTEST, freq="MS"))
        preds["Prophet"] = fc_df.tail(N_BACKTEST)["yhat"].values
    except Exception:
        preds["Prophet"] = None

    try:
        ft = _xgb_features(train)
        mx = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          verbosity=0, random_state=42)
        mx.fit(ft[_FEAT], ft["y"])
        preds["XGBoost"] = _xgb_predict(mx, train.values, N_BACKTEST, 3, 2025)
    except Exception:
        preds["XGBoost"] = None

    return {"actual": actual, "preds": preds}

# ── Sidebar (1~3) ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("예측 설정")
    target = st.radio("분석 대상", ["수입", "수출"])
    selected_models = st.multiselect("모델 선택", ALL_MODELS, default=ALL_MODELS)
    start_year = st.slider("학습 시작 연도", 2014, 2022, 2018)

if not selected_models:
    st.warning("사이드바에서 모델을 하나 이상 선택하세요.")
    st.stop()

# Data (n_train 계산용)
df_full  = load_import_data() if target == "수입" else load_export_data()
amount_col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
n_train = len(df_full.loc[df_full.index >= f"{start_year}-01-01"])
_n_xgb_clean = max(0, n_train - 13)
if _n_xgb_clean < 24:
    st.sidebar.warning(
        f"⚠️ XGBoost 유효 학습 데이터 {_n_xgb_clean}행 예상 (권장 24행↑). "
        "학습 시작 연도를 앞당기세요."
    )

# ── Sidebar (4~5) ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.caption(f"현재 학습 데이터: {n_train}개월")
    w_s = w_p = w_x = 1 / 3
    if "앙상블" in selected_models:
        st.subheader("앙상블 가중치")
        w_s_raw = st.slider("SARIMA 가중치",  0.0, 1.0, 0.4, 0.05, key="ws")
        w_p_raw = st.slider("Prophet 가중치", 0.0, 1.0, 0.4, 0.05, key="wp")
        w_x_raw = st.slider("XGBoost 가중치", 0.0, 1.0, 0.2, 0.05, key="wx")
        tot = w_s_raw + w_p_raw + w_x_raw
        if tot > 0:
            w_s, w_p, w_x = w_s_raw / tot, w_p_raw / tot, w_x_raw / tot
        st.caption(f"적용 — SARIMA {w_s:.2f} / Prophet {w_p:.2f} / XGBoost {w_x:.2f}")
    show_ci = st.toggle("신뢰구간 표시 (앙상블)", value=True)

# ── Model training ────────────────────────────────────────────────────────────
fc_index  = pd.date_range(start=FC_START, periods=N_PERIODS, freq="MS")
forecasts: dict[str, pd.Series]  = {}
ci_bands:  dict[str, tuple]      = {}
fit_errors: dict[str, str]       = {}

with st.spinner("SARIMA 학습 중..."):
    try:
        m_s = fit_sarima(target, start_year)
        fc_arr, ci_arr = m_s.predict(n_periods=N_PERIODS, return_conf_int=True, alpha=0.05)
        forecasts["SARIMA"]  = pd.Series(fc_arr, index=fc_index)
        ci_bands["SARIMA"]   = (ci_arr[:, 0], ci_arr[:, 1])
    except Exception as e:
        fit_errors["SARIMA"] = str(e)

with st.spinner("Prophet 학습 중..."):
    try:
        m_p, _ = fit_prophet(target, start_year)
        fc_df   = m_p.predict(m_p.make_future_dataframe(periods=N_PERIODS, freq="MS"))
        tail    = fc_df.tail(N_PERIODS)
        forecasts["Prophet"] = pd.Series(tail["yhat"].values, index=fc_index)
        ci_bands["Prophet"]  = (tail["yhat_lower"].values, tail["yhat_upper"].values)
    except Exception as e:
        fit_errors["Prophet"] = str(e)

with st.spinner("XGBoost 학습 중..."):
    try:
        m_x, xgb_series = fit_xgboost(target, start_year)
        forecasts["XGBoost"] = pd.Series(
            _xgb_predict(m_x, xgb_series.values, N_PERIODS, 3, 2026), index=fc_index
        )
    except Exception as e:
        fit_errors["XGBoost"] = str(e)

# Ensemble forecast
ens_parts = [(w_s, "SARIMA"), (w_p, "Prophet"), (w_x, "XGBoost")]
avail = [(w, n) for w, n in ens_parts if n in forecasts]
if avail:
    tot = sum(w for w, _ in avail)
    ens_fc = sum(w * forecasts[n].values for w, n in avail) / tot
    forecasts["앙상블"] = pd.Series(ens_fc, index=fc_index)

    # Ensemble CI: models with native CI contribute upper/lower; XGBoost contributes point estimate
    lo_parts, hi_parts = [], []
    for w, n in [(w_s, "SARIMA"), (w_p, "Prophet"), (w_x, "XGBoost")]:
        if n not in forecasts:
            continue
        if n in ci_bands:
            lo_parts.append((w, ci_bands[n][0]))
            hi_parts.append((w, ci_bands[n][1]))
        else:
            lo_parts.append((w, forecasts[n].values))
            hi_parts.append((w, forecasts[n].values))
    if lo_parts:
        tot_ci = sum(w for w, _ in lo_parts)
        ci_bands["앙상블"] = (
            sum(w * v for w, v in lo_parts) / tot_ci,
            sum(w * v for w, v in hi_parts) / tot_ci,
        )

for name, msg in fit_errors.items():
    st.error(f"{name} 학습 오류: {msg}")

if not forecasts:
    st.error("학습에 성공한 모델이 없습니다.")
    st.stop()

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = go.Figure()

# 과거 실적
hist = df_full[amount_col]
fig.add_trace(go.Scatter(
    x=hist.index, y=hist.values, mode="lines",
    name="과거 실적", line=dict(color="gray", width=1.5),
))

# 앙상블 CI (가장 먼저 추가 → 선 아래에 렌더링)
if show_ci and "앙상블" in selected_models and "앙상블" in ci_bands:
    lo, hi = ci_bands["앙상블"]
    fig.add_trace(go.Scatter(
        x=list(fc_index) + list(fc_index[::-1]),
        y=list(hi) + list(lo[::-1]),
        fill="toself", fillcolor="rgba(220,20,60,0.12)",
        line=dict(width=0), name="95% 신뢰구간 (앙상블)",
    ))

# 각 모델 예측선
line_widths = {"앙상블": 3, "SARIMA": 1.8, "Prophet": 1.8, "XGBoost": 1.8}
for name in ALL_MODELS:
    if name not in selected_models or name not in forecasts:
        continue
    fig.add_trace(go.Scatter(
        x=forecasts[name].index, y=forecasts[name].values, mode="lines",
        name=name,
        line=dict(color=MODEL_COLORS[name], width=line_widths.get(name, 2)),
    ))

# COVID-19 구간
fig.add_vrect(
    x0="2020-02-01", x1="2020-06-01",
    fillcolor="rgba(255,80,80,0.12)", layer="below", line_width=0,
    annotation_text="COVID-19", annotation_position="top left",
    annotation_font=dict(color="crimson", size=11),
)

fig.update_layout(
    title=f"전자상거래 {target} 수요예측 — 멀티모델 비교",
    xaxis_title="날짜", yaxis_title="금액(USD)",
    yaxis=dict(tickformat=","),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=540, hovermode="x unified",
)
fig.update_layout(dragmode='pan')
st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})

# ── 모델 성능 비교 (백테스트) ─────────────────────────────────────────────────
st.subheader("📊 모델 성능 비교 (백테스트: 2025-03 ~ 2026-02)")
with st.spinner("백테스트 계산 중..."):
    bt = compute_backtest(target, start_year)

actual_bt = bt["actual"]
preds_bt  = dict(bt["preds"])   # shallow copy

# 앙상블 백테스트 예측값
bt_avail = [(w, n) for w, n in ens_parts if preds_bt.get(n) is not None]
if bt_avail:
    tot_bt = sum(w for w, _ in bt_avail)
    preds_bt["앙상블"] = sum(w * preds_bt[n] for w, n in bt_avail) / tot_bt

metric_rows = []
for name in ["SARIMA", "Prophet", "XGBoost", "앙상블"]:
    p = preds_bt.get(name)
    if p is not None and len(p) == len(actual_bt):
        mae, mape, rmse = _metrics(actual_bt, p)
        metric_rows.append({"모델명": name, "MAE": mae, "MAPE(%)": mape, "RMSE": rmse})

if metric_rows:
    metrics_df = pd.DataFrame(metric_rows)
    best_name  = metrics_df.loc[metrics_df["MAPE(%)"].idxmin(), "모델명"]
    metrics_df["추천여부"] = metrics_df["모델명"].apply(
        lambda m: "✅ 추천" if m == best_name else ""
    )
    st.session_state["best_model"] = best_name

    st.dataframe(
        metrics_df.style.format({"MAE": "{:,.0f}", "MAPE(%)": "{:.2f}", "RMSE": "{:,.0f}"}),
        hide_index=True, use_container_width=True,
    )

# ── 예측값 상세 테이블 ────────────────────────────────────────────────────────
st.subheader("예측값 상세 (2026-03 ~ 2026-12)")
table_data = {"연월": fc_index.strftime("%Y-%m")}
fmt_cols: dict[str, str] = {}

for name in ALL_MODELS:
    if name in forecasts and name in selected_models:
        col_label = f"{name}(USD)"
        table_data[col_label] = forecasts[name].values.round().astype(int)
        fmt_cols[col_label] = "{:,}"

if "앙상블" in ci_bands and "앙상블" in selected_models:
    lo, hi = ci_bands["앙상블"]
    table_data["하한 95%"] = lo.round().astype(int)
    table_data["상한 95%"] = hi.round().astype(int)
    fmt_cols["하한 95%"] = "{:,}"
    fmt_cols["상한 95%"] = "{:,}"
    if "앙상블" in forecasts:
        _ens_fc = forecasts["앙상블"].values
        _ci_pct = (hi - lo) / np.maximum(_ens_fc, 1) * 100
        table_data["신뢰도"] = [
            "🟢 높음" if v < 15 else ("🟡 보통" if v <= 25 else "🔴 낮음")
            for v in _ci_pct
        ]

# ── 자동 인사이트 ────────────────────────────────────────────────────────────
if "앙상블" in forecasts:
    _ens = forecasts["앙상블"].values
    _months_lbl = fc_index.strftime("%Y-%m")
    _auto_insights: list = []
    for _i in range(1, len(_ens)):
        _chg = (_ens[_i] - _ens[_i - 1]) / (_ens[_i - 1] + 1e-9) * 100
        if _chg >= 10:
            _auto_insights.append(("warning", f"⚠️ {_months_lbl[_i]}: 전월 대비 {_chg:.1f}% 급증 예측"))
        elif _chg <= -10:
            _auto_insights.append(("warning", f"⚠️ {_months_lbl[_i]}: 전월 대비 {abs(_chg):.1f}% 급감 예측"))
    for _i in range(2, len(_ens)):
        if _ens[_i] > _ens[_i - 1] > _ens[_i - 2]:
            _auto_insights.append(("info", f"📈 {_months_lbl[_i-2]}~{_months_lbl[_i]}: 3개월 연속 성장 구간"))
    _bm_fc = {n: forecasts[n].values for n in ["SARIMA", "Prophet", "XGBoost"] if n in forecasts}
    if len(_bm_fc) >= 2:
        for _i in range(len(_ens)):
            _vi = [v[_i] for v in _bm_fc.values()]
            _spread_pct = (max(_vi) - min(_vi)) / (np.mean(_vi) + 1e-9) * 100
            if _spread_pct >= 20:
                _auto_insights.append(("info", f"🔍 {_months_lbl[_i]}: 모델 간 예측 불확실성 높음 (편차 {_spread_pct:.0f}%)"))
    if _auto_insights:
        with st.expander("📢 자동 인사이트 (앙상블 기준)", expanded=True):
            for _lvl, _msg in _auto_insights:
                getattr(st, _lvl)(_msg)

if len(table_data) > 1:
    _fc_df = pd.DataFrame(table_data)
    st.dataframe(
        _fc_df.style.format(fmt_cols),
        hide_index=True, use_container_width=True,
    )
    _today = pd.Timestamp.now().strftime("%Y%m%d")
    _dl1, _dl2 = st.columns(2)
    with _dl1:
        _csv_df = _fc_df.copy()
        for _c in _csv_df.columns:
            if _c == "연월":
                continue
            if _c == "신뢰도":
                _csv_df[_c] = _csv_df[_c].str.split(" ").str[-1]
            elif pd.api.types.is_numeric_dtype(_csv_df[_c]):
                _csv_df[_c] = _csv_df[_c].apply(lambda x: f"{int(x):,}")
        st.download_button(
            "📥 CSV 다운로드",
            data=_csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"전자상거래_수요예측_{target}_{_today}.csv",
            mime="text/csv",
        )
    with _dl2:
        if not _kaleido_ok:
            st.warning("PNG 다운로드를 위해 터미널에서 'pip install kaleido' 실행 필요")
        else:
            try:
                st.download_button(
                    "🖼️ 차트 PNG 다운로드",
                    data=fig.to_image(format="png", width=1400, height=600, scale=2),
                    file_name=f"수요예측_차트_{_today}.png",
                    mime="image/png",
                )
            except Exception as _png_err:
                st.warning(f"PNG 변환 오류: {_png_err}")

# ── 모델 파라미터 요약 ────────────────────────────────────────────────────────
with st.expander("모델 파라미터 요약"):
    if "SARIMA" not in fit_errors:
        st.text(str(fit_sarima(target, start_year).summary()))
