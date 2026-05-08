import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pmdarima import auto_arima

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_export_data, load_import_data

st.set_page_config(page_title="대시보드", layout="wide", page_icon="📊")
st.title("📊 대시보드")

TRAIN_START = "2018-01-01"
N_PERIODS = 10
FC_START = "2026-03-01"
IMP_COL = "전자상거래 수입 금액"
EXP_COL = "전자상거래 수출 금액"

_BT_TRAIN_END = "2025-02-01"
_BT_TEST_START = "2025-03-01"
_BT_TEST_END = "2026-02-01"
_BT_N = 12

_SARIMA_KW_BT = dict(
    start_p=1, max_p=3, start_q=1, max_q=3, d=1,
    start_P=0, max_P=2, start_Q=0, max_Q=2, D=1,
    seasonal=True, m=12, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    information_criterion="aic",
)

_BT_FEAT = [
    "month", "year",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "rolling_3", "rolling_6", "rolling_12",
]


def _bt_xgb_features(series: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"y": series})
    d["month"] = d.index.month
    d["year"] = d.index.year
    for lag in [1, 2, 3, 6, 12]:
        d[f"lag_{lag}"] = d["y"].shift(lag)
    for w in [3, 6, 12]:
        d[f"rolling_{w}"] = d["y"].shift(1).rolling(w).mean()
    return d.dropna()


def _bt_xgb_predict(
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
        p = float(model.predict(pd.DataFrame([row])[_BT_FEAT])[0])
        preds.append(p)
        buf.append(p)
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return np.array(preds)


@st.cache_resource
def _get_dashboard_best_model(cache_key: str, target: str) -> tuple:
    """4개 모델 백테스트 MAPE → 최적 모델명과 MAPE 반환 (캐싱)."""
    from prophet import Prophet
    from xgboost import XGBRegressor

    df = load_import_data() if target == "수입" else load_export_data()
    col = IMP_COL if target == "수입" else EXP_COL
    train = df.loc[(df.index >= TRAIN_START) & (df.index <= _BT_TRAIN_END), col]
    actual = df.loc[(df.index >= _BT_TEST_START) & (df.index <= _BT_TEST_END), col]
    n = len(actual)
    if n == 0:
        return "SARIMA", 0.0

    preds: dict = {}

    try:
        m = auto_arima(train, **_SARIMA_KW_BT)
        fc, _ = m.predict(n_periods=n, return_conf_int=True, alpha=0.05)
        preds["SARIMA"] = np.array(fc)
    except Exception:
        pass

    try:
        mp = Prophet(
            yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False, changepoint_prior_scale=0.1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mp.fit(pd.DataFrame({"ds": train.index, "y": train.values}))
        future = mp.make_future_dataframe(periods=n, freq="MS")
        preds["Prophet"] = mp.predict(future).tail(n)["yhat"].values
    except Exception:
        pass

    try:
        ft = _bt_xgb_features(train)
        mx = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          verbosity=0, random_state=42)
        mx.fit(ft[_BT_FEAT], ft["y"])
        preds["XGBoost"] = _bt_xgb_predict(
            mx, train.values, n, actual.index[0].month, actual.index[0].year,
        )
    except Exception:
        pass

    w_map = {"SARIMA": 0.4, "Prophet": 0.4, "XGBoost": 0.2}
    avail = [(w_map[nm], nm) for nm in ["SARIMA", "Prophet", "XGBoost"] if nm in preds]
    if avail:
        tot = sum(w for w, _ in avail)
        preds["앙상블"] = sum(w * preds[nm] for w, nm in avail) / tot

    if not preds:
        return "SARIMA", 0.0

    best_name, best_mape = "SARIMA", float("inf")
    for name, p in preds.items():
        mape = float(np.mean(np.abs((actual.values - p) / actual.values)) * 100)
        if mape < best_mape:
            best_mape, best_name = mape, name

    return best_name, round(best_mape, 2)


@st.cache_resource
def get_dashboard_data() -> dict:
    df_imp = load_import_data()
    df_exp = load_export_data()

    s_imp = df_imp.loc[df_imp.index >= TRAIN_START, IMP_COL]
    s_exp = df_exp.loc[df_exp.index >= TRAIN_START, EXP_COL]

    _sarima_kwargs = dict(
        start_p=1, max_p=3,
        start_q=1, max_q=3,
        d=1,
        start_P=0, max_P=2,
        start_Q=0, max_Q=2,
        D=1,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",
    )
    model_imp = auto_arima(s_imp, **_sarima_kwargs)
    model_exp = auto_arima(s_exp, **_sarima_kwargs)

    fc_index = pd.date_range(start=FC_START, periods=N_PERIODS, freq="MS")
    fc_imp_arr, _ = model_imp.predict(n_periods=N_PERIODS, return_conf_int=True, alpha=0.05)
    fc_exp_arr, _ = model_exp.predict(n_periods=N_PERIODS, return_conf_int=True, alpha=0.05)

    return {
        "df_imp": df_imp,
        "df_exp": df_exp,
        "fc_index": fc_index,
        "fc_imp": pd.Series(fc_imp_arr, index=fc_index),
        "fc_exp": pd.Series(fc_exp_arr, index=fc_index),
    }


def _mini_chart(hist: pd.Series, fc: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="실적",
        line=dict(color="gray", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=fc.index, y=fc.values,
        mode="lines", name="예측",
        line=dict(color="royalblue", width=2, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=300,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(tickformat=","),
        hovermode="x unified",
    )
    return fig


try:
    with st.spinner("대시보드 데이터 로딩 중..."):
        data = get_dashboard_data()

    df_imp = data["df_imp"]
    df_exp = data["df_exp"]
    fc_imp = data["fc_imp"]
    fc_exp = data["fc_exp"]

    # ── 최적 모델 분석 (백테스트 캐시) ───────────────────────────────────────
    with st.spinner("최적 모델 분석 중... (최초 1회 약 10~20초)"):
        try:
            _best_name, _best_mape = _get_dashboard_best_model(
                "dashboard_best_model", "수입"
            )
        except Exception:
            _best_name, _best_mape = "SARIMA", 0.0

    # ── 수입 요약 카드 ────────────────────────────────────────────────────────
    st.subheader("📥 수입 현황")
    c1, c2, c3, c4 = st.columns(4)

    latest_val = int(df_imp.loc["2026-02-01", IMP_COL])
    c1.metric("최근 실적 (수입 2026-02)", f"${latest_val:,}")

    fc_total = int(fc_imp.sum())
    c2.metric("2026년 예측 합계 (수입)", f"${fc_total:,}")

    actual_2025 = int(df_imp.loc["2025-03-01":"2025-12-01", IMP_COL].sum())
    yoy = (fc_total - actual_2025) / actual_2025 * 100
    c3.metric(
        "전년 동기 대비 증감률",
        value=f"{yoy:+.1f}%",
        delta=round(yoy, 1),
    )

    c4.metric("추천 모델", f"{_best_name} (MAPE: {_best_mape:.2f}%)")

    st.divider()

    # ── 수출 요약 카드 ────────────────────────────────────────────────────────
    st.subheader("📤 수출 현황")
    with st.spinner("수출 추천 모델 분석 중..."):
        try:
            _best_name_exp, _best_mape_exp = _get_dashboard_best_model(
                "dashboard_best_model_exp", "수출"
            )
        except Exception:
            _best_name_exp, _best_mape_exp = "SARIMA", 0.0

    e1, e2, e3, e4 = st.columns(4)

    latest_exp = int(df_exp.loc["2026-02-01", EXP_COL])
    e1.metric("최근 실적 (수출 2026-02)", f"${latest_exp:,}")

    fc_exp_total = int(fc_exp.sum())
    e2.metric("2026년 예측 합계 (수출)", f"${fc_exp_total:,}")

    actual_2025_exp = int(df_exp.loc["2025-03-01":"2025-12-01", EXP_COL].sum())
    yoy_exp = (fc_exp_total - actual_2025_exp) / actual_2025_exp * 100
    e3.metric(
        "전년 동기 대비 증감률",
        value=f"{yoy_exp:+.1f}%",
        delta=round(yoy_exp, 1),
    )

    e4.metric("추천 모델", f"{_best_name_exp} (MAPE: {_best_mape_exp:.2f}%)")

    st.divider()

    # ── 미니 차트 ─────────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            _mini_chart(df_imp[IMP_COL], fc_imp, "전자상거래 수입 추이"),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            _mini_chart(df_exp[EXP_COL], fc_exp, "전자상거래 수출 추이"),
            use_container_width=True,
        )

    st.divider()

    # ── 핵심 인사이트 ─────────────────────────────────────────────────────────
    st.subheader("💡 핵심 인사이트")
    _ins_c1, _ins_c2, _ins_c3 = st.columns(3)
    _peak_idx = int(fc_imp.values.argmax())
    _peak_month = fc_imp.index[_peak_idx].strftime("%Y년 %m월")
    _peak_val = int(fc_imp.values.max())
    _ins_c1.info(
        f"🔝 **최고 예측 월**\n\n"
        f"{_peak_month}에 최대 **${_peak_val:,}** 수입 예측 (연중 최고)"
    )
    _yoy_icon = "📈" if yoy >= 0 else "📉"
    _yoy_sign = "+" if yoy >= 0 else ""
    _ins_c2.info(
        f"{_yoy_icon} **성장 모멘텀**\n\n"
        f"2026년 수입은 전년 대비 **{_yoy_sign}{yoy:.1f}%** 성장 예측"
    )
    _hist_m = df_imp.loc[df_imp.index.year < 2026, IMP_COL]
    _peak_mon = int(_hist_m.groupby(_hist_m.index.month).mean().idxmax())
    _ins_c3.info(
        f"📅 **계절성 패턴**\n\n"
        f"역사적으로 **{_peak_mon}월**에 전자상거래 수입 최고치"
    )

    # ── 하단 정보 ─────────────────────────────────────────────────────────────
    st.caption(
        "데이터 출처: 관세청 전자상거래 무역통계 (bandtrass.or.kr)  |  "
        "데이터 기간: 2014.01 ~ 2026.02  |  "
        "분석 기준일: 2026년 5월  |  "
        f"수입 추천: {_best_name} ({_best_mape:.2f}%)  |  "
        f"수출 추천: {_best_name_exp} ({_best_mape_exp:.2f}%)"
    )

except Exception as e:
    st.error(f"대시보드 로딩 중 오류가 발생했습니다: {e}")
