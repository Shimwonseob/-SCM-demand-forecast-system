import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_export_data, load_import_data

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

st.set_page_config(page_title="예측검증", layout="wide", page_icon="✅")
st.title("✅ 예측 검증")
st.caption("실제 발표된 수치와 예측값을 비교하여 모델 정확도를 검증합니다.")

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_START = "2018-01-01"
N_PERIODS = 10
FC_START = "2026-03-01"
FC_MONTHS = [f"2026-{m:02d}" for m in range(3, 13)]

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


@st.cache_resource
def get_validation_forecasts(target: str) -> dict:
    """SARIMA / XGBoost / 앙상블 예측값 (2026-03 ~ 2026-12) 캐싱."""
    from pmdarima import auto_arima
    from xgboost import XGBRegressor

    df = load_import_data() if target == "수입" else load_export_data()
    col = "전자상거래 수입 금액" if target == "수입" else "전자상거래 수출 금액"
    s = df.loc[df.index >= TRAIN_START, col]
    fc_index = pd.date_range(start=FC_START, periods=N_PERIODS, freq="MS")
    result: dict = {"fc_index": fc_index}

    try:
        m = auto_arima(s, **_SARIMA_KW)
        result["SARIMA"] = pd.Series(m.predict(n_periods=N_PERIODS), index=fc_index)
    except Exception:
        result["SARIMA"] = None

    try:
        ft = _xgb_features(s)
        mx = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            verbosity=0, random_state=42,
        )
        mx.fit(ft[_FEAT], ft["y"])
        result["XGBoost"] = pd.Series(
            _xgb_predict(mx, s.values, N_PERIODS, 3, 2026), index=fc_index
        )
    except Exception:
        result["XGBoost"] = None

    w_map = {"SARIMA": 0.5, "XGBoost": 0.5}
    avail = [(w_map[n], n) for n in ["SARIMA", "XGBoost"] if result.get(n) is not None]
    if avail:
        tot = sum(w for w, _ in avail)
        result["앙상블"] = pd.Series(
            sum(w * result[n].values for w, n in avail) / tot,
            index=fc_index,
        )
    else:
        result["앙상블"] = None

    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("검증 설정")
    target = st.radio("분석 대상", ["수입", "수출"])
    verify_month = st.selectbox("검증 월 선택", FC_MONTHS)
    st.markdown("---")
    st.caption("💡 실제값은 USD 전체 금액으로 입력하세요. 예: 2억 3천만 달러 → 230,000,000")
    actual_value = st.number_input(
        "실제값 입력 (USD)",
        min_value=0,
        value=0,
        step=1_000_000,
        format="%d",
        help="관세청 발표 실제 금액(USD). 0이면 예측값 미리보기만 표시됩니다.",
    )

# ── 예측값 로딩 ───────────────────────────────────────────────────────────────
with st.spinner("예측 모델 로딩 중... (최초 1회 약 10~20초)"):
    try:
        fcs = get_validation_forecasts(target)
    except Exception as e:
        st.error(f"모델 로딩 오류: {e}")
        st.stop()

fc_index = fcs["fc_index"]
target_ts = pd.Timestamp(verify_month + "-01")
month_idx = int(np.where(fc_index == target_ts)[0][0])

MODEL_LIST = ["SARIMA", "XGBoost", "앙상블"]
fc_val = {
    n: float(fcs[n].iloc[month_idx])
    for n in MODEL_LIST
    if fcs.get(n) is not None
}

# ── 예측값 미리보기 (실제값 미입력 시) ────────────────────────────────────────
if actual_value == 0:
    st.info("사이드바에서 검증할 월을 선택하고 **실제값을 입력**하면 오차 분석이 시작됩니다.")
    st.subheader("📋 전체 예측값 미리보기 (2026-03 ~ 2026-12)")
    rows = []
    for ms in FC_MONTHS:
        ts = pd.Timestamp(ms + "-01")
        idx = int(np.where(fc_index == ts)[0][0])
        row: dict = {"연월": ms}
        for n in MODEL_LIST:
            if fcs.get(n) is not None:
                row[f"{n}(USD)"] = int(fcs[n].iloc[idx])
        rows.append(row)
    preview_df = pd.DataFrame(rows)
    num_cols = [c for c in preview_df.columns if c != "연월"]
    st.dataframe(
        preview_df.style.format({c: "{:,}" for c in num_cols}),
        hide_index=True,
        use_container_width=True,
    )
    st.stop()

# ── 오차 계산 ─────────────────────────────────────────────────────────────────
err_abs  = {n: actual_value - v for n, v in fc_val.items()}
err_rate = {n: abs(actual_value - v) / actual_value * 100 for n, v in fc_val.items()}

best_model = min(err_rate, key=err_rate.get) if err_rate else "앙상블"
best_fc    = fc_val.get(best_model, 0.0)
best_err   = actual_value - best_fc
best_rate  = err_rate.get(best_model, 0.0)

# ── 요약 카드 4개 ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("실제값", f"${actual_value:,}")
c2.metric(f"예측값 ({best_model})", f"${best_fc:,.0f}")
_sign = "+" if best_err >= 0 else ""
c3.metric("오차 (실제 − 예측)", f"{_sign}${best_err:,.0f}")
c4.metric("오차율 (%)", f"{best_rate:.2f}%")

st.divider()

# ── 정확도 판정 및 후속조치 ───────────────────────────────────────────────────
st.subheader("🏷️ 정확도 판정")

if best_rate < 5:
    st.success(
        f"**판정: ✅ 우수** (오차율 {best_rate:.2f}%)\n\n"
        "**해결책:** 현재 모델과 파라미터 유지. 다음 달 정기 검증 권장."
    )
elif best_rate < 10:
    st.info(
        f"**판정: 🟢 양호** (오차율 {best_rate:.2f}%)\n\n"
        "**해결책:** 모델 유지. 학습 시작 연도를 최근으로 조정(2020 이후) 시 정확도 개선 가능."
    )
elif best_rate < 20:
    st.warning(
        f"**판정: 🟡 보통** (오차율 {best_rate:.2f}%)\n\n"
        "**해결책:**\n"
        "- SARIMA: 학습 시작 연도를 2020년 이후로 조정\n"
        "- XGBoost: lag 피처 범위 축소(최근 6개월 중심)\n"
        "- Prophet: changepoint\\_prior\\_scale을 0.05로 낮춰 과적합 방지\n"
        "- 앙상블 가중치에서 오차 큰 모델 비중 낮추기"
    )
else:
    st.error(
        f"**판정: 🔴 주의 필요** (오차율 {best_rate:.2f}%)\n\n"
        "**해결책:**\n"
        "- 이상치 처리 ON으로 변경\n"
        "- 학습 시작 연도를 2022년 이후로 조정 (최근 데이터 중심)\n"
        "- SARIMA 단독보다 XGBoost 또는 앙상블 모델 사용 권장\n"
        "- 외부 이벤트(정책 변화, 환율 급변 등) 발생 여부 확인 필요"
    )

st.divider()

# ── 비교 테이블 ───────────────────────────────────────────────────────────────
st.subheader(f"📊 {verify_month} 예측 vs 실제 상세 비교")

tbl_rows = []
for n in MODEL_LIST:
    if n not in fc_val:
        continue
    v = fc_val[n]
    ea = actual_value - v
    er = abs(ea) / actual_value * 100
    tbl_rows.append({
        "검증 월":     verify_month,
        "모델":        n + (" ✅" if n == best_model else ""),
        "실제값(USD)": actual_value,
        "예측값(USD)": int(v),
        "오차(USD)":   int(ea),
        "오차율(%)":   round(er, 2),
    })

if tbl_rows:
    tbl_df = pd.DataFrame(tbl_rows)
    st.dataframe(
        tbl_df.style.format({
            "실제값(USD)": "{:,}",
            "예측값(USD)": "{:,}",
            "오차(USD)":   "{:+,}",
            "오차율(%)":   "{:.2f}",
        }),
        hide_index=True,
        use_container_width=True,
    )
