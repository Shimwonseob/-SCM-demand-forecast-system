import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_export_data, load_import_data

st.set_page_config(page_title="시나리오분석", layout="wide", page_icon="🎯")
st.title("🔀 시나리오 분석")

# ── Constants ─────────────────────────────────────────────────────────────────
N_PERIODS  = 10
FC_START   = "2026-03-01"
TRAIN_START = "2018-01-01"
IMP_COL    = "전자상거래 수입 금액"
EXP_COL    = "전자상거래 수출 금액"

_SARIMA_KW = dict(
    start_p=1, max_p=3, start_q=1, max_q=3, d=1,
    start_P=0, max_P=2, start_Q=0, max_Q=2, D=1,
    seasonal=True, m=12, stepwise=True,
    suppress_warnings=True, error_action="ignore",
    information_criterion="aic",
)

BAR_COLORS = {
    "낙관": "mediumseagreen",
    "기본": "royalblue",
    "비관": "crimson",
    "충격": "darkorange",
}

# ── Cached SARIMA (cache key: target + "scenario" page) ───────────────────────
@st.cache_resource
def get_base_forecast(target: str) -> tuple:
    from pmdarima import auto_arima
    df  = load_import_data() if target == "수입" else load_export_data()
    col = IMP_COL if target == "수입" else EXP_COL
    s   = df.loc[df.index >= TRAIN_START, col]
    m   = auto_arima(s, **_SARIMA_KW)
    fc, _ = m.predict(n_periods=N_PERIODS, return_conf_int=True, alpha=0.05)
    idx   = pd.date_range(start=FC_START, periods=N_PERIODS, freq="MS")
    return fc, idx

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("시나리오 설정")
    target = st.radio("분석 대상", ["수입", "수출"])

    st.subheader("시나리오 승수")
    opt_mult = st.slider("낙관 시나리오 승수", 1.05, 1.30, 1.15, 0.01)
    pes_mult = st.slider("비관 시나리오 승수", 0.70, 0.99, 0.85, 0.01)
    st.caption(
        f"낙관: 기본 예측 대비 +{(opt_mult - 1) * 100:.0f}%  |  "
        f"비관: 기본 예측 대비 {(pes_mult - 1) * 100:.0f}%"
    )

    st.subheader("충격 시뮬레이션")
    apply_shock = st.toggle("충격 적용", value=False)
    if apply_shock:
        shock_month_str = st.selectbox(
            "충격 발생 월",
            [f"2026-{m:02d}" for m in range(3, 13)],
        )
        shock_intensity = st.slider("충격 강도 (%)", -50, 50, 0, 1)
        shock_duration  = st.slider("충격 지속 기간 (개월)", 1, 6, 1)
    else:
        shock_month_str = "2026-03"
        shock_intensity = 0
        shock_duration  = 1

# ── Data & Scenario Computation ───────────────────────────────────────────────
try:
    df_main    = load_import_data() if target == "수입" else load_export_data()
    amount_col = IMP_COL if target == "수입" else EXP_COL

    with st.spinner("시나리오 계산 중..."):
        base_fc, fc_index = get_base_forecast(target)

    opt_fc  = base_fc * opt_mult
    pes_fc  = base_fc * pes_mult

    # 충격 시나리오
    shock_fc = base_fc.copy()
    if apply_shock:
        shock_start_ts  = pd.Timestamp(shock_month_str + "-01")
        shock_start_idx = int(np.where(fc_index == shock_start_ts)[0][0])
        shock_end_idx   = min(shock_start_idx + shock_duration, N_PERIODS)
        shock_fc[shock_start_idx:shock_end_idx] = (
            base_fc[shock_start_idx:shock_end_idx] * (1 + shock_intensity / 100)
        )

    # ── 섹션 1: 시나리오 비교 차트 ───────────────────────────────────────────
    st.subheader("시나리오 비교 차트")
    hist = df_main.loc[df_main.index >= "2022-01-01", amount_col]

    fig1 = go.Figure()

    # 낙관-비관 음영 (맨 먼저 → 선 뒤에 렌더)
    fig1.add_trace(go.Scatter(
        x=list(fc_index) + list(fc_index[::-1]),
        y=list(opt_fc) + list(pes_fc[::-1]),
        fill="toself", fillcolor="rgba(0,180,0,0.10)",
        line=dict(width=0), name="낙관-비관 범위",
    ))

    # 과거 실적 (2022-01~)
    fig1.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines",
        name="과거 실적", line=dict(color="gray", width=1.5),
    ))

    # 시나리오 선
    for label, fc_arr, color, dash in [
        ("낙관 시나리오", opt_fc,  "mediumseagreen", "solid"),
        ("기본 시나리오", base_fc, "royalblue",      "solid"),
        ("비관 시나리오", pes_fc,  "crimson",        "solid"),
    ]:
        fig1.add_trace(go.Scatter(
            x=fc_index, y=fc_arr, mode="lines", name=label,
            line=dict(color=color, width=2, dash=dash),
        ))

    if apply_shock:
        fig1.add_trace(go.Scatter(
            x=fc_index, y=shock_fc, mode="lines",
            name="충격 시나리오",
            line=dict(color="darkorange", width=2, dash="dot"),
        ))
        # 충격 구간 박스
        shock_end_ts = fc_index[min(shock_start_idx + shock_duration - 1, N_PERIODS - 1)]
        fig1.add_vrect(
            x0=shock_start_ts.strftime("%Y-%m-%d"),
            x1=(shock_end_ts + pd.DateOffset(months=1)).strftime("%Y-%m-%d"),
            fillcolor="rgba(255,165,0,0.14)", layer="below", line_width=0,
            annotation_text="충격 구간", annotation_position="top left",
            annotation_font=dict(color="darkorange", size=11),
        )

    fig1.update_layout(
        title=f"시나리오별 전자상거래 {target} 예측 비교",
        xaxis_title="날짜", yaxis_title="금액(USD)",
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480, hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    # ── 섹션 2: 연간 합계 비교 ───────────────────────────────────────────────
    st.subheader("2026년 시나리오별 연간 예측 합계 (2026-03 ~ 2026-12)")

    # 2025년 동기 실적 (3~12월, 비교 기준)
    actual_2025 = float(
        df_main.loc["2025-03-01":"2025-12-01", amount_col].sum()
    )

    scenarios: dict[str, float] = {
        "낙관": float(opt_fc.sum()),
        "기본": float(base_fc.sum()),
        "비관": float(pes_fc.sum()),
    }
    if apply_shock:
        scenarios["충격"] = float(shock_fc.sum())

    bar_labels = list(scenarios.keys())
    bar_vals   = list(scenarios.values())
    bar_colors_list = [BAR_COLORS[n] for n in bar_labels]
    yoy_pct    = [(v - actual_2025) / actual_2025 * 100 for v in bar_vals]
    bar_texts  = [
        f"${v:,.0f}<br>({p:+.1f}%)"
        for v, p in zip(bar_vals, yoy_pct)
    ]

    fig2 = go.Figure(go.Bar(
        x=bar_labels,
        y=bar_vals,
        marker_color=bar_colors_list,
        text=bar_texts,
        textposition="outside",
    ))
    fig2.add_hline(
        y=actual_2025,
        line_dash="dash", line_color="dimgray", line_width=1.5,
        annotation_text=f"2025 실적 3~12월: ${actual_2025:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="dimgray", size=11),
    )
    fig2.update_layout(
        title="2026년 시나리오별 연간 예측 합계",
        yaxis=dict(
            tickformat=",", title="합계 금액(USD)",
            range=[0, max(bar_vals) * 1.22],
        ),
        height=440,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── 섹션 3: 시나리오별 상세 테이블 ──────────────────────────────────────
    st.subheader("시나리오별 상세 테이블")

    table: dict = {
        "연월":         fc_index.strftime("%Y-%m"),
        "낙관(USD)":    opt_fc.round().astype(int),
        "기본(USD)":    base_fc.round().astype(int),
        "비관(USD)":    pes_fc.round().astype(int),
        "낙관-비관 범위": (opt_fc - pes_fc).round().astype(int),
    }
    if apply_shock:
        table["충격 시나리오(USD)"] = shock_fc.round().astype(int)

    table_df = pd.DataFrame(table)
    fmt_map  = {c: "{:,}" for c in table_df.columns if c != "연월"}
    st.dataframe(
        table_df.style.format(fmt_map),
        hide_index=True,
        use_container_width=True,
    )
    _today_sc = pd.Timestamp.now().strftime("%Y%m%d")
    _csv_sc = table_df.copy()
    for _c in _csv_sc.columns:
        if _c != "연월" and pd.api.types.is_numeric_dtype(_csv_sc[_c]):
            _csv_sc[_c] = _csv_sc[_c].apply(lambda x: f"{int(x):,}")
    st.download_button(
        "📥 CSV 다운로드",
        data=_csv_sc.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"시나리오분析_{target}_{_today_sc}.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"시나리오 분석 오류: {e}")
