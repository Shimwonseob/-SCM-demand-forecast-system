import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_export_data, load_import_data

st.set_page_config(page_title="데이터탐색", layout="wide", page_icon="🔍")
st.title("🔍 데이터 탐색")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("탐색 설정")
    target = st.radio("분석 대상", ["수입", "수출"])

# ── Data ──────────────────────────────────────────────────────────────────────
df_imp = load_import_data()
df_exp = load_export_data()
IMP_COL = "전자상거래 수입 금액"
EXP_COL = "전자상거래 수출 금액"

df_main    = df_imp    if target == "수입" else df_exp
amount_col = IMP_COL  if target == "수입" else EXP_COL
series     = df_main[amount_col]

# ── 섹션 1: 시계열 개요 ───────────────────────────────────────────────────────
st.subheader("시계열 전체 개요")
try:
    ma12 = series.rolling(12, center=True).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name="월별 금액",
        line=dict(color="steelblue", width=1.5),
    ))
    fig1.add_trace(go.Scatter(
        x=ma12.index, y=ma12.values,
        mode="lines", name="12개월 이동평균",
        line=dict(color="darkorange", width=2.5),
    ))
    fig1.add_vrect(
        x0="2020-02-01", x1="2020-06-01",
        fillcolor="rgba(255,80,80,0.12)", layer="below", line_width=0,
        annotation_text="COVID-19", annotation_position="top left",
        annotation_font=dict(color="crimson", size=11),
    )
    fig1.update_layout(
        title=f"전자상거래 {target} 시계열 전체 개요",
        xaxis_title="날짜", yaxis_title="금액(USD)",
        yaxis=dict(tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420, hovermode="x unified",
    )
    fig1.update_layout(dragmode='pan')
    st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})
except Exception as e:
    st.error(f"시계열 개요 오류: {e}")

st.divider()

# ── 섹션 2: 계절성 분해 ───────────────────────────────────────────────────────
st.subheader("계절성 분해")
try:
    decomp = seasonal_decompose(series, model="additive", period=12, extrapolate_trend="freq")

    fig2 = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Trend", "Seasonality", "Residual"],
        shared_xaxes=True,
        vertical_spacing=0.07,
    )
    fig2.add_trace(go.Scatter(
        x=decomp.trend.index, y=decomp.trend.values,
        mode="lines", name="Trend",
        line=dict(color="steelblue", width=2),
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=decomp.seasonal.index, y=decomp.seasonal.values,
        mode="lines", name="Seasonality",
        line=dict(color="mediumseagreen", width=1.5),
    ), row=2, col=1)
    resid_clean = decomp.resid.dropna()
    fig2.add_trace(go.Scatter(
        x=resid_clean.index, y=resid_clean.values,
        mode="markers", name="Residual",
        marker=dict(color="gray", size=4),
    ), row=3, col=1)

    for row in [1, 2, 3]:
        fig2.update_yaxes(tickformat=",", row=row, col=1)

    fig2.update_layout(
        title="시계열 분해 (Trend / Seasonality / Residual)",
        height=650, showlegend=False, hovermode="x unified",
    )
    fig2.update_layout(dragmode='pan')
    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})
except Exception as e:
    st.error(f"계절성 분해 오류: {e}")

st.divider()

# ── 섹션 3: 월별 패턴 히트맵 ─────────────────────────────────────────────────
st.subheader("월별 패턴 히트맵")
try:
    df_heat = df_main[[amount_col]].copy()
    df_heat["year"]  = df_heat.index.year
    df_heat["month"] = df_heat.index.month
    pivot = (
        df_heat.pivot_table(values=amount_col, index="year", columns="month", aggfunc="sum")
        .sort_index(ascending=False)   # 최신 연도가 위
    )

    fig3 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{m}월" for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale="Blues",
        colorbar=dict(title="금액(USD)"),
        hovertemplate="연도: %{y}<br>월: %{x}<br>금액: %{z:,.0f}<extra></extra>",
    ))
    fig3.update_layout(
        title="연도별 월 패턴 히트맵",
        xaxis_title="월", yaxis_title="연도",
        height=480,
    )
    fig3.update_layout(dragmode='pan')
    st.plotly_chart(fig3, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})
except Exception as e:
    st.error(f"히트맵 오류: {e}")

st.divider()

# ── 섹션 4: 코로나 전/중/후 비교 ─────────────────────────────────────────────
st.subheader("코로나 전/중/후 비교")
try:
    period_config = [
        ("코로나 이전<br>(2018.01~2020.01)", "steelblue",      "2018-01-01", "2020-01-01"),
        ("코로나 충격<br>(2020.02~2020.06)", "crimson",        "2020-02-01", "2020-06-01"),
        ("코로나 이후<br>(2020.07~2022.12)", "mediumseagreen", "2020-07-01", "2022-12-01"),
        ("회복/성장기<br>(2023.01~2026.02)", "darkorange",     "2023-01-01", "2026-02-01"),
    ]

    labels, avgs, colors = [], [], []
    for label, color, start, end in period_config:
        mask = (series.index >= start) & (series.index <= end)
        labels.append(label)
        avgs.append(float(series[mask].mean()))
        colors.append(color)

    fig4 = go.Figure(go.Bar(
        x=labels,
        y=avgs,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in avgs],
        textposition="outside",
    ))
    fig4.update_layout(
        title="구간별 월평균 전자상거래 금액 비교",
        yaxis=dict(tickformat=",", title="월평균 금액(USD)", range=[0, max(avgs) * 1.2]),
        height=480,
    )
    fig4.update_layout(dragmode='pan')
    st.plotly_chart(fig4, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})
except Exception as e:
    st.error(f"코로나 비교 오류: {e}")

st.divider()

# ── 섹션 5: 수입 vs 수출 상관관계 ────────────────────────────────────────────
st.subheader("수입 vs 수출 상관관계")
try:
    common_idx = df_imp.index.intersection(df_exp.index)
    x     = df_imp.loc[common_idx, IMP_COL].values.astype(float)
    y     = df_exp.loc[common_idx, EXP_COL].values.astype(float)
    years = common_idx.year

    r = float(np.corrcoef(x, y)[0, 1])

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="월별 데이터",
        text=common_idx.strftime("%Y-%m"),
        hovertemplate="날짜: %{text}<br>수입: %{x:,.0f}<br>수출: %{y:,.0f}<extra></extra>",
        marker=dict(
            color=years,
            colorscale="Blues",      # 오래될수록 밝은색, 최근일수록 진한 파랑
            size=7,
            showscale=True,
            colorbar=dict(title="연도"),
            line=dict(width=0.5, color="gray"),
        ),
    ))
    fig5.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name="추세선 (OLS)",
        line=dict(color="crimson", width=2, dash="dash"),
    ))
    fig5.add_annotation(
        x=0.04, y=0.93,
        xref="paper", yref="paper",
        text=f"<b>r = {r:.3f}</b>",
        showarrow=False,
        font=dict(size=14, color="crimson"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="crimson",
        borderwidth=1,
        borderpad=6,
    )
    fig5.update_layout(
        title="수입 vs 수출 상관관계",
        xaxis=dict(title="수입 금액(USD)", tickformat=","),
        yaxis=dict(title="수출 금액(USD)", tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    fig5.update_layout(dragmode='pan')
    st.plotly_chart(fig5, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})
except Exception as e:
    st.error(f"상관관계 분석 오류: {e}")


st.divider()

# ── 섹션 6: 전자상거래 무역수지 추이 ─────────────────────────────────────────
st.subheader("전자상거래 무역수지 추이 (수출 - 수입)")
try:
    _common = df_imp.index.intersection(df_exp.index)
    _balance = df_exp.loc[_common, EXP_COL] - df_imp.loc[_common, IMP_COL]
    _bar_colors = ["royalblue" if v >= 0 else "crimson" for v in _balance.values]

    fig6 = go.Figure(go.Bar(
        x=_balance.index, y=_balance.values,
        marker_color=_bar_colors, name="무역수지",
        hovertemplate="%{x|%Y-%m}<br>무역수지: %{y:,.0f} USD<extra></extra>",
    ))
    fig6.add_hline(y=0, line=dict(color="white", dash="dash", width=1))
    fig6.update_layout(
        title="전자상거래 무역수지 추이 (수출 - 수입)",
        xaxis_title="날짜", yaxis_title="무역수지 (USD)",
        yaxis=dict(tickformat=","),
        height=450, hovermode="x unified",
    )
    fig6.update_layout(dragmode='pan')
    st.plotly_chart(fig6, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': [], 'displaylogo': False})

    _avg6 = float(_balance.iloc[-6:].mean())
    _latest_bal = float(_balance.iloc[-1])
    _trend_txt = "흑자 기조" if _avg6 > 0 else "적자 기조"
    _bal_dir = "흑자" if _latest_bal > 0 else "적자"
    st.info(
        f"📊 **2026년 무역수지 전망**: 최근 6개월 평균 {_trend_txt} "
        f"(평균 ${_avg6:+,.0f}). "
        f"2026-02 기준 {_bal_dir} ${abs(_latest_bal):,.0f}."
    )
except Exception as e:
    st.error(f"무역수지 오류: {e}")