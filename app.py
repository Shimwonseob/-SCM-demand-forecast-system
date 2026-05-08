import streamlit as st
from utils.data_loader import get_date_range, get_sample_bytes, validate_and_load_upload

st.set_page_config(
    page_title="전자상거래 수요예측 시스템",
    page_icon="📦",
    layout="wide",
)

st.markdown("""
<style>
/* ── 전체 최대 너비 ──────────────────────────────────────── */
.main .block-container {
    max-width: 1400px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ── 메트릭 카드 ────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background-color: #1C2333;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #2D3748;
    transition: border-color 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #4F8BF9;
}

/* ── 사이드바 헤더 폰트 통일 ───────────────────────────── */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 1rem !important;
}

/* ── 섹션 구분선 ────────────────────────────────────────── */
hr {
    border-color: #2D3748 !important;
    opacity: 1 !important;
}

/* ── 카드 컨테이너 (st.container(border=True)) ───────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1C2333;
    border-radius: 12px !important;
    border: 1px solid #2D3748 !important;
}

/* ── 테이블 헤더 배경 ────────────────────────────────────── */
thead tr th {
    background-color: #1C2333 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 전자상거래 수요예측 시스템")
st.markdown("사이드바에서 분석 메뉴를 선택하세요.")

with st.sidebar:
    # ── 데이터 소스 ────────────────────────────────────────────────────────────
    st.header("데이터 소스")
    _data_src = st.radio(
        "데이터 선택",
        ["기본 데이터 사용", "직접 업로드"],
        label_visibility="collapsed",
    )

    if _data_src == "직접 업로드":
        _uploaded = st.file_uploader(
            "엑셀 파일 업로드 (.xlsx)",
            type=["xlsx"],
            help="기본 파일과 동일한 시트·컬럼 구조 필요 (샘플 참고)",
        )
        if _uploaded is not None:
            _bytes = _uploaded.read()
            _imp_df, _exp_df, _err = validate_and_load_upload(_bytes)
            if _err:
                st.error(_err)
                for _k in ["custom_imp_df", "custom_exp_df"]:
                    st.session_state.pop(_k, None)
            else:
                if _imp_df is not None:
                    st.session_state["custom_imp_df"] = _imp_df
                if _exp_df is not None:
                    st.session_state["custom_exp_df"] = _exp_df
                _loaded = (["수입"] if _imp_df is not None else []) + (["수출"] if _exp_df is not None else [])
                st.success(f"업로드 성공 ({', '.join(_loaded)})")
        else:
            st.caption("파일 업로드 시 모든 페이지에 반영됩니다.")

        st.download_button(
            "📎 샘플 파일 다운로드",
            data=get_sample_bytes(),
            file_name="전자상거래무역_샘플.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        for _k in ["custom_imp_df", "custom_exp_df"]:
            st.session_state.pop(_k, None)

    st.divider()

    # ── 시스템 정보 ────────────────────────────────────────────────────────────
    date_start, date_end = get_date_range()
    st.header("시스템 정보")
    st.markdown(f"""
| 항목 | 내용 |
|------|------|
| **데이터 출처** | 관세청 전자상거래무역통계 |
| **데이터 기간** | {date_start} ~ {date_end} |
| **마지막 업데이트** | 2026년 2월 |
| **예측 기준일** | 2026년 5월 |
| **버전** | v1.1.0 |
""")
