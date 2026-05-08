import io

import pandas as pd
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "전자상거래무역.xlsx"

OUTLIER_START = "2020-02"
OUTLIER_END = "2020-06"

_IMP_REQUIRED = {"연도", "월", "전자상거래 수입 금액"}
_EXP_REQUIRED = {"연도", "월", "전자상거래 수출 금액"}


def _process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(
        df["연도"].astype(str) + "-" + df["월"].astype(str).str.zfill(2)
    )
    df = df.set_index("date").sort_index()
    df.index = df.index.to_period("M").to_timestamp()
    outlier_mask = (df.index >= OUTLIER_START) & (df.index <= OUTLIER_END)
    df["is_outlier"] = outlier_mask
    return df


@st.cache_data
def _load_default_import() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="전자상거래 수입")
    return _process(df)


@st.cache_data
def _load_default_export() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="전자상거래 수출")
    return _process(df)


def load_import_data() -> pd.DataFrame:
    if "custom_imp_df" in st.session_state:
        return st.session_state["custom_imp_df"]
    return _load_default_import()


def load_export_data() -> pd.DataFrame:
    if "custom_exp_df" in st.session_state:
        return st.session_state["custom_exp_df"]
    return _load_default_export()


def get_date_range() -> tuple[str, str]:
    df = load_import_data()
    return df.index.min().strftime("%Y-%m"), df.index.max().strftime("%Y-%m")


SAMPLE_PATH = Path(__file__).parent.parent / "전자상거래_수요예측_샘플 (5).xlsx"


def get_sample_bytes() -> bytes:
    return SAMPLE_PATH.read_bytes()


def validate_and_load_upload(file_bytes: bytes) -> tuple:
    """(imp_df, exp_df, error_msg) 반환. 오류 시 imp_df/exp_df는 None."""
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as e:
        return None, None, f"파일 파싱 오류: {e}"

    sheets = xl.sheet_names
    imp_df = exp_df = None

    if "전자상거래 수입" in sheets:
        raw = xl.parse("전자상거래 수입")
        missing = _IMP_REQUIRED - set(raw.columns)
        if missing:
            return None, None, f"수입 시트 필수 컬럼 누락: {', '.join(sorted(missing))}"
        try:
            imp_df = _process(raw)
        except Exception as e:
            return None, None, f"수입 데이터 처리 오류: {e}"

    if "전자상거래 수출" in sheets:
        raw = xl.parse("전자상거래 수출")
        missing = _EXP_REQUIRED - set(raw.columns)
        if missing:
            return None, None, f"수출 시트 필수 컬럼 누락: {', '.join(sorted(missing))}"
        try:
            exp_df = _process(raw)
        except Exception as e:
            return None, None, f"수출 데이터 처리 오류: {e}"

    if imp_df is None and exp_df is None:
        return None, None, (
            "'전자상거래 수입' 또는 '전자상거래 수출' 시트가 없습니다. "
            "샘플 파일을 다운로드하여 형식을 확인하세요."
        )

    return imp_df, exp_df, None
