"""Shared Streamlit helpers: page setup, cached artifact loading and the active alert policy."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import design
import motion
from fraud_core.model import ARTIFACTS, FraudModel, default_policy

POLICY_KEY = "policy"


def require_artifacts() -> None:
    if not (ARTIFACTS / "meta.json").exists():
        design.banner("critical", "Chưa có mô hình",
                      "Chạy <code>app\\.venv\\Scripts\\python app\\train.py</code> từ thư mục "
                      "<code>C:\\BKU\\DM</code> rồi tải lại trang.")
        st.stop()


def setup() -> None:
    """Call at the top of every page: styles and the artifact check. The active policy is shown in each
    page header (policy_chip), since navigation now lives in the top bar and the sidebar is gone."""
    design.inject_css()
    require_artifacts()
    get_policy()


@st.cache_resource(show_spinner="Đang nạp mô hình")
def load_model() -> FraudModel:
    return FraudModel.load()


@st.cache_data(show_spinner="Đang nạp điểm tháng 6 và tháng 7")
def load_scores() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "scores.parquet")


@st.cache_data(show_spinner="Đang nạp hồ sơ tháng 6 và tháng 7")
def load_samples() -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "samples_raw.parquet")


def get_policy() -> dict:
    if POLICY_KEY not in st.session_state:
        st.session_state[POLICY_KEY] = default_policy(load_model().meta)
    return st.session_state[POLICY_KEY]


def set_policy(policy: dict) -> None:
    st.session_state[POLICY_KEY] = policy


def policy_chip(policy: dict) -> str:
    """Active thresholds; they roll to new values after a policy is applied (motion keys shared by all pages)."""
    def thr(name: str) -> str:
        v = policy[name]
        return motion.roll(f"chip.{name}", v, f"{v:.4f}", digits=4, locale="en-US")

    if policy["mode"] == "group":
        return (f'Chính sách: ngưỡng theo tuổi <b class="cx-mono">{thr("thr_young")}</b> / '
                f'<b class="cx-mono">{thr("thr_old")}</b>')
    return f'Chính sách: ngưỡng chung <b class="cx-mono">{thr("thr_common")}</b>'


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%".replace(".", ",")


def num(x: float, digits: int = 0) -> str:
    """Vietnamese number format: dot for thousands, comma for decimals."""
    s = f"{x:,.{digits}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")
