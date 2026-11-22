"""Entry point: run `streamlit run app/Home.py` from C:\\BKU\\DM (so .streamlit/config.toml applies)."""
import streamlit as st

st.set_page_config(page_title="Bàn thẩm định gian lận", layout="wide", initial_sidebar_state="collapsed")

# Three pages tell the story in order: results -> a live application -> the business trade-off.
# Batch scoring and drift monitoring stay built and reachable by URL (/hang-loat, /troi-dat), out of the menu.
pages = st.navigation(
    [
        st.Page("views/overview.py", title="Tổng quan", default=True),
        st.Page("views/score_one.py", title="Chấm điểm hồ sơ", url_path="cham-diem"),
        st.Page("views/policy.py", title="Chính sách ngưỡng", url_path="nguong"),
        st.Page("views/batch.py", title="Chấm điểm hàng loạt", url_path="hang-loat", visibility="hidden"),
        st.Page("views/drift.py", title="Giám sát trôi dạt", url_path="troi-dat", visibility="hidden"),
    ],
    position="top",
)
pages.run()
