import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import design as d
import motion
from fraud_core.explain import contributions, explain_row, label
from fraud_core.model import risk_band, threshold_for
from fraud_core.preprocessing import BIN_COLS, CAT_COLS, MASKED_COLS
from ui import get_policy, load_model, load_samples, num, pct, policy_chip, setup

setup()
model = load_model()
meta = model.meta
form = meta["form"]
k = d.t()

GROUPS = {
    "Hồ sơ cá nhân": ["customer_age", "income", "employment_status", "housing_status",
                      "current_address_months_count", "prev_address_months_count", "name_email_similarity"],
    "Tài chính": ["proposed_credit_limit", "credit_risk_score", "intended_balcon_amount",
                  "bank_months_count", "has_other_cards", "payment_type"],
    "Thiết bị và danh tính": ["device_os", "source", "email_is_free", "phone_home_valid", "phone_mobile_valid",
                              "foreign_request", "keep_alive_session", "session_length_in_minutes",
                              "device_distinct_emails_8w", "date_of_birth_distinct_emails_4w"],
    "Vận tốc và địa lý": ["velocity_6h", "velocity_24h", "velocity_4w", "zip_count_4w",
                          "bank_branch_count_8w", "days_since_request"],
}
AGES = list(range(10, 100, 10))


def is_missing(col, value) -> bool:
    return value < 0 if col == "intended_balcon_amount" else value == -1


def cast(col, value):
    if col in CAT_COLS:
        return str(value)
    if col == "customer_age":
        return int(min(AGES, key=lambda a: abs(a - value)))
    return int(value) if form[col]["integer"] else float(value)


def set_defaults() -> None:
    for col, spec in form.items():
        st.session_state[f"v_{col}"] = spec["mode"] if col in CAT_COLS else cast(col, spec["median"])
        if col in MASKED_COLS:
            st.session_state[f"m_{col}"] = False
    st.session_state["loaded_label"] = None


def load_row(fraud_only: bool) -> None:
    samples = load_samples()
    pool = samples[samples.month == 7]
    if fraud_only:
        pool = pool[pool.fraud_bool == 1]
    row = pool.sample(1).iloc[0]
    for col in form:
        value = row[col]
        missing = col in MASKED_COLS and is_missing(col, value)
        if col in MASKED_COLS:
            st.session_state[f"m_{col}"] = missing
        st.session_state[f"v_{col}"] = cast(col, form[col]["median"] if missing else value)
    st.session_state["loaded_label"] = int(row["fraud_bool"])


if "v_income" not in st.session_state:
    set_defaults()

policy = get_policy()
with d.band("score"):
    d.header("Chấm điểm một hồ sơ",
             "Nhập thông tin đơn đăng ký hoặc lấy một hồ sơ thật của tháng 7 để xem mô hình quyết định ra sao và vì sao.",
             chips=[policy_chip(policy)])
    b1, b2, b3, _ = st.columns([1.1, 1.1, 1, 1.6])
    b1.button("Hồ sơ ngẫu nhiên tháng 7", on_click=load_row, args=(False,), width="stretch")
    b2.button("Hồ sơ gian lận tháng 7", on_click=load_row, args=(True,), width="stretch")
    b3.button("Khôi phục mặc định", on_click=set_defaults, width="stretch", type="tertiary")


def fmt_range(v: float) -> str:
    if abs(v) < 0.001:
        return "0"
    return num(v, 0) if abs(v) >= 100 or float(v).is_integer() else num(v, 2)


def widget(col: str) -> None:
    key = f"v_{col}"
    name = label(col)
    spec = form[col]
    if col in CAT_COLS:
        st.selectbox(name, spec["levels"], key=key)
    elif col in BIN_COLS:
        st.segmented_control(name, [0, 1], key=key, format_func=lambda v: "Có" if v else "Không",
                             selection_mode="single", required=True)
    elif col == "customer_age":
        st.selectbox(name, AGES, key=key, format_func=lambda a: f"{a} tuổi")
    else:
        disabled = False
        if col in MASKED_COLS:
            disabled = st.checkbox(f"{name}: không khai", key=f"m_{col}")
        step = 1 if spec["integer"] else max((spec["max"] - spec["min"]) / 200, 1e-4)
        st.number_input(name, key=key, step=step, disabled=disabled,
                        format=None if spec["integer"] else "%.4f",
                        label_visibility="collapsed" if col in MASKED_COLS else "visible")
        st.caption(f"Khoảng huấn luyện {fmt_range(spec['min'])} đến {fmt_range(spec['max'])}"
                   + (f" · {pct(spec['missing_rate'])} hồ sơ không khai" if col in MASKED_COLS else ""))


form_col, result_col = st.columns([1.45, 1], gap="medium")
with form_col, d.surface("form"):
    with st.form("application", border=False):
        tabs = st.tabs(list(GROUPS))
        for tab, cols in zip(tabs, GROUPS.values()):
            with tab:
                grid = st.columns(2, gap="medium")
                for i, col in enumerate(cols):
                    with grid[i % 2]:
                        widget(col)
        st.form_submit_button("Chấm điểm hồ sơ", type="primary")

row = {}
for col in form:
    value = st.session_state[f"v_{col}"]
    row[col] = -1 if col in MASKED_COLS and st.session_state.get(f"m_{col}") else value
scores, _, X = model.score(pd.DataFrame([row]))
p = float(scores[0])
thr = threshold_for(row["customer_age"], policy)
band = risk_band(p, thr)
tone = d.RISK_TONE[band]
base = meta["train_fraud_rate"]
st.session_state["last_score"] = p

with result_col:
    with d.surface("verdict"):
        if p >= thr:
            d.verdict("Chuyển thẩm định", tone, f"Rủi ro {band.lower()}",
                      f"Điểm vượt ngưỡng <span class='cx-mono'>{thr:.4f}</span>; hồ sơ cần chuyên viên xem xét "
                      "trước khi mở tài khoản.", swap_key="score.verdict")
        else:
            d.verdict("Duyệt tự động", tone, f"Rủi ro {band.lower()}",
                      f"Điểm dưới ngưỡng <span class='cx-mono'>{thr:.4f}</span>; hồ sơ đi tiếp theo quy trình "
                      "thông thường.", swap_key="score.verdict")
        d.hero(pct(p, 2), "xác suất hồ sơ là gian lận, trung bình của 3 mô hình",
               key="score.p", raw=p, digits=2, scale=100, suffix="%")
        scale = max(0.25, min(1.0, np.ceil(max(p, thr) * 1.25 * 20) / 20))
        d.meter(p, thr, scale, tone, key="score.meter")
        loss = p * row["proposed_credit_limit"]
        d.facts([
            ("So với mức nền", motion.roll("score.ratio", p / base, f"gấp {p / base:.1f} lần".replace(".", ","),
                                           digits=1, prefix="gấp ", suffix=" lần")),
            ("Tổn thất kỳ vọng", motion.roll("score.loss", loss, num(loss, 1), digits=1)),
            ("Nhóm ngưỡng", "Từ 50 tuổi" if policy["mode"] == "group" and row["customer_age"] >= 50
             else ("Dưới 50 tuổi" if policy["mode"] == "group" else "Chung")),
        ])
        st.caption(f"Mức nền là tỷ lệ gian lận trung bình khi huấn luyện ({pct(base, 2)}). "
                   "Tổn thất kỳ vọng = xác suất gian lận nhân hạn mức đề xuất.")
        if st.session_state.get("loaded_label") is not None:
            truth = st.session_state["loaded_label"] == 1
            st.html(f"<div style='margin-top:4px'>Nhãn thật của hồ sơ mẫu "
                    f"{d.pill('Gian lận' if truth else 'Hợp lệ', 'critical' if truth else 'good')}</div>")

contrib, bias = contributions(model.boosters, X)
top = explain_row(contrib[0], X.iloc[0], top=10)
with d.surface("reasons"):
    d.section("Vì sao mô hình cho điểm này",
              "Mười yếu tố tác động mạnh nhất. Thanh đỏ đẩy rủi ro lên, thanh xanh kéo rủi ro xuống; độ dài là mức "
              "đóng góp vào điểm (log-odds, SHAP tích hợp của LightGBM).")
    chart_tab, table_tab = st.tabs(["Biểu đồ", "Bảng số liệu"])
    with chart_tab:
        plot = top.iloc[::-1]
        up, down = k["series_red"], k["accent"]
        d.legend([("Tăng rủi ro", up), ("Giảm rủi ro", down)])
        fig = go.Figure(go.Bar(
            x=plot["contrib"], y=[f"{a}  ·  {b}" for a, b in zip(plot["Đặc trưng"], plot["Giá trị"])],
            orientation="h", marker_color=np.where(plot["contrib"] > 0, up, down), width=0.6,
            hovertemplate="%{y}<br>Đóng góp %{x:+.3f}<extra></extra>"))
        fig.add_vline(x=0, line_width=1, line_color=k["axis"])
        d.style(fig, height=420, xtitle="Đóng góp vào điểm (log-odds)")
        fig.update_yaxes(showgrid=False, showline=False, tickfont=dict(color=k["ink2"], size=12.5))
        d.show(fig)
    with table_tab:
        st.dataframe(top[["Đặc trưng", "Giá trị", "Tác động", "contrib"]].rename(columns={"contrib": "Đóng góp"}),
                     hide_index=True, column_config={"Đóng góp": st.column_config.NumberColumn(format="%+.3f")})
    st.caption(f"Điểm cơ sở {bias[0]:+.2f} log-odds; cộng tất cả đóng góp ra điểm của hồ sơ.")
