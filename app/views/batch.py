import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import design as d
import motion
from fraud_core.explain import contributions, top_reasons
from fraud_core.metrics import confusion_at, rates
from fraud_core.model import ARTIFACTS, decide, risk_band
from fraud_core.preprocessing import RAW_INPUT_COLS, TARGET, validate_raw
from ui import get_policy, load_model, num, pct, policy_chip, setup

setup()
model = load_model()
k = d.t()
REASON_ROWS = 5000  # explain at most this many highest-risk rows
policy = get_policy()

d.header("Chấm điểm hàng loạt",
         "Tải tệp CSV theo định dạng gốc của Base.csv (29 cột đầu vào; fraud_bool và month không bắt buộc). "
         "Giá trị -1, hoặc số âm ở intended_balcon_amount, nghĩa là người nộp không khai.",
         chips=[policy_chip(policy)])

SAMPLE, UPLOAD = "Dữ liệu mẫu tháng 7", "Tải tệp CSV"
source = st.segmented_control("Nguồn dữ liệu", [SAMPLE, UPLOAD], default=SAMPLE, required=True, key="batch_source")
if source == UPLOAD:
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    up = c1.file_uploader("Tệp CSV hồ sơ", type=["csv"])
    c2.download_button("Tải tệp mẫu trống", ",".join(RAW_INPUT_COLS) + "\n", "mau_ho_so.csv", "text/csv",
                       width="stretch")
    if up is None:
        d.banner("neutral", "Chờ tệp", "Chọn một tệp CSV để chấm điểm, hoặc chuyển sang dữ liệu mẫu tháng 7.")
        st.stop()
    raw, name = pd.read_csv(up), up.name
else:
    raw, name = pd.read_csv(ARTIFACTS / "sample_batch_month7.csv"), "sample_batch_month7.csv"

errors, warnings = validate_raw(raw, model.categories)
for w in warnings:
    d.banner("warning", "Lưu ý", d.esc(w))
if errors:
    for e in errors:
        d.banner("critical", "Không thể chấm điểm", d.esc(e))
    st.stop()


@st.cache_data(show_spinner="Đang chấm điểm")
def score_frame(raw: pd.DataFrame, _name: str):
    scores, _, X = model.score(raw)
    order = np.argsort(-scores)[:REASON_ROWS]
    reasons = np.full(len(raw), "", dtype=object)
    contrib, _ = contributions(model.boosters, X.iloc[order])
    reasons[order] = top_reasons(contrib, k=3)
    return scores, reasons


scores, reasons = score_frame(raw, name)
ages = raw["customer_age"].to_numpy()
flags = decide(scores, ages, policy)
thr_row = (np.where(ages < 50, policy["thr_young"], policy["thr_old"]) if policy["mode"] == "group"
           else np.full(len(raw), policy["thr_common"]))
loss = scores * raw["proposed_credit_limit"].to_numpy()
st.session_state["batch_flagged"] = int(flags.sum())

out = raw.copy()
out.insert(0, "Xác suất", scores)
out.insert(1, "Mức rủi ro", [risk_band(p, t) for p, t in zip(scores, thr_row)])
out.insert(2, "Quyết định", np.where(flags, "Thẩm định thủ công", "Duyệt tự động"))
out.insert(3, "Tổn thất kỳ vọng", loss)
out.insert(4, "Lý do chính", reasons)
out = out.sort_values("Xác suất", ascending=False).reset_index(drop=True)
out.insert(0, "Hạng", np.arange(1, len(out) + 1))

d.tiles([
    {"label": "Hồ sơ trong lô", "value": num(len(raw)), "note": name, "key": "batch.n", "raw": len(raw)},
    {"label": "Cần thẩm định", "value": num(flags.sum()), "note": f"{pct(flags.mean())} của lô",
     "key": "batch.flagged", "raw": int(flags.sum())},
    {"label": "Tổn thất kỳ vọng được chặn", "value": num(loss[flags].sum()),
     "key": "batch.loss", "raw": float(loss[flags].sum()),
     "note": "Tổng xác suất nhân hạn mức của hồ sơ bị chặn"},
    {"label": "Điểm trung bình", "value": pct(scores.mean(), 2), "note": "Xác suất gian lận trung bình",
     "key": "batch.mean", "raw": float(scores.mean()), "digits": 2, "scale": 100, "suffix": "%"},
])

if TARGET in raw.columns:
    tp, fp, fn, tn = confusion_at(raw[TARGET], flags)
    r = rates(tp, fp, fn, tn)
    d.section("Đối chiếu với nhãn thật", "Tệp có cột fraud_bool, nên có thể đo chính sách hiện tại trên chính lô này.")
    left, right = st.columns([3, 2], gap="large", vertical_alignment="center")
    with left:
        d.tiles([
            {"label": "Bắt được gian lận", "value": pct(r["TPR"]), "note": f"{num(tp)} trên {num(tp + fn)} ca"},
            {"label": "Báo động giả", "value": pct(r["FPR"], 2), "note": f"{num(fp)} khách hàng hợp lệ"},
            {"label": "Độ chính xác cảnh báo", "value": pct(r["Precision"])},
        ])
    with right:
        st.html(f"""<table class="cx-matrix" aria-label="Ma trận nhầm lẫn">
<tr><th></th><th>Duyệt tự động</th><th>Thẩm định</th></tr>
<tr><th>Thực tế hợp lệ</th><td>{num(tn)}</td><td>{num(fp)}</td></tr>
<tr><th>Thực tế gian lận</th><td>{num(fn)}</td><td class="hit">{num(tp)}</td></tr></table>""")

d.section("Phân bố điểm của lô", "Trục dọc theo thang log để thấy cả phần đuôi rủi ro cao.")
fig = go.Figure(go.Histogram(x=scores, nbinsx=80, marker_color=k["accent"], marker_line_width=0,
                             hovertemplate="Điểm %{x}<br>%{y:,} hồ sơ<extra></extra>"))
for thv in sorted(set(np.round(thr_row, 6))):
    fig.add_vline(x=thv, line_width=1.5, line_color=k["ink"],
                  annotation_text=f"ngưỡng {thv:.4f}", annotation_position="top right",
                  annotation_font=dict(color=k["ink2"], size=11.5, family=d.MONO))
d.style(fig, height=260, xtitle="Xác suất gian lận", ytitle="Số hồ sơ")
fig.update_layout(bargap=0.08)
fig.update_yaxes(type="log")
d.show(fig)

d.section("Hàng đợi thẩm định", "Xếp từ rủi ro cao xuống thấp. Lý do chính là ba yếu tố đẩy điểm lên mạnh nhất.")
f1, f2, f3 = st.columns([1.2, 1.6, 2], vertical_alignment="bottom")
only_flagged = f1.toggle("Chỉ hồ sơ cần thẩm định", value=True)
levels = f2.multiselect("Mức rủi ro", ["Rất cao", "Cao", "Trung bình", "Thấp"], default=[],
                        placeholder="Tất cả mức rủi ro")
min_p = f3.slider("Xác suất tối thiểu", 0.0, 1.0, 0.0, 0.01)
mask = out["Xác suất"] >= min_p
if only_flagged:
    mask &= out["Quyết định"] == "Thẩm định thủ công"
if levels:
    mask &= out["Mức rủi ro"].isin(levels)
shown = motion.roll("batch.shown", int(mask.sum()), num(mask.sum()))
st.html(f"<div class='cx-tile-note'>Đang hiển thị {shown} trên {num(len(out))} hồ sơ.</div>")
st.dataframe(
    out[mask], hide_index=True, height=460,
    column_config={
        "Hạng": st.column_config.NumberColumn(width="small"),
        "Xác suất": st.column_config.ProgressColumn(format="%.3f", min_value=0.0, max_value=1.0),
        "Tổn thất kỳ vọng": st.column_config.NumberColumn(format="%.1f"),
        "Lý do chính": st.column_config.TextColumn(width="large"),
    })
if len(raw) > REASON_ROWS:
    st.caption(f"Lý do chỉ được tính cho {num(REASON_ROWS)} hồ sơ rủi ro cao nhất.")

buf = io.StringIO()
out.to_csv(buf, index=False)
st.download_button("Tải toàn bộ kết quả (CSV)", buf.getvalue().encode("utf-8-sig"), f"ket_qua_{name}",
                   "text/csv", type="primary")
