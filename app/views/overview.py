import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import design as d
import scene3d
from ui import get_policy, load_model, num, pct, policy_chip, setup

setup()
meta = load_model().meta
k = d.t()
test = meta["results"]["test"]["LightGBM"]

policy = get_policy()
with d.band("overview"):
    d.header(
        "Phát hiện gian lận mở tài khoản",
        "Mô hình LightGBM xếp hạng rủi ro của đơn mở tài khoản trực tuyến; hồ sơ vượt ngưỡng chuyển sang thẩm định "
        "thủ công. Mọi số liệu đo trên tháng 7, tháng mô hình chưa từng thấy.",
        chips=[policy_chip(policy),
               "Huấn luyện tháng 0-5 · chọn ngưỡng tháng 6 · kiểm tra tháng 7",
               f'Cập nhật <b class="cx-mono">{meta["created"]}</b>'],
    )
    left, right = st.columns([1, 1.35], gap="large", vertical_alignment="center")
    with left:
        d.hero(pct(test["TPR@ngưỡng"]),
               f"hồ sơ gian lận tháng 7 bị chặn tại ngưỡng triển khai, trong khi chỉ {pct(test['FPR'], 2)} "
               "khách hàng hợp lệ bị chặn nhầm.")
        st.html("<div style='height:12px'></div>")
        d.tiles([
            {"label": "TPR @ FPR 5%", "value": pct(test["TPR@FPR5% (xếp hạng)"]),
             "note": "Bài báo gốc khoảng 56%"},
            {"label": "Độ chính xác", "value": pct(test["Precision"]),
             "note": "Cảnh báo đúng là gian lận"},
            {"label": "ROC-AUC", "value": f"{test['ROC-AUC']:.3f}".replace(".", ","),
             "note": "Khả năng xếp hạng"},
        ], cols=3)
    with right:
        scene = scene3d.render(policy["thr_common"], height=430)
        d.legend([("Gian lận", "#ff6b6b"), ("Hợp lệ", "#86b6ef"), ("Mặt phẳng: ngưỡng chung", "rgba(255,255,255,0.55)")])
        st.caption(f"{num(scene['nFraud'])} ca gian lận và {num(scene['nLegit'])} hồ sơ hợp lệ lấy mẫu từ tháng 7; "
                   f"{num(scene['caught'])} ca gian lận nằm trên ngưỡng. Kéo để xoay, nhấp đúp để đặt lại.")

m = pd.DataFrame(meta["monthly"])
base = meta["train_fraud_rate"] * 100
split = ["Huấn luyện"] * 6 + ["Chọn ngưỡng", "Kiểm tra"]

with d.surface("trend"):
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        d.section("Tỷ lệ gian lận theo tháng",
                  "Tăng rõ ở tháng 6-7 so với giai đoạn huấn luyện, nên ngưỡng cần hiệu chỉnh định kỳ.")
        fig = go.Figure()
        fig.add_hline(y=base, line_width=1, line_color=k["axis"],
                      annotation_text=f"Trung bình huấn luyện {base:.2f}%".replace(".", ","),
                      annotation_position="bottom right", annotation_font=dict(color=k["ink2"], size=11.5))
        fig.add_scatter(x=m.month, y=m.rate * 100, mode="lines+markers", name="Tỷ lệ gian lận",
                        line=dict(color=k["accent"], width=2),
                        marker=dict(size=9, color=k["accent"], line=dict(color=k["surface"], width=2)),
                        customdata=split, hovertemplate="Tháng %{x} · %{customdata}<br>%{y:.2f}%<extra></extra>")
        last = m.iloc[-1]
        fig.add_annotation(x=last.month, y=last.rate * 100, text=f"{last.rate * 100:.2f}%".replace(".", ","),
                           showarrow=False, xanchor="left", xshift=8, font=dict(color=k["ink"], size=12.5))
        d.style(fig, height=290, xtitle="Tháng", ytitle="Tỷ lệ gian lận (%)")
        fig.update_xaxes(dtick=1, range=[-0.3, 7.6])
        d.show(fig)
    with c2:
        d.section("Số hồ sơ theo tháng", "Chia tập theo thời gian, không trộn ngẫu nhiên.")
        colors = [k["deemph"]] * 6 + [k["accent"]] * 2
        fig2 = go.Figure(go.Bar(x=m.month, y=m.N, marker_color=colors, width=0.55, customdata=split,
                                hovertemplate="Tháng %{x} · %{customdata}<br>%{y:,} hồ sơ<extra></extra>"))
        d.style(fig2, height=262, xtitle="Tháng", ytitle="Số hồ sơ")
        fig2.update_xaxes(dtick=1)
        fig2.update_yaxes(tickformat="~s")
        d.legend([("Tháng 0-5 · huấn luyện", k["deemph"]), ("Tháng 6-7 · chọn ngưỡng và kiểm tra", k["accent"])])
        d.show(fig2)

d.section("Giới hạn cần nhớ")
st.html(f"""<ul class="cx-notes">
<li>Độ chính xác cảnh báo khoảng {pct(test['Precision'], 0)}: phần lớn cảnh báo vẫn là khách hàng hợp lệ, vì vậy
cảnh báo chỉ dẫn tới thẩm định thủ công, không tự động từ chối.</li>
<li>Ngưỡng chung làm nhóm từ 50 tuổi bị chặn nhầm nhiều hơn khoảng ba lần. Trang Chính sách ngưỡng cho thấy đánh đổi
khi dùng ngưỡng riêng theo tuổi.</li>
<li>Dữ liệu là bộ tổng hợp BAF (NeurIPS 2022); các mã như <code>AA</code>, <code>BA</code>, <code>CC</code> đã được
ẩn danh. Hồ sơ mới không có tháng nộp, ứng dụng gán <code>month = 5</code>.</li>
</ul>""")
