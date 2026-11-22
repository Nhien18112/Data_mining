import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import design as d
from fraud_core import drift
from fraud_core.explain import label as feat_label
from fraud_core.model import decide
from fraud_core.preprocessing import TARGET, validate_raw
from ui import get_policy, load_model, load_samples, load_scores, num, pct, policy_chip, setup

setup()
model = load_model()
meta = model.meta
k = d.t()
policy = get_policy()

d.header("Giám sát trôi dạt dữ liệu",
         "So sánh một lô hồ sơ với giai đoạn huấn luyện (tháng 0-5) bằng chỉ số ổn định quần thể PSI. "
         "Dưới 0,10 là ổn định, 0,10 đến 0,25 cần theo dõi, trên 0,25 là trôi dạt đáng kể.",
         chips=[policy_chip(policy)])

M6, M7, UPLOAD = "Tháng 6", "Tháng 7", "Tải tệp CSV"
source = st.segmented_control("Lô dữ liệu", [M6, M7, UPLOAD], default=M7, required=True, key="drift_source")
if source == UPLOAD:
    up = st.file_uploader("Tệp CSV theo định dạng Base.csv", type=["csv"])
    if up is None:
        d.banner("neutral", "Chờ tệp", "Chọn một tệp CSV, hoặc chọn tháng 6 hay tháng 7 để xem ví dụ.")
        st.stop()
    raw = pd.read_csv(up)
    errors, warnings = validate_raw(raw, model.categories)
    for w in warnings:
        d.banner("warning", "Lưu ý", d.esc(w))
    if errors:
        for e in errors:
            d.banner("critical", "Không thể phân tích", d.esc(e))
        st.stop()
    key = up.name
else:
    samples = load_samples()
    raw = samples[samples.month == (6 if source == M6 else 7)].reset_index(drop=True)
    key = source


@st.cache_data(show_spinner="Đang tính PSI và chấm điểm")
def analyse(raw: pd.DataFrame, _key: str):
    scores, cleaned, _ = model.score(raw)
    table = drift.psi_table(cleaned, meta["reference"])
    ref = meta["score_reference"]
    score_psi = drift.psi(ref["expected"], drift._numeric_shares(pd.Series(scores), ref["edges"]))
    return scores, cleaned, table, score_psi


scores, cleaned, table, score_psi = analyse(raw, key)
n_drift = int((table.PSI >= drift.PSI_WATCH).sum())
n_watch = int(((table.PSI >= drift.PSI_STABLE) & (table.PSI < drift.PSI_WATCH)).sum())
score_level = drift.label(score_psi)

d.tiles([
    {"label": "Hồ sơ trong lô", "value": num(len(raw)), "note": key, "key": "drift.n", "raw": len(raw)},
    {"label": "Thuộc tính trôi dạt", "value": str(n_drift), "note": "PSI từ 0,25 trở lên",
     "status": ("Trôi dạt", "critical") if n_drift else ("Không có", "good"), "key": "drift.n_drift", "raw": n_drift},
    {"label": "Thuộc tính cần theo dõi", "value": str(n_watch), "note": "PSI từ 0,10 đến dưới 0,25",
     "status": ("Cần theo dõi", "warning") if n_watch else ("Không có", "good"), "key": "drift.n_watch", "raw": n_watch},
    {"label": "PSI của điểm mô hình", "value": f"{score_psi:.3f}".replace(".", ","),
     "status": (score_level, d.DRIFT_TONE[score_level]), "note": "So với điểm trên tập huấn luyện",
     "key": "drift.score_psi", "raw": score_psi, "digits": 3},
])

d.section("Thuộc tính lệch nhiều nhất", "Hai mươi thuộc tính có PSI cao nhất; vạch dọc là hai mốc 0,10 và 0,25.")
chart_tab, table_tab = st.tabs(["Biểu đồ", "Bảng số liệu"])
with chart_tab:
    d.legend([(name, d.STATUS[tone][0]) for name, tone in d.DRIFT_TONE.items()][::-1])
    top = table.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.PSI, y=[feat_label(c) for c in top["Thuộc tính"]], orientation="h", width=0.6,
        marker_color=[d.STATUS[d.DRIFT_TONE[m]][0] for m in top["Mức"]],
        customdata=np.stack([top["Thuộc tính"], top["Mức"]], axis=1),
        hovertemplate="%{y} (%{customdata[0]})<br>PSI %{x:.3f} · %{customdata[1]}<extra></extra>"))
    for x in (drift.PSI_STABLE, drift.PSI_WATCH):
        fig.add_vline(x=x, line_width=1, line_color=k["axis"])
    d.style(fig, height=560, xtitle="PSI so với huấn luyện")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=k["ink2"], size=12.5))
    d.show(fig)
with table_tab:
    view = table.assign(**{"Tên": [feat_label(c) for c in table["Thuộc tính"]]})[["Tên", "Thuộc tính", "PSI", "Mức"]]
    st.dataframe(view, hide_index=True, column_config={"PSI": st.column_config.NumberColumn(format="%.3f")})

d.section("Chi tiết một thuộc tính", "Tỷ lệ hồ sơ trong từng khoảng giá trị, so với giai đoạn huấn luyện.")
col = st.selectbox("Thuộc tính", table["Thuộc tính"], format_func=lambda c: f"{feat_label(c)} ({c})")
share = drift.shares_table(cleaned, col, meta["reference"][col])
d.legend([("Huấn luyện, tháng 0-5", k["deemph"]), ("Lô đang xét", k["accent"])])
bar = go.Figure()
bar.add_bar(x=share["Khoảng"], y=share["Train"] * 100, name="Huấn luyện", marker_color=k["deemph"],
            hovertemplate="%{x}<br>Huấn luyện %{y:.1f}%<extra></extra>")
bar.add_bar(x=share["Khoảng"], y=share["Lô mới"] * 100, name="Lô đang xét", marker_color=k["accent"],
            hovertemplate="%{x}<br>Lô đang xét %{y:.1f}%<extra></extra>")
d.style(bar, height=320, ytitle="% hồ sơ", animate=True)
bar.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
d.show(bar, key=f"drift_detail_{col}")  # key per column: switching month tweens, switching column redraws

d.section("Khối lượng cảnh báo và hiệu chỉnh ngưỡng",
          "Khi lô mới lệch, cùng một ngưỡng sẽ tạo ra nhiều hoặc ít cảnh báo hơn lúc chọn ngưỡng.")
val = load_scores().query("split == 'val'")
expected_rate = decide(val.lgbm.to_numpy(), val.customer_age.to_numpy(), policy).mean()
actual_rate = decide(scores, raw["customer_age"].to_numpy(), policy).mean()
items = [
    {"label": "Tỷ lệ cảnh báo lúc chọn ngưỡng", "value": pct(expected_rate, 2), "note": "Tháng 6"},
    {"label": "Tỷ lệ cảnh báo trên lô này", "value": pct(actual_rate, 2),
     "note": f"{'+' if actual_rate >= expected_rate else '-'}{pct(abs(actual_rate - expected_rate), 2)} điểm",
     "key": "drift.alert_rate", "raw": float(actual_rate), "digits": 2, "scale": 100, "suffix": "%"},
]
if TARGET in raw.columns:
    items.append({"label": "Tỷ lệ gian lận thực của lô", "value": pct(raw[TARGET].mean(), 2),
                  "note": f"Huấn luyện {pct(meta['train_fraud_rate'], 2)}",
                  "key": "drift.fraud_rate", "raw": float(raw[TARGET].mean()), "digits": 2, "scale": 100,
                  "suffix": "%"})
d.tiles(items)

recal = float(np.quantile(scores, 1 - expected_rate))
if abs(actual_rate - expected_rate) / max(expected_rate, 1e-9) > 0.10:
    d.banner("warning", "Nên hiệu chỉnh",
             f"Tỷ lệ cảnh báo lệch hơn 10% so với lúc chọn ngưỡng. Để giữ khối lượng {pct(expected_rate, 2)}, "
             f"ngưỡng chung nên chuyển từ <span class='cx-mono'>{policy['thr_common']:.4f}</span> sang khoảng "
             f"<span class='cx-mono'>{recal:.4f}</span>.", swap_key="drift.advice")
else:
    d.banner("good", "Ổn định", "Khối lượng cảnh báo gần với lúc chọn ngưỡng, chưa cần hiệu chỉnh.",
             swap_key="drift.advice")
