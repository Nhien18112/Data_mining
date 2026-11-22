import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import design as d
from fraud_core.metrics import confusion_at, rates, threshold_at_fpr
from fraud_core.model import decide
from ui import get_policy, load_model, load_scores, num, pct, policy_chip, set_policy, setup

setup()
load_model()
k = d.t()
scores = load_scores()
val, test = scores[scores.split == "val"], scores[scores.split == "test"]
current = get_policy()

with d.band("policy"):
    d.header("Chính sách ngưỡng cảnh báo",
             "Ngưỡng được chọn trên tháng 6 rồi giữ nguyên để đánh giá trên tháng 7, đúng như khi triển khai thật. "
             "Điều chỉnh các lựa chọn bên dưới, so sánh với chính sách đang áp dụng, rồi áp dụng cho toàn bàn "
             "thẩm định.", chips=[policy_chip(current)])

COMMON, GROUP = "Chung", "Theo nhóm tuổi"
BUDGET, VOLUME = "Báo động giả", "Khối lượng"
with d.surface("controls"):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1.3, 1.5, 0.95, 0.95], gap="medium", vertical_alignment="top")
    mode = c1.segmented_control("Kiểu ngưỡng", [COMMON, GROUP], required=True, key="pol_mode",
                                default=GROUP if current["mode"] == "group" else COMMON)
    method = c2.segmented_control("Chọn ngưỡng theo", [BUDGET, VOLUME], required=True, key="pol_method", default=BUDGET)
    budget = volume = None
    if method == BUDGET:
        start = current.get("fpr_budget", 0.05)
        start = 0.05 if not np.isfinite(start) else start
        budget = c3.slider("Báo động giả tối đa trên tháng 6 (%)", 0.5, 10.0, float(round(start * 200) / 2), 0.5,
                           key="pol_budget") / 100
    else:
        volume = c3.slider("Tỷ lệ hồ sơ chuyển thẩm định (%)", 0.5, 15.0, 6.0, 0.5, key="pol_volume") / 100
    fp_cost = c4.number_input("Chi phí mỗi ca chặn nhầm", 0.0, 10_000.0, 20.0, 5.0, key="pol_fp")
    fn_mult = c5.number_input("Tổn thất bỏ lọt (× hạn mức)", 0.0, 5.0, 1.0, 0.1, key="pol_fn")


def pick(y, s):
    return threshold_at_fpr(y, s, budget) if budget is not None else float(np.quantile(s, 1 - volume))


young, old = val.customer_age < 50, val.customer_age >= 50
policy = {
    "mode": "group" if mode == GROUP else "common",
    "fpr_budget": budget if budget is not None else float("nan"),
    "thr_common": pick(val.y.to_numpy(), val.lgbm.to_numpy()),
    "thr_young": pick(val.y[young].to_numpy(), val.lgbm[young].to_numpy()),
    "thr_old": pick(val.y[old].to_numpy(), val.lgbm[old].to_numpy()),
}


def assess(df: pd.DataFrame, pol: dict) -> dict:
    flags = decide(df.lgbm.to_numpy(), df.customer_age.to_numpy(), pol)
    y, age = df.y.to_numpy(), df.customer_age.to_numpy()
    tp, fp, fn, tn = confusion_at(y, flags)
    r = rates(tp, fp, fn, tn)
    fpr_g = {g: rates(*confusion_at(y[m], flags[m]))["FPR"] for g, m in [("<50", age < 50), ("≥50", age >= 50)]}
    missed = df.proposed_credit_limit.to_numpy()[(y == 1) & ~flags].sum() * fn_mult
    return {"Cảnh báo": int(flags.sum()), "% hồ sơ": flags.mean(), "TPR": r["TPR"], "FPR": r["FPR"],
            "Precision": r["Precision"], "FPR <50": fpr_g["<50"], "FPR ≥50": fpr_g["≥50"],
            "Tỷ số FPR": fpr_g["<50"] / fpr_g["≥50"] if fpr_g["≥50"] else np.nan,
            "Gian lận bỏ lọt": fn, "Tổn thất bỏ lọt": missed, "Chi phí chặn nhầm": fp * fp_cost,
            "Tổng chi phí": missed + fp * fp_cost}


rows = {"Đề xuất · tháng 6": assess(val, policy), "Đề xuất · tháng 7": assess(test, policy),
        "Đang áp dụng · tháng 7": assess(test, current)}
t, cur = rows["Đề xuất · tháng 7"], rows["Đang áp dụng · tháng 7"]
st.session_state["policy_preview"] = t


def delta(diff: float, fmt, higher_is_good: bool) -> tuple[str, str]:
    if abs(diff) < 1e-12:
        return "Bằng chính sách đang áp dụng", ""
    good = (diff > 0) == higher_is_good
    sign = "+" if diff > 0 else "-"
    trend = ("up" if diff > 0 else "down") + ("-good" if good else "-bad")
    return f"{sign}{fmt(abs(diff))} so với đang áp dụng", trend


ratio = t["Tỷ số FPR"]
fair_tone = "good" if 0.8 <= ratio <= 1.25 else ("warning" if 0.5 <= ratio <= 2 else "critical")
fair_text = {"good": "Cân bằng", "warning": "Lệch vừa", "critical": "Lệch nhiều"}[fair_tone]
n1, tr1 = delta(t["TPR"] - cur["TPR"], lambda v: pct(v) + " điểm", True)
n2, tr2 = delta(t["FPR"] - cur["FPR"], lambda v: pct(v, 2) + " điểm", False)
n3, tr3 = delta(t["Tổng chi phí"] - cur["Tổng chi phí"], num, False)
with d.surface("outcome"):
    d.section("Kết quả trên tháng 7 nếu áp dụng đề xuất")
    PCT = {"scale": 100, "suffix": "%"}
    d.tiles([
        {"label": "Bắt được gian lận", "value": pct(t["TPR"]), "note": n1, "trend": tr1,
         "key": "pol.tpr", "raw": t["TPR"], "digits": 1, **PCT},
        {"label": "Báo động giả", "value": pct(t["FPR"], 2), "note": n2, "trend": tr2,
         "key": "pol.fpr", "raw": t["FPR"], "digits": 2, **PCT},
        {"label": "Độ chính xác cảnh báo", "value": pct(t["Precision"]),
         "note": f"{num(t['Cảnh báo'])} cảnh báo, {pct(t['% hồ sơ'])} hồ sơ",
         "key": "pol.prec", "raw": t["Precision"], "digits": 1, **PCT},
        {"label": "Cân bằng theo tuổi", "value": f"{ratio:.2f}".replace(".", ","),
         "status": (fair_text, fair_tone), "note": "Báo động giả dưới 50 / từ 50 tuổi; bằng 1 là công bằng",
         "key": "pol.ratio", "raw": ratio, "digits": 2},
        {"label": "Tổng chi phí", "value": num(t["Tổng chi phí"]), "note": n3, "trend": tr3,
         "key": "pol.cost", "raw": t["Tổng chi phí"]},
    ])

    thr_text = (f"dưới 50 tuổi <span class='cx-mono'>{policy['thr_young']:.4f}</span>, từ 50 tuổi "
                f"<span class='cx-mono'>{policy['thr_old']:.4f}</span>" if policy["mode"] == "group"
                else f"<span class='cx-mono'>{policy['thr_common']:.4f}</span> cho mọi hồ sơ")
    same = (policy["mode"] == current["mode"] and all(
        abs(policy[k] - current[k]) < 1e-12 for k in ("thr_common", "thr_young", "thr_old")))
    just_applied = st.session_state.pop("policy_just_applied", False)
    a1, a2 = st.columns([4, 1], vertical_alignment="center")
    with a1:
        confirm = ("<div class='cx-confirm' style='color:var(--c-good-text);font-weight:600;margin-top:2px'>"
                   "Đã áp dụng. Chấm điểm, hàng loạt và giám sát đang dùng ngưỡng mới.</div>") if just_applied else ""
        st.html(f"<div class='cx-lede'>Ngưỡng đề xuất: {thr_text}.</div>{confirm}")
    if a2.button("Đang áp dụng" if same else "Áp dụng chính sách", type="primary", width="stretch", disabled=same):
        if budget is None:
            policy["fpr_budget"] = rows["Đề xuất · tháng 6"]["FPR"]
        set_policy(policy)
        st.session_state["policy_just_applied"] = True
        st.rerun()

with d.surface("fairness"):
    g1, g2 = st.columns([1, 2], gap="large")
    with g1:
        d.section("Báo động giả theo nhóm tuổi", "Tháng 7. Vạch dọc là ngân sách báo động giả đã chọn.")
        groups = ["Dưới 50 tuổi", "Từ 50 tuổi"]
        vals = [t["FPR <50"] * 100, t["FPR ≥50"] * 100]
        fig = go.Figure(go.Bar(x=vals, y=groups, orientation="h", marker_color=k["accent"], width=0.45,
                               text=[f"{v:.2f}%".replace(".", ",") for v in vals], textposition="outside",
                               textfont=dict(color=k["ink"]), cliponaxis=False,
                               hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
        ref = (budget if budget is not None else rows["Đề xuất · tháng 6"]["FPR"]) * 100
        fig.add_vline(x=ref, line_width=1.5, line_color=k["ink2"])
        d.style(fig, height=220, xtitle="Báo động giả (%)", animate=True)
        fig.update_xaxes(range=[0, max(vals + [ref]) * 1.3])
        fig.update_yaxes(showgrid=False, tickfont=dict(color=k["ink2"], size=12.5))
        d.show(fig, key="pol_fpr_groups")
    with g2:
        d.section("So sánh chi tiết", "Đề xuất trên tháng 6 và tháng 7, và chính sách đang áp dụng trên tháng 7.")
        res = pd.DataFrame(rows).T
        fmt = {c: (lambda v: pct(v, 2)) for c in ["% hồ sơ", "TPR", "FPR", "Precision", "FPR <50", "FPR ≥50"]}
        fmt.update({c: num for c in ["Tổn thất bỏ lọt", "Chi phí chặn nhầm", "Tổng chi phí", "Cảnh báo",
                                     "Gian lận bỏ lọt"]})
        fmt["Tỷ số FPR"] = lambda v: f"{v:.3f}".replace(".", ",")
        st.dataframe(res.style.format(fmt))


def point_marker(fig, x, y, text):
    fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(size=11, color=k["ink"],
                    line=dict(color=k["surface"], width=2)), hoverinfo="skip", showlegend=False)
    fig.add_annotation(x=x, y=y, text=text, showarrow=False, xanchor="left", xshift=10, yshift=-10,
                       font=dict(color=k["ink"], size=12))


with d.surface("curves"):
    d.section("Đường cong trên tháng 7", "Chấm đen là điểm vận hành của chính sách đề xuất.")
    r1, r2 = st.columns(2, gap="large")
    with r1:
        fpr, tpr, _ = roc_curve(test.y, test.lgbm)
        roc = go.Figure(go.Scatter(x=fpr * 100, y=tpr * 100, mode="lines", line=dict(color=k["accent"], width=2),
                                   hovertemplate="FPR %{x:.2f}% · TPR %{y:.1f}%<extra></extra>"))
        point_marker(roc, t["FPR"] * 100, t["TPR"] * 100, f"TPR {pct(t['TPR'])}")
        d.style(roc, height=320, xtitle="Báo động giả (%)", ytitle="Bắt được gian lận (%)", animate=True)
        roc.update_xaxes(range=[0, 30])
        d.show(roc, key="pol_roc")
    with r2:
        prec, rec, _ = precision_recall_curve(test.y, test.lgbm)
        pr = go.Figure(go.Scatter(x=rec * 100, y=prec * 100, mode="lines", line=dict(color=k["accent"], width=2),
                                  hovertemplate="Recall %{x:.1f}% · Precision %{y:.1f}%<extra></extra>"))
        point_marker(pr, t["TPR"] * 100, t["Precision"] * 100, f"Precision {pct(t['Precision'])}")
        d.style(pr, height=320, xtitle="Bắt được gian lận (%)", ytitle="Độ chính xác cảnh báo (%)", animate=True)
        d.show(pr, key="pol_pr")

    d.section("Đánh đổi theo ngân sách báo động giả",
              "Ngưỡng chung, chọn trên tháng 6, đo trên tháng 7. Chấm đen là chi phí thấp nhất với chi phí đã nhập.")
    budgets = np.arange(0.005, 0.1001, 0.005)
    v_fpr, _, v_thr = roc_curve(val.y, val.lgbm)
    sw = pd.DataFrame([{"budget": b * 100, **assess(test, {"mode": "common",
                                                          "thr_common": float(v_thr[np.where(v_fpr <= b)[0][-1]])})}
                       for b in budgets])
    best = sw.loc[sw["Tổng chi phí"].idxmin()]
    s1, s2 = st.columns(2, gap="large")
    with s1:
        f1 = go.Figure(go.Scatter(x=sw.budget, y=sw.TPR * 100, mode="lines", line=dict(color=k["accent"], width=2),
                                  hovertemplate="Ngân sách %{x:.1f}% · TPR %{y:.1f}%<extra></extra>"))
        point_marker(f1, best.budget, best.TPR * 100, pct(best.TPR))
        d.style(f1, height=280, xtitle="Ngân sách báo động giả trên tháng 6 (%)", ytitle="Bắt được gian lận (%)",
                animate=True)
        d.show(f1, key="pol_sweep_tpr")
    with s2:
        f2 = go.Figure(go.Scatter(x=sw.budget, y=sw["Tổng chi phí"], mode="lines",
                                  line=dict(color=k["accent"], width=2),
                                  hovertemplate="Ngân sách %{x:.1f}% · chi phí %{y:,.0f}<extra></extra>"))
        point_marker(f2, best.budget, best["Tổng chi phí"], num(best["Tổng chi phí"]))
        d.style(f2, height=280, xtitle="Ngân sách báo động giả trên tháng 6 (%)", ytitle="Tổng chi phí", animate=True)
        f2.update_yaxes(tickformat="~s")
        d.show(f2, key="pol_sweep_cost")
    st.caption(f"Chi phí thấp nhất ở ngân sách {best.budget:.1f}% ".replace(".", ",") +
               "trên tháng 7. Chỉ dùng để tham khảo: tối ưu trực tiếp trên dữ liệu kiểm tra cho kết quả lạc quan hơn thực tế.")
