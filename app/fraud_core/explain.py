"""Per-application explanations from LightGBM's built-in SHAP values (pred_contrib=True)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import FEATURE_COLS

FEATURE_LABELS = {
    "income": "Thu nhập",
    "name_email_similarity": "Độ giống tên - email",
    "customer_age": "Tuổi",
    "days_since_request": "Số ngày từ lúc yêu cầu",
    "prev_address_months_count": "Số tháng ở địa chỉ cũ",
    "current_address_months_count": "Số tháng ở địa chỉ hiện tại",
    "intended_balcon_amount": "Số tiền nạp dự kiến",
    "bank_months_count": "Thâm niên ngân hàng cũ (tháng)",
    "zip_count_4w": "Số đơn cùng mã bưu chính (4 tuần)",
    "velocity_6h": "Vận tốc nộp đơn 6h",
    "velocity_24h": "Vận tốc nộp đơn 24h",
    "velocity_4w": "Vận tốc nộp đơn 4 tuần",
    "bank_branch_count_8w": "Số đơn tại chi nhánh (8 tuần)",
    "date_of_birth_distinct_emails_4w": "Số email cùng ngày sinh (4 tuần)",
    "credit_risk_score": "Điểm rủi ro tín dụng",
    "proposed_credit_limit": "Hạn mức đề xuất",
    "session_length_in_minutes": "Thời lượng phiên (phút)",
    "device_distinct_emails_8w": "Số email trên thiết bị (8 tuần)",
    "email_is_free": "Email miễn phí",
    "phone_home_valid": "SĐT bàn hợp lệ",
    "phone_mobile_valid": "SĐT di động hợp lệ",
    "has_other_cards": "Đã có thẻ khác",
    "foreign_request": "Yêu cầu từ nước ngoài",
    "keep_alive_session": "Duy trì đăng nhập",
    "prev_address_months_count_flag": "Không khai địa chỉ cũ",
    "intended_balcon_amount_flag": "Không khai số tiền nạp",
    "bank_months_count_flag": "Không khai thâm niên ngân hàng",
    "current_address_months_count_flag": "Không khai địa chỉ hiện tại",
    "session_length_in_minutes_flag": "Thiếu thời lượng phiên",
    "device_distinct_emails_8w_flag": "Thiếu số email trên thiết bị",
    "vel_6h_24h": "Tỷ lệ vận tốc 6h/24h",
    "vel_24h_4w": "Tỷ lệ vận tốc 24h/4 tuần",
    "addr_per_age": "Thời gian ở địa chỉ / tuổi",
    "bank_per_age": "Thâm niên ngân hàng / tuổi",
    "limit_per_income": "Hạn mức / thu nhập",
    "balcon_per_income": "Tiền nạp / thu nhập",
    "payment_type": "Phương thức thanh toán",
    "employment_status": "Nghề nghiệp",
    "housing_status": "Tình trạng nhà ở",
    "source": "Kênh nộp đơn",
    "device_os": "Hệ điều hành",
    "month": "Tháng",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def contributions(boosters, X_lgb: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Average log-odds contributions over seeds. Returns (contrib[n, n_features], bias[n])."""
    total = None
    for b in boosters:
        c = np.asarray(b.predict(X_lgb, pred_contrib=True))
        total = c if total is None else total + c
    total = total / len(boosters)
    return total[:, :-1], total[:, -1]


def _fmt(value) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "thiếu"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.3g}"
    return str(value)


def explain_row(contrib_row: np.ndarray, x_row: pd.Series, top: int = 8) -> pd.DataFrame:
    df = pd.DataFrame({"feature": FEATURE_COLS, "contrib": contrib_row})
    df["value"] = [x_row[f] for f in FEATURE_COLS]
    df = df.reindex(df["contrib"].abs().sort_values(ascending=False).index).head(top)
    df["Đặc trưng"] = [label(f) for f in df["feature"]]
    df["Giá trị"] = [_fmt(v) for v in df["value"]]
    df["Tác động"] = np.where(df["contrib"] > 0, "Tăng rủi ro", "Giảm rủi ro")
    return df.reset_index(drop=True)


def top_reasons(contrib: np.ndarray, k: int = 3) -> list[str]:
    """Top-k risk-increasing feature labels for each row."""
    order = np.argsort(-contrib, axis=1)[:, :k]
    return ["; ".join(label(FEATURE_COLS[j]) for j in row if contrib[i, j] > 0)
            for i, row in enumerate(order)]
