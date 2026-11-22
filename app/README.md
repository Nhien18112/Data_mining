# Ứng dụng phát hiện gian lận mở tài khoản (Streamlit)

Ứng dụng cho chuyên viên chống gian lận, xây trên mô hình của notebook
`bank_account_fraud_complete-v2.ipynb`.

## Cài đặt & chạy (Windows, từ thư mục `C:\BKU\DM`)

```bash
python -m venv app\.venv
app\.venv\Scripts\python -m pip install -r app\requirements.txt

# 1. Huấn luyện một lần (~10 phút; tự tải Base.csv qua kagglehub nếu chưa có)
app\.venv\Scripts\python app\train.py            # thêm --fast để bỏ bootstrap + thí nghiệm bỏ tuổi
#    hoặc chỉ định file: app\train.py --data D:\duong\dan\Base.csv

# 2. Chạy ứng dụng
app\.venv\Scripts\streamlit run app\Home.py

# 3. Kiểm thử
app\.venv\Scripts\python -m pytest app\tests -q
```

## Các trang

| Trang | Chức năng |
|---|---|
| Tổng quan | Chỉ số chính trên tháng 7, xu hướng gian lận theo tháng, giới hạn của mô hình |
| Chấm điểm hồ sơ | Nhập/lấy mẫu 1 hồ sơ, xem xác suất, quyết định và lý do (SHAP của LightGBM) |
| Chấm điểm hàng loạt | Tải CSV, nhận hàng đợi thẩm định kèm lý do chính, tải kết quả |
| Chính sách ngưỡng | Ngân sách báo động giả / khối lượng cảnh báo, ngưỡng chung hay theo tuổi, chi phí, công bằng |
| Giám sát trôi dạt | PSI từng thuộc tính so với huấn luyện, gợi ý hiệu chỉnh ngưỡng |

Giao diện: chủ đề sáng/tối trong `C:\BKU\DM\.streamlit\config.toml` (chạy Streamlit từ thư mục đó),
thành phần giao diện và kiểu biểu đồ trong `app/design.py`. Giao diện không dùng biểu tượng (icon).

## Cấu trúc

- `fraud_core/` - tiền xử lý (chuyển nguyên từ notebook), metric, giải thích, PSI, nạp mô hình.
- `train.py` - huấn luyện LR + LightGBM (3 seed) đúng như notebook, lưu vào `artifacts/`.
- `Home.py` + `views/` - giao diện Streamlit (đọc `artifacts/`, không bao giờ huấn luyện lại).

Hồ sơ mới không có `month`; ứng dụng gán `month = 5` (tháng huấn luyện cuối). Xem *Thẻ mô hình*.
