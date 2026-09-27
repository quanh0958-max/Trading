# 📈 Hệ Thống Kiểm Định Chiến Lược Giao Dịch Định Lượng: SMA + OBV

Ứng dụng web tương tác được xây dựng trên nền tảng **Streamlit** và **Plotly**, cho phép trực quan hóa, kiểm định lịch sử (Backtest), tối ưu hóa tham số bằng thuật toán Bayesian (**Hyperopt TPE**), và đánh giá chuyên sâu hiệu quả của chiến lược kết hợp giữa **Đường trung bình động giản đơn (SMA)** và **Chỉ số khối lượng cân bằng (OBV)**.

Dự án được chuẩn hóa từ nghiên cứu định lượng trong notebook `NHOM_5_SMA_+_OBV.ipynb` trên tập dữ liệu giá cổ phiếu **ACB** (`ACB.csv`).

---

## 🌟 Tính Năng Nổi Bật Của Web App

1. **Giao diện Trực quan & Tương tác cao**:
   - Biểu đồ nến kỹ thuật đa tầng (Price Candlestick, Volume, OBV & OBV Moving Average) với khả năng phóng to, thu nhỏ và rê chuột kiểm tra chi tiết từng phiên.
   - Đánh dấu trực quan các điểm mua (🟢 BUY) và điểm bán (🔴 SELL).
2. **Kiểm định Chặt chẽ (Train/Test Out-of-Sample)**:
   - Phân chia dữ liệu theo thời gian (mặc định 80% Train, 20% Test) để kiểm định khả năng thích ứng của chiến lược với dữ liệu chưa từng biết trong quá khứ.
3. **So sánh Đối sánh 3 Mô hình**:
   - Đối chiếu trực tiếp hiệu suất giữa: **Chỉ dùng SMA**, **Chỉ dùng OBV**, và **Chiến lược Kết hợp SMA + OBV**.
   - Bảng tổng hợp các chỉ số cốt lõi: Tỷ suất sinh lời (*Return %*), Chỉ số *Sharpe Ratio*, Mức sụt giảm tối đa (*Max Drawdown %*), Tỷ lệ thắng (*Win Rate %*), và Số lượng lệnh (*# Trades*).
4. **Tích hợp Thuật toán Tối ưu hóa Bayesian (Hyperopt TPE)**:
   - Cho phép người dùng chạy thuật toán tìm kiếm bộ tham số tối ưu (`ma_short`, `ma_long`, `obv_window`) **chỉ trên tập Train**, đảm bảo không rò rỉ dữ liệu (*Data Leakage*).
5. **Hỗ trợ 2 Cơ chế Kết hợp Tín hiệu**:
   - **Giao cắt đồng thời (Strict AND - Chuẩn Notebook)**: BUY khi cả SMA và OBV cùng giao cắt trên đúng cùng 1 phiên.
   - **Bộ lọc Xu hướng + Điểm Kích hoạt (Regime Filter + Momentum Trigger)**: SMA đóng vai trò định hướng xu hướng lớn, OBV kích hoạt điểm vào lệnh — giúp khắc phục hiện tượng 0 giao dịch trên tập Test.
6. **Linh hoạt Nguồn Dữ liệu**:
   - Tự động nạp dữ liệu mẫu `ACB.csv` hoặc cho phép người dùng tải lên bất kỳ file CSV cổ phiếu nào có định dạng chuẩn (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                   # Mã nguồn chính của ứng dụng Streamlit
├── requirements.txt         # Danh sách các thư viện cần thiết
├── README.md                # Hướng dẫn chi tiết sử dụng và triển khai
├── ACB.csv                  # Dữ liệu giá cổ phiếu mẫu (ACB 2014 - 2023)
└── NHOM_5_SMA_+_OBV.ipynb   # Notebook nghiên cứu và thực nghiệm ban đầu
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Trên Máy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Đã cài đặt **Python** (phiên bản khuyến nghị: từ `3.9` đến `3.11`).

### 2. Khởi tạo môi trường ảo (Khuyến nghị)
Mở Terminal hoặc Command Prompt/PowerShell tại thư mục dự án:

```bash
# Tạo môi trường ảo venv
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows:
.\venv\Scripts\activate

# Trên macOS/Linux:
source venv/bin/activate
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Khởi chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Sau khi lệnh chạy thành công, trình duyệt web của bạn sẽ tự động mở địa chỉ: `http://localhost:8501`.

---

## 🌐 Hướng Dẫn Đưa Lên GitHub & Deploy Miễn Phí Trên Streamlit Cloud

### Bước 1: Tạo Repository trên GitHub
1. Đăng nhập vào tài khoản [GitHub](https://github.com).
2. Bấm nút **New** (Tạo mới repository).
3. Đặt tên repository (ví dụ: `trading-sma-obv-streamlit`), chọn chế độ **Public**, sau đó bấm **Create repository**.

### Bước 2: Tải các file lên GitHub
Bạn có thể dùng Git CLI hoặc kéo thả trực tiếp trên giao diện web GitHub:
- Các file bắt buộc cần tải lên:
  - `app.py`
  - `requirements.txt`
  - `README.md`
  - `ACB.csv`

*(Nếu dùng Git qua dòng lệnh):*
```bash
git init
git add app.py requirements.txt README.md ACB.csv
git commit -m "feat: initial commit for trading backtest web app"
git branch -M main
git remote add origin https://github.com/<TEN_TAI_KHOAN>/trading-sma-obv-streamlit.git
git push -u origin main
```

### Bước 3: Triển khai lên Streamlit Community Cloud
1. Truy cập vào [Streamlit Community Cloud](https://share.streamlit.io) và đăng nhập bằng tài khoản GitHub của bạn.
2. Bấm nút **New app** (hoặc **Create app**).
3. Điền các thông tin:
   - **Repository**: Chọn repository bạn vừa tạo (ví dụ: `<TEN_TAI_KHOAN>/trading-sma-obv-streamlit`).
   - **Branch**: `main`.
   - **Main file path**: `app.py`.
4. Bấm **Deploy!**. Hệ thống của Streamlit sẽ tự động cài đặt các thư viện trong `requirements.txt` và khởi chạy web app trong khoảng 1-2 phút. Bạn sẽ nhận được một đường link public để gửi cho bạn bè hoặc giảng viên truy cập.

---

## 🔬 Tóm Tắt Kết Quả Nghiên Cứu & Nhận Định Định Lượng

Dựa trên thực nghiệm chạy thuật toán Hyperopt trên tập dữ liệu cổ phiếu ACB:
- **Tập Train (2014 - 2021)**:
  - Tham số tối ưu tìm được: $SMA_{short} = 150$, $SMA_{long} = 210$, $OBV_{window} = 5$.
  - Chiến lược SMA: Đạt tỷ suất **+513.51%** (Sharpe: 1.03, 4 lệnh).
  - Chiến lược OBV: Đạt tỷ suất **+744.28%** (Sharpe: 1.35, 247 lệnh).
  - Chiến lược Kết hợp SMA + OBV (Strict AND): Đạt tỷ suất **+6.87%** (Sharpe: 0.44, 1 lệnh).
- **Tập Test (2022 - 2023 - Out of Sample)**:
  - Chiến lược SMA: Đạt **+12.80%** (Sharpe: 0.66).
  - Chiến lược OBV: Đạt **+2.55%** (Sharpe: 0.16) — *cho thấy hiện tượng Overfitting rõ rệt của OBV khi tối ưu cửa sổ quá ngắn*.
  - Chiến lược Kết hợp SMA + OBV (Strict AND): Đạt **0.00%** (0 lệnh).

> **💡 Bài học định lượng quan trọng:**
> Việc ép hai chỉ báo độc lập phải có điểm giao cắt trong **chính xác cùng 1 phiên giao dịch** là điều kiện quá nghiêm ngặt trong phân tích định lượng. Trên ứng dụng web, người dùng có thể kích hoạt tùy chọn **"Bộ lọc Xu hướng + Điểm Kích hoạt"** để kiểm chứng phương pháp tiếp cận thực chiến hiệu quả hơn.

---

## 👥 Nhóm Tác Giả & Bản Quyền
- Dự án bài tập định lượng: **Nhóm 5**
- Mã nguồn mở dưới giấy phép MIT License.
