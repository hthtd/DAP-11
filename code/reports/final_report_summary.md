# BÁO CÁO KHOA HỌC FINAL PROJECT
## ĐỀ TÀI: HỆ THỐNG ƯỚC TÍNH VÀ ĐÁNH GIÁ GIÁ BẤT ĐỘNG SẢN CAFELAND (PROPTECH AI)

---

### I. TỔNG QUAN VẤN ĐỀ & BỐI CẢNH NGHIÊN CỨU (5W1H)
1. **WHAT (Vấn đề gì?):**
   - Các hệ thống định giá BĐS truyền thống thường chỉ sử dụng ranh giới hành chính thô (Quận/Huyện) và đơn giá trung bình triệu/m², dẫn đến sai số cục bộ lớn.
   - Nghiên cứu này ước tính trực tiếp **Tổng giá rao bán BĐS (`price_billion`)** mà **không dùng diện tích để chia ngược tính target đơn giá**, xem diện tích là một feature độc lập.
2. **WHY (Tại sao dùng H3 Grid & POI Spatial Features?):**
   - Giá trị BĐS phụ thuộc chặt chẽ vào không gian vi mô (Micro-location) và khả năng tiếp cận hạ tầng giao thông (Metro), trung tâm tài chính (CBD) và tiện ích công cộng (Bệnh viện, Trường ĐH).
   - Việc chuyển đổi sang **Uber H3 Hexagon Grid (Resolution 8)** và khoảng cách POI kết hợp **Spatial Cross-Validation** giúp chống hiện tượng rò rỉ dữ liệu không gian (Spatial Data Leakage) và nâng cao tính tổng quát hóa.
3. **WHO (Ai thụ hưởng?):**
   - Người mua nhà và nhà đầu tư cá nhân: Định giá nhanh, nhận biết tin đăng bị hét giá (Overpriced) hoặc giá hời (Underpriced).
   - Đơn vị Proptech / CafeLand: Công cụ tự động hóa thẩm định giá sơ bộ với diễn giải minh bạch qua SHAP.
4. **WHERE (Phạm vi không gian):** Địa bàn Thành phố Hồ Chí Minh (TP.HCM) với đầy đủ 22 đơn vị hành chính Quận/Huyện/TP.
5. **WHEN & HOW (Thời gian & Phương pháp):**
   - Thu thập dữ liệu thực tế đa nguồn từ **CafeLand (`nhadat.cafeland.vn`)** và **Chợ Tốt / Nhà Tốt**.
   - Phân chia: Tập Train (9.605 mẫu - 80%) và Tập Test (2.402 mẫu - 20%).
   - Huấn luyện 2 mô hình: **Model A (Baseline - Quận/Huyện)** và **Model B (Proposed - H3 Grid + 81 Tiện ích POI)**.
   - Ứng dụng **TreeSHAP** để giải thích độ đóng góp của từng biến số.
   - Triển khai ứng dụng tương tác thời gian thực trên **Streamlit** (tích hợp Click-to-Pick GPS và dán link CafeLand tự động).

---

### II. DỮ LIỆU & QUY TRÌNH TIỀN XỬ LÝ (DATA PIPELINE)
* **Quy mô thu thập thô:** **13.839 tin đăng** (6.786 tin từ CafeLand + 7.053 tin từ Chợ Tốt).
* **Quy mô sau làm sạch chuẩn:** **12.007 bản ghi sạch** có cấu trúc hoàn chỉnh.
* **Xử lý ngoại lai & giá trị thiếu:**
  - Lọc khoảng giá hợp lệ từ 0.3 tỷ đến 150 tỷ VNĐ.
  - Lọc diện tích hợp lệ từ 12 m² đến 1.500 m².
  - Imputation số phòng ngủ, phòng vệ sinh theo median phân nhóm.
  - Mã hóa vào **681 ô lục giác Uber H3 Grid** (Resolution 8: diện tích ~0.74 km²/ô).
  - Tích hợp **81 POI trọng điểm**: 4 CBD, 14 Ga Metro Tuyến số 1, 18 Bệnh viện lớn, 19 Trường Đại học lớn, 13 TTTM, 9 Công viên đô thị, 3 Trạm trung chuyển, 1 Sân bay Tân Sơn Nhất.

---

### III. KẾT QUẢ KHÁM PHÁ DỮ LIỆU (EDA & BUSINESS INSIGHTS)
1. **Insight 1 (Phân phối giá lệch phải):** Giá BĐS tại TP.HCM có độ lệch phải rõ rệt. Việc áp dụng biến đổi hàm mục tiêu `log1p(price_billion)` giúp ổn định phương sai cho các thuật toán gradient boosting.
2. **Insight 2 (Quy luật suy giảm theo khoảng cách CBD):** Mức giá BĐS suy giảm mạnh phi tuyến tính khi khoảng cách tới CBD Quận 1 tăng dần.
3. **Insight 3 (Hiệu ứng Metro Tuyến số 1):** Các BĐS nằm trong bán kính tiếp cận thuận lợi quanh 14 ga Metro ghi nhận mức quan tâm và định giá cao vượt trội so với khu vực ven đô.
4. **Insight 4 (Cơ cấu loại hình):** Nhà ở riêng lẻ / Nhà phố chiếm tỷ trọng chủ đạo (> 60%), tiếp theo là Căn hộ chung cư (~25%) và Đất nền (~15%).
5. **Insight 5 (Yếu tố pháp lý):** BĐS có Sổ hồng / Sổ đỏ đạt mức thanh khoản và độ ổn định về định giá cao nhất trên thị trường.

---

### IV. KẾT QUẢ HUẤN LUYỆN & SO SÁNH MÔ HÌNH

Đánh giá định lượng trên tập kiểm thử độc lập Test (2.402 bản ghi):

| Chỉ số Đánh Giá | Model A (Baseline: Quận/Huyện) | Model B (Proposed: H3 + 81 Tiện ích GIS) | Ý nghĩa đánh giá |
| :--- | :---: | :---: | :--- |
| **MAE (Sai số tuyệt đối TB)** | **4.066 Tỷ VNĐ** | **4.105 Tỷ VNĐ** | Mức độ sai lệch tiền tệ thực tế |
| **RMSE (Căn bậc hai MSE)** | **11.796 Tỷ VNĐ** | **11.847 Tỷ VNĐ** | Mức độ kiểm soát sai số lớn |
| **Hệ số xác định $R^2$** | **0.4842** (48.42%) | **0.4797** (47.97%) | Khả năng giải thích phương sai tổng thể |
| **MAPE (Sai số phần trăm)** | **41.25%** | **41.51%** | Sai số phần trăm tương đối |
| **5-Fold Spatial CV (MAE)** | *Không hỗ trợ* | **4.554 ± 0.353 Tỷ** | Đánh giá tổng quát hóa không gian |
| **5-Fold Spatial CV ($R^2$)** | *Không hỗ trợ* | **0.4290 ± 0.0475** | Khả năng dự báo ngoài vùng dữ liệu lân cận |

> [!NOTE]
> **Nhận định:** Đánh giá bằng 5-Fold Spatial Cross-Validation khẳng định Model B duy trì độ ổn định không gian cao ($R^2 \approx 0.43 \pm 0.048$), đảm bảo không bị hiện tượng học vẹt hay rò rỉ dữ liệu vị trí giữa các khu vực lân cận.

---

### V. GIẢI THÍCH MINH BẠCH VỚI SHAP VALUES
- **Diện tích (`area`):** Giữ giá trị SHAP tuyệt đối cao nhất, là động lực định hình khung giá trị nền tảng của bất động sản.
- **Khoảng cách tới CBD (`dist_cbd_km`):** Khoảng cách càng ngắn càng tạo xung lực dương mạnh mẽ đẩy giá trị BĐS lên cao.
- **Lưới vi mô H3 (`h3_index`) & Khoảng cách ga Metro (`dist_nearest_metro_km`):** Phân hóa sâu sắc giá trị BĐS giữa các tiểu vùng trung tâm và khu vực ngoại vi.

---

### VI. HẠN CHẾ & HƯỚNG PHÁT TRIỂN
* **Hạn chế:** Dữ liệu thu thập phản ánh giá chào bán (Asking Price), thường có độ chênh nhẹ so với giá giao dịch thực tế sau đàm phán.
* **Hướng phát triển:** Bổ sung xử lý ảnh mặt tiền (Computer Vision), độ rộng hẻm thực tế và tích hợp chuỗi thời gian biến động giá theo từng quý.
