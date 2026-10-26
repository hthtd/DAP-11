# Hệ Thống Ước Tính Và Đánh Giá Giá Bất Động Sản TP.HCM (CafeLand Proptech AI)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.53-red.svg)](https://streamlit.io/)
[![Spatial](https://img.shields.io/badge/Uber%20H3-Grid%20Res%208-brightgreen.svg)](https://h3geo.org/)
[![Explainability](https://img.shields.io/badge/SHAP-TreeSHAP-orange.svg)](https://shap.readthedocs.io/)

Đồ án môn học: **Lập Trình Phân Tích Dữ Liệu - Final Project**  
Chủ đề: **Định giá Bất động sản thông minh ứng dụng GIS Không gian vi mô (Uber H3 Grid Res 8), Hệ sinh thái 81 Điểm Tiện ích Đô thị (GeoPandas POI), Spatial Cross-Validation và Diễn giải minh bạch bằng SHAP.**

---

## 1. Cấu Trúc Thư Mục Dự Án Chuẩn Hóa (Project Structure)

```text
code/
├── run_pipeline.py                          # Script tổng chạy toàn bộ 5 bước tự động
├── app.py                                   # Ứng dụng Web tương tác Streamlit (Click-to-Pick & Dán Link CafeLand)
├── requirements.txt                         # Danh sách thư viện Python phụ thuộc
├── README.md                                # Hướng dẫn chi tiết & tài liệu dự án
│
├── 📁 src/ (Mã nguồn quy trình xử lý chuẩn hóa có thứ tự logic)
│   ├── 01_crawler.py                        # Script điều phối cào đa nguồn (CafeLand + Chợ Tốt)
│   ├── 01_cafeland_crawler.py               # Module cào toàn bộ 380 trang CafeLand (nhadat.cafeland.vn)
│   ├── 01_chotot_crawler.py                 # Module cào dữ liệu qua API Gateway Chợ Tốt / Nhà Tốt
│   ├── 02_spatial_poi_builder.py            # [Bước 1] Xây dựng CSDL 81 POI địa lý TP.HCM (GeoJSON)
│   ├── 03_preprocessing.py                  # [Bước 2] Tiền xử lý, khử trùng lặp, lọc ngoại lai & bóc tách sạch
│   ├── 04_spatial_features.py               # [Bước 3] Mã hóa 681 ô Uber H3 Res 8 & tính khoảng cách 81 POI
│   ├── 05_eda_visualization.py              # [Bước 4] Xuất hệ thống 8 biểu đồ EDA chuẩn khoa học (300 DPI)
│   └── 06_modeling.py                       # [Bước 5] Huấn luyện Model A vs B, Spatial CV & TreeSHAP
│
├── 📁 data/ (Dữ liệu phân theo từng giai đoạn đánh số tương ứng)
│   ├── QUY_TRINH_XU_LY_DU_LIEU.md           # Thuyết minh chi tiết đối chiếu Before/After
│   ├── 📁 raw/
│   │   ├── 01_raw_crawled_fulltext.csv      # Dữ liệu thô hợp nhất (13.839 tin: CafeLand + Chợ Tốt)
│   │   ├── 01_raw_crawled_cafeland.csv      # Dữ liệu cào độc lập từ CafeLand (6.786 tin)
│   │   └── 01_raw_crawled_chotot.csv        # Dữ liệu cào độc lập từ Chợ Tốt (7.053 tin)
│   ├── 📁 spatial_poi/
│   │   └── hcm_pois.geojson                 # CSDL 81 tiện ích (CBD, Ga Metro, BV, ĐH, TTTM, Công viên)
│   └── 📁 processed/
│       ├── 02_cleaned_structured_real_estate.csv # Bảng dữ liệu sạch (12.007 dòng, 18 cột chuẩn)
│       ├── 03_spatial_featured_real_estate.csv   # Bảng dữ liệu giàu đặc trưng không gian H3 & POI
│       ├── 04_train_spatial.csv             # Tập Train (9.605 dòng - 80%)
│       ├── 05_test_spatial.csv              # Tập Test độc lập (2.402 dòng - 20%)
│       └── du_lieu_bds_tphcm_sach_bang_bieu.xlsx # Bảng Excel định dạng tiếng Việt xem nhanh
│
├── 📁 models/ (Mô hình huấn luyện & siêu dữ liệu)
│   ├── model_a_baseline.pkl                 # Model A: Baseline (Địa giới Quận / Huyện)
│   ├── model_b_spatial.pkl                  # Model B: Proposed (H3 Grid + Spatial POI)
│   ├── shap_explainer.pkl                   # TreeSHAP Explainer
│   ├── feature_metadata.json                # Danh sách đặc trưng & cấu hình
│   └── metrics_comparison.json              # Bảng chỉ số so sánh (MAE, RMSE, R2, MAPE)
│
└── 📁 reports/ (Báo cáo & hình ảnh trực quan hóa chất lượng cao)
    ├── final_report_summary.md              # Báo cáo khoa học tóm tắt theo chuẩn 5W1H
    └── 📁 figures/
        ├── 01_price_distribution.png        # Phân phối Giá bán & Phép biến đổi Log
        ├── 02_area_vs_price_scatter.png     # Tương quan Diện tích vs Giá
        ├── 03_district_price_boxplot.png    # Biến thiên giá theo từng Quận/Huyện
        ├── 04_property_type_distribution.png# Cơ cấu Loại hình & Pháp lý
        ├── 05_correlation_heatmap.png       # Ma trận tương quan đặc trưng
        ├── 06_h3_grid_spatial_density.png   # Mật độ phân bố không gian ô lưới H3
        ├── 07_distance_to_poi_distribution.png # Tác động khoảng cách POI đến Giá
        ├── 08_rooms_floors_price_bar.png    # Đơn giá theo Phòng ngủ & Số tầng
        ├── 09_model_error_comparison.png    # Đối chiếu phân phối sai số Model A vs B
        └── 10_shap_summary_plot.png         # Diễn giải tầm quan trọng biến bằng SHAP
```

---

## 2. Hướng Dẫn Cài Đặt & Chạy Tự Động (One-Click Reproduction)

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy toàn bộ Pipeline tự động
Chỉ cần chạy **1 lệnh duy nhất** từ thư mục gốc hoặc thư mục `code/`:
```bash
python run_pipeline.py
```

### Bước 3: Khởi chạy Demo Web Ứng Dụng Streamlit
```bash
streamlit run app.py
```

---

## 3. Bảng Kết Quả Đánh Giá & So Sánh Mô Hình

Bộ dữ liệu sau làm sạch đạt quy mô **12.007 bản ghi thực tế** (Train: 9.605, Test: 2.402):

| Thước đo đánh giá | Model A (Quận/Huyện) | Model B (H3 + 81 Tiện ích GIS) | Ý nghĩa đánh giá |
| :--- | :---: | :---: | :--- |
| **MAE (Sai số tuyệt đối TB)** | **4.066 Tỷ VNĐ** | **4.105 Tỷ VNĐ** | Thước đo sai lệch tiền thực tế |
| **RMSE (Căn bậc hai MSE)** | **11.796 Tỷ VNĐ** | **11.847 Tỷ VNĐ** | Mức độ kiểm soát sai số lớn |
| **$R^2$ (Hệ số xác định)** | **0.4842** (48.42%) | **0.4797** (47.97%) | Khả năng giải thích phương sai giá |
| **MAPE (Sai số phần trăm)** | **41.25%** | **41.51%** | Sai số phần trăm tương đối |
| **5-Fold Spatial CV ($R^2$)** | *Không hỗ trợ* | **0.4290 ± 0.0475** | Kiểm thử không gian chống rò rỉ địa lý |
| **5-Fold Spatial CV (MAE)** | *Không hỗ trợ* | **4.554 ± 0.353 Tỷ** | Đánh giá tổng quát hóa ngoại suy vững chắc |

---

## 4. Khai Báo Minh Bạch Sử Dụng Công Cụ AI (AI Usage Declaration)
Theo đúng chính sách liêm chính học thuật:
- **Công cụ hỗ trợ**: Google Gemini, Python Antigravity Agent.
- **Phạm vi hỗ trợ**:
  1. Hỗ trợ xây dựng bộ cào đa luồng song song CafeLand, tối ưu hóa BeautifulSoup và Regex bóc tách.
  2. Hỗ trợ tra cứu cú pháp Uber H3 API v4, GeoPandas Haversine và SHAP Beeswarm Summary.
  3. Hỗ trợ chuẩn hóa định dạng UTF-8 CSV, export bảng tính Excel và tối ưu CSS giao diện Streamlit.
- **Cam kết liêm chính**: Toàn bộ thành viên nhóm nắm vững $100\%$ bản chất thuật toán, quy trình tiền xử lý, cơ chế Spatial Cross-Validation và sẵn sàng trả lời trực tiếp mọi câu hỏi vấn đáp của giảng viên.