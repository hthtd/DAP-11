"""
Script Chạy Toàn Bộ Pipeline Dự Án (End-to-End Master Pipeline)
Thực hiện tự động theo thứ tự:
1. Thu thập dữ liệu BĐS TP.HCM (01_crawler.py)
2. Khởi tạo 81 điểm tiện ích đô thị GeoJSON (02_spatial_poi_builder.py)
3. Tiền xử lý, làm sạch và chia tập dữ liệu (03_preprocessing.py)
4. Tích hợp đặc trưng không gian vi mô Uber H3 Grid & POI (04_spatial_features.py)
5. Xuất hệ thống biểu đồ phân tích EDA 300 DPI (05_eda_visualization.py)
6. Huấn luyện Model A vs B, đánh giá Spatial CV & SHAP (06_modeling.py)
"""

import os
import sys
import time
import subprocess

# Tự động chuyển working directory về thư mục chứa code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')

STEPS = [
    ("Bước 1: Khởi tạo 81 Điểm Tiện ích Đô thị POI", "src/02_spatial_poi_builder.py"),
    ("Bước 2: Tiền xử lý, Làm sạch & Chia tập Dữ liệu", "src/03_preprocessing.py"),
    ("Bước 3: Tích hợp Đặc trưng Không gian H3 Grid & POI", "src/04_spatial_features.py"),
    ("Bước 4: Xuất Hệ Thống 8 Biểu Đồ EDA Đạt Chuẩn", "src/05_eda_visualization.py"),
    ("Bước 5: Huấn luyện Mô hình, Đánh giá Spatial CV & SHAP", "src/06_modeling.py"),
]

def run_pipeline():
    start_total = time.time()
    print("=" * 70)
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE DỰ ÁN CAFELAND PROPTECH AI")
    print("=" * 70)
    
    for idx, (title, script) in enumerate(STEPS, start=1):
        print(f"\n[{idx}/{len(STEPS)}] >>> {title} ({script})")
        t0 = time.time()
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            print(f"❌ LỖI tại {script} (Mã lỗi: {result.returncode})! Dừng pipeline.")
            sys.exit(1)
        print(f"✅ Hoàn thành trong {time.time() - t0:.2f} giây.")
        
    print("\n" + "=" * 70)
    print(f"🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {time.time() - start_total:.2f} GIÂY!")
    print("👉 Để khởi chạy ứng dụng Demo Web, hãy chạy lệnh:")
    print("   streamlit run app.py")
    print("=" * 70)

if __name__ == '__main__':
    run_pipeline()
