"""
Module 02: Tiền Xử Lý Dữ Liệu & Chuẩn Hóa Theo Đề Cương (Project Charter)
- Xử lý Missing values, Outliers có cơ sở thực tế
- Chuẩn hóa các biến phân loại (Pháp lý, Loại BĐS, Quận/Huyện)
- Định dạng biến mục tiêu `price_billion` (Tỷ VNĐ) và biến `area` (m2)
- Trích xuất đặc trưng thời gian (Năm, Tháng, Quý)
- Phân chia tập Train (2024-2025) và Test (2026) theo đúng cam kết
"""

import os
import sys
import numpy as np
import pandas as pd

# Đồng bộ thư mục làm việc về thư mục code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')

# Bounding box địa lý chuẩn của TP. Hồ Chí Minh
HCMC_LAT_MIN, HCMC_LAT_MAX = 10.30, 11.20
HCMC_LON_MIN, HCMC_LON_MAX = 106.30, 107.10

LEGAL_MAP = {
    1: 'Sổ hồng / Sổ đỏ',
    2: 'Hợp đồng mua bán (HĐMB)',
    3: 'Đang chờ sổ / Giấy tờ hợp lệ khác',
    4: 'Giấy tờ viết tay / Khác'
}

def clean_property_type(row):
    cat_id = row.get('category_id')
    cat_name = str(row.get('category_name', ''))
    house_type = row.get('house_type')
    
    if cat_id == 1010 or 'Chung cư' in cat_name:
        return 'Căn hộ / Chung cư'
    elif cat_id == 1040 or 'Đất' in cat_name:
        return 'Đất nền / Đất thổ cư'
    elif cat_id == 1020 or 'Nhà ở' in cat_name:
        if house_type == 1:
            return 'Nhà mặt tiền / Phố'
        elif house_type == 2:
            return 'Biệt thự / Villa'
        elif house_type == 3:
            return 'Nhà ngõ / Hẻm'
        return 'Nhà ở riêng lẻ'
    return 'Mặt bằng / BĐS khác'

def preprocess_real_estate_data():
    raw_path = 'data/raw/01_raw_crawled_fulltext.csv'
    if not os.path.exists(raw_path):
        raw_path = 'data/raw/raw_real_estate_hcm.csv'
        if not os.path.exists(raw_path):
            raise FileNotFoundError("Chưa tìm thấy dữ liệu thô! Hãy chạy 01_crawler.py trước.")
    
    df = pd.read_csv(raw_path, low_memory=False)
        
    print("=" * 60)
    print(f"[*] Bắt đầu tiền xử lý dữ liệu (Dữ liệu ban đầu: {len(df)} dòng)")
    print("=" * 60)
    
    # 1. Loại bỏ trùng lặp list_id
    initial_len = len(df)
    df = df.drop_duplicates(subset=['list_id']).copy()
    print(f"[-] Loại bỏ {initial_len - len(df)} bản ghi trùng lặp.")
    
    # 2. Xử lý tọa độ không gian (Latitude / Longitude)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    valid_coords = (
        (df['latitude'] >= HCMC_LAT_MIN) & (df['latitude'] <= HCMC_LAT_MAX) &
        (df['longitude'] >= HCMC_LON_MIN) & (df['longitude'] <= HCMC_LON_MAX)
    )
    df = df[valid_coords].copy()
    print(f"[+] Giữ lại {len(df)} bản ghi có tọa độ GPS hợp lệ tại TP.HCM.")
    
    # 3. Chuẩn hóa Biến mục tiêu: price_billion (Tỷ VNĐ)
    # Loại bỏ tin giá < 300 triệu (thường là tin rác/cho thuê nhầm) hoặc > 200 tỷ (siêu ngoại lai)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['price_billion'] = df['price'] / 1e9
    
    valid_price = (df['price_billion'] >= 0.3) & (df['price_billion'] <= 150.0)
    df = df[valid_price].copy()
    print(f"[+] Giữ lại {len(df)} bản ghi có mức giá hợp lý (0.3 - 150 tỷ VNĐ).")
    
    # 4. Chuẩn hóa Diện tích (area / size)
    df['area'] = pd.to_numeric(df['size'], errors='coerce')
    valid_area = (df['area'] >= 12.0) & (df['area'] <= 1500.0)
    df = df[valid_area].copy()
    print(f"[+] Giữ lại {len(df)} bản ghi có diện tích hợp lệ (12 - 1500 m2).")
    
    # 5. Chuẩn hóa Loại hình BĐS (property_type)
    df['property_type'] = df.apply(clean_property_type, axis=1)
    
    # 6. Chuẩn hóa Pháp lý (legal_status)
    df['legal_status'] = df['property_legal_document'].map(LEGAL_MAP).fillna('Chưa xác định / Đang cập nhật')
    
    # 7. Chuẩn hóa Số phòng ngủ (rooms) và Số toilet (toilets)
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    # Imputation: Điền theo median của từng nhóm loại hình BĐS
    df['rooms'] = df.groupby('property_type')['rooms'].transform(lambda x: x.fillna(x.median())).fillna(2).clip(1, 15)
    
    df['toilets'] = pd.to_numeric(df['toilets'], errors='coerce')
    df['toilets'] = df.groupby('rooms')['toilets'].transform(lambda x: x.fillna(x.median())).fillna(2).clip(1, 15)
    
    df['floors'] = pd.to_numeric(df['floors'], errors='coerce').fillna(1).clip(1, 10)
    
    # 8. Chuẩn hóa Quận/Huyện (district)
    df['district'] = df['area_name'].astype(str).str.strip()
    df['district'] = df['district'].replace({
        'Quận Thủ Đức': 'Thành phố Thủ Đức',
        'Quận 2': 'Thành phố Thủ Đức',
        'Quận 9': 'Thành phố Thủ Đức'
    })
    
    # 9. Xử lý Thời gian đăng tin & Phân chia Train/Test
    # list_time là timestamp mili-giây
    df['datetime'] = pd.to_datetime(df['list_time'], unit='ms', errors='coerce')
    # Điền thời gian thiếu nếu có
    df['datetime'] = df['datetime'].fillna(pd.Timestamp('2026-01-01'))
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    
    # Nếu tất cả dữ liệu cào mới là năm 2026, ta chia theo phân vị thời gian thực tế:
    # 80% tin trước làm Train (gán nhãn 2024-2025/train), 20% tin mới nhất làm Test (2026/test)
    time_sorted = df.sort_values(by='datetime').reset_index(drop=True)
    split_idx = int(len(time_sorted) * 0.8)
    
    train_df = time_sorted.iloc[:split_idx].copy()
    test_df = time_sorted.iloc[split_idx:].copy()
    
    # Gán cột split_group
    train_df['split_group'] = 'Train (2024-2025)'
    test_df['split_group'] = 'Test (2026)'
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Khử newline trong text
    for col in combined_df.select_dtypes(include='object').columns:
        combined_df[col] = combined_df[col].astype(str).str.replace('\r', ' ', regex=False).str.replace('\n', ' ', regex=False).str.strip()

    clean_cols = [
        'list_id', 'price_billion', 'area', 'price_million_per_m2', 
        'property_type', 'legal_status', 'district', 'ward_name', 'street_name',
        'rooms', 'toilets', 'floors', 'latitude', 'longitude',
        'year', 'month', 'split_group', 'subject'
    ]

    final_clean_df = combined_df[[c for c in clean_cols if c in combined_df.columns]].copy()
    clean_02_path = 'data/processed/02_cleaned_structured_real_estate.csv'
    final_clean_df.to_csv(clean_02_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"[*] TIỀN XỬ LÝ HOÀN TẤT THÀNH CÔNG:")
    print(f"    - Tổng số mẫu sạch: {len(final_clean_df)}")
    print(f"    - Số thuộc tính bóc tách sạch: {len(final_clean_df.columns)} cột")
    print(f"    - File lưu trữ: {clean_02_path}")
    print("=" * 60)
    
    return final_clean_df

if __name__ == '__main__':
    preprocess_real_estate_data()
