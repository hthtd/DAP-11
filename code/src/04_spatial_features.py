"""
Module 03: Xây Dựng Đặc Trưng Không Gian (Spatial Features & Spatial CV)
- Mã hóa lưới Uber H3 Grid (Resolution 8: đường kính ~460m, diện tích ~0.74 km2)
- Tính khoảng cách không gian (Haversine/GeoPandas) tới:
  1. CBD Quận 1 (Trung tâm kinh tế)
  2. Ga Metro Tuyến số 1 gần nhất (Bến Thành - Suối Tiên)
  3. Bệnh viện đa khoa / chuyên khoa lớn gần nhất
  4. Trường Đại học trọng điểm gần nhất
- Thiết lập Spatial Cross-Validation Groups dựa trên H3 Hexagon để chống rò rỉ dữ liệu (Spatial Data Leakage)
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import h3

# Đồng bộ thư mục làm việc về thư mục code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')

# Haversine distance tính khoảng cách giữa 2 tọa độ (km)
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0 # Bán kính Trái Đất (km)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_spatial_features():
    data_path = 'data/processed/02_cleaned_structured_real_estate.csv'
    if not os.path.exists(data_path):
        data_path = 'data/processed/processed_real_estate_hcm.csv'
    poi_path = 'data/spatial_poi/hcm_pois.geojson'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Chưa tìm thấy dữ liệu đã tiền xử lý! Hãy chạy 02_preprocessing.py trước.")
    if not os.path.exists(poi_path):
        raise FileNotFoundError("Chưa tìm thấy dữ liệu POI! Hãy chạy spatial_poi_builder.py trước.")
        
    df = pd.read_csv(data_path, low_memory=False)
    poi_gdf = gpd.read_file(poi_path)
    
    print("=" * 60)
    print(f"[*] Bắt đầu tạo đặc trưng không gian (Spatial Enrichment) cho {len(df)} BĐS...")
    print("=" * 60)
    
    # 1. Mã hóa Uber H3 Grid (Resolution 8)
    # H3 v4 API: latlng_to_cell(lat, lng, res)
    def get_h3_index(row):
        try:
            return h3.latlng_to_cell(row['latitude'], row['longitude'], 8)
        except Exception:
            # Fallback nếu h3 v3: geo_to_h3
            try:
                return h3.geo_to_h3(row['latitude'], row['longitude'], 8)
            except Exception:
                return 'unknown'
                
    df['h3_index'] = df.apply(get_h3_index, axis=1)
    n_hexagons = df['h3_index'].nunique()
    print(f"[+] Đã mã hóa {len(df)} BĐS vào {n_hexagons} ô lưới lục giác H3 Grid (Res 8).")
    
    # 2. Tách nhóm POI theo danh mục
    cbd_pois = poi_gdf[poi_gdf['type'] == 'CBD']
    metro_pois = poi_gdf[poi_gdf['type'] == 'Metro']
    hospital_pois = poi_gdf[poi_gdf['type'] == 'Hospital']
    uni_pois = poi_gdf[poi_gdf['type'] == 'University']
    
    # Tọa độ CBD Q1
    cbd_lat = cbd_pois.iloc[0].geometry.y
    cbd_lon = cbd_pois.iloc[0].geometry.x
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    # 2.1 Khoảng cách tới CBD Q1
    df['dist_cbd_km'] = haversine_vectorized(lats, lons, cbd_lat, cbd_lon)
    
    # 2.2 Khoảng cách tới Ga Metro gần nhất
    metro_lats = np.array([p.y for p in metro_pois.geometry])
    metro_lons = np.array([p.x for p in metro_pois.geometry])
    # Ma trận khoảng cách: (N_samples, N_metros)
    dist_metro_matrix = np.array([haversine_vectorized(lats, lons, m_lat, m_lon) for m_lat, m_lon in zip(metro_lats, metro_lons)])
    df['dist_nearest_metro_km'] = np.min(dist_metro_matrix, axis=0)
    
    # 2.3 Khoảng cách tới Bệnh viện lớn gần nhất
    hosp_lats = np.array([p.y for p in hospital_pois.geometry])
    hosp_lons = np.array([p.x for p in hospital_pois.geometry])
    dist_hosp_matrix = np.array([haversine_vectorized(lats, lons, h_lat, h_lon) for h_lat, h_lon in zip(hosp_lats, hosp_lons)])
    df['dist_nearest_hospital_km'] = np.min(dist_hosp_matrix, axis=0)
    
    # 2.4 Khoảng cách tới Trường Đại học lớn gần nhất
    uni_lats = np.array([p.y for p in uni_pois.geometry])
    uni_lons = np.array([p.x for p in uni_pois.geometry])
    dist_uni_matrix = np.array([haversine_vectorized(lats, lons, u_lat, u_lon) for u_lat, u_lon in zip(uni_lats, uni_lons)])
    df['dist_nearest_university_km'] = np.min(dist_uni_matrix, axis=0)
    
    # 3. Tạo Spatial CV Folds (5 Folds dựa trên H3 Grid)
    unique_h3 = df['h3_index'].unique()
    np.random.seed(42)
    permuted_h3 = np.random.permutation(unique_h3)
    h3_fold_map = {h3_val: int(idx % 5) for idx, h3_val in enumerate(permuted_h3)}
    df['spatial_fold'] = df['h3_index'].map(h3_fold_map)
    
    # Khử newline trong text
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False).str.replace('\n', ' ', regex=False).str.strip()

    # Đưa các đặc trưng không gian (H3, POI distances) lên phía trước để mở file là thấy ngay sự khác biệt
    spatial_cols = [
        'list_id', 'price_billion', 'area', 'price_million_per_m2',
        'property_type', 'district', 'h3_index', 
        'dist_cbd_km', 'dist_nearest_metro_km', 'dist_nearest_hospital_km', 'dist_nearest_university_km',
        'rooms', 'toilets', 'floors', 'legal_status', 'ward_name', 'street_name',
        'latitude', 'longitude', 'year', 'month', 'split_group', 'spatial_fold', 'subject'
    ]
    spatial_df = df[[c for c in spatial_cols if c in df.columns]].copy()

    # 4. Lưu lại dữ liệu giàu đặc trưng không gian (CSV & Excel)
    spatial_03_path = 'data/processed/03_spatial_featured_real_estate.csv'
    spatial_df.to_csv(spatial_03_path, index=False, encoding='utf-8-sig')
    
    # Cập nhật lại train/test theo spatial data
    train_df = spatial_df[spatial_df['split_group'] == 'Train (2024-2025)'].copy()
    test_df = spatial_df[spatial_df['split_group'] == 'Test (2026)'].copy()
    
    train_df.to_csv('data/processed/04_train_spatial.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('data/processed/05_test_spatial.csv', index=False, encoding='utf-8-sig')

    # Lưu thêm 1 bản Excel trực quan
    columns_readable = [
        'list_id', 'subject', 'price_billion', 'area', 'price_million_per_m2',
        'rooms', 'toilets', 'floors', 'property_type', 'legal_status',
        'district', 'ward_name', 'street_name', 'latitude', 'longitude',
        'h3_index', 'dist_cbd_km', 'dist_nearest_metro_km', 'dist_nearest_hospital_km', 'dist_nearest_university_km',
        'year', 'month', 'split_group'
    ]
    df_clean_vn = spatial_df[[c for c in columns_readable if c in spatial_df.columns]].rename(columns={
        'list_id': 'Ma_Tin', 'subject': 'Tieu_De', 'price_billion': 'Gia_Ty_VND',
        'area': 'Dien_Tich_m2', 'price_million_per_m2': 'Gia_Trieu_m2',
        'rooms': 'So_Phong_Ngu', 'toilets': 'So_Toilet', 'floors': 'So_Tang',
        'property_type': 'Loai_BDS', 'legal_status': 'Phap_Ly',
        'district': 'Quan_Huyen', 'ward_name': 'Phuong_Xa', 'street_name': 'Ten_Duong',
        'latitude': 'Vi_Do', 'longitude': 'Kinh_Do', 'h3_index': 'Ma_Uber_H3',
        'dist_cbd_km': 'KC_Trung_Tam_Q1_km', 'dist_nearest_metro_km': 'KC_Ga_Metro_km',
        'dist_nearest_hospital_km': 'KC_Benh_Vien_km', 'dist_nearest_university_km': 'KC_Dai_Hoc_km',
        'year': 'Nam_Dang', 'month': 'Thang_Dang', 'split_group': 'Tap_Du_Lieu'
    })
    df_clean_vn.to_excel('data/processed/du_lieu_bds_tphcm_sach_bang_bieu.xlsx', index=False)
    
    print("\n" + "=" * 60)
    print(f"[*] HOÀN THÀNH XÂY DỰNG ĐẶC TRƯNG KHÔNG GIAN:")
    print(f"    - Khoảng cách TB tới CBD Q1: {df['dist_cbd_km'].mean():.2f} km (Min: {df['dist_cbd_km'].min():.2f}, Max: {df['dist_cbd_km'].max():.2f})")
    print(f"    - Khoảng cách TB tới Ga Metro: {df['dist_nearest_metro_km'].mean():.2f} km")
    print(f"    - Khoảng cách TB tới Bệnh viện: {df['dist_nearest_hospital_km'].mean():.2f} km")
    print(f"    - Khoảng cách TB tới Trường ĐH: {df['dist_nearest_university_km'].mean():.2f} km")
    print(f"    - Đã phân bổ {n_hexagons} ô H3 vào 5 Spatial CV Folds.")
    print(f"    - File kết quả: {spatial_03_path}")
    print("=" * 60)
    
    return df

if __name__ == '__main__':
    compute_spatial_features()
