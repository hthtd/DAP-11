"""
Module 01: Thu Thập Dữ Liệu Bất Động Sản Thực Tế Tại TP. Hồ Chí Minh
Nguồn dữ liệu: Cổng thông tin BĐS Nhà Tốt / Chợ Tốt Real Estate Gateway API
Quy mô mục tiêu: >= 5.000 dòng dữ liệu BĐS với đầy đủ tọa độ, giá bán, diện tích, pháp lý, thời gian đăng.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime

# Đảm bảo UTF-8 cho stdout
sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json',
    'Origin': 'https://www.nhatot.com',
    'Referer': 'https://www.nhatot.com/'
}

CATEGORIES = [
    (1010, 'Chung cư / Căn hộ'),
    (1020, 'Nhà ở / Nhà phố / Biệt thự'),
    (1040, 'Đất nền / Đất thổ cư'),
    (1030, 'Mặt bằng / Văn phòng / Kho xưởng')
]

PRICE_TIERS = [
    ('300000000-2000000000', '300tr - 2 tỷ'),
    ('2000000000-4000000000', '2 tỷ - 4 tỷ'),
    ('4000000000-7000000000', '4 tỷ - 7 tỷ'),
    ('7000000000-15000000000', '7 tỷ - 15 tỷ'),
    ('15000000000-100000000000', '15 tỷ - 100 tỷ')
]

BASE_URL = 'https://gateway.chotot.com/v1/public/ad-listing'

def fetch_real_estate_data(target_count=8000):
    os.makedirs('data/raw', exist_ok=True)
    
    all_ads = []
    seen_ids = set()
    
    print("=" * 60)
    print(f"[*] Bắt đầu cào dữ liệu MUA BÁN BĐS thực tế tại TP.HCM (Mục tiêu: >= {target_count} tin)")
    print("=" * 60)
    
    for cg_id, cg_name in CATEGORIES:
        print(f"\n[+] Đang cào danh mục {cg_id} ({cg_name})...")
        cat_count = 0
        
        for price_range, price_label in PRICE_TIERS:
            for page in range(25): # Mỗi phân khúc giá lấy tối đa 25 trang * 50 = 1250 tin
                offset = page * 50
                params = {
                    'region_v2': 13000, # TP. Hồ Chí Minh
                    'cg': cg_id,
                    'type': 's',       # Tin bán (Sale)
                    'price': price_range,
                    'limit': 50,
                    'o': offset
                }
                
                try:
                    res = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=12)
                    if res.status_code != 200:
                        break
                    
                    data = res.json()
                    ads = data.get('ads', [])
                    if not ads:
                        break
                    
                    new_in_page = 0
                    for ad in ads:
                        list_id = ad.get('list_id')
                        price_val = ad.get('price')
                        if list_id and price_val and list_id not in seen_ids:
                            # Đảm bảo tin bán có giá >= 300 triệu
                            if price_val >= 300_000_000:
                                seen_ids.add(list_id)
                                
                                record = {
                                    'list_id': list_id,
                                    'subject': ad.get('subject', ''),
                                    'body': ad.get('body', ''),
                                    'category_id': cg_id,
                                    'category_name': cg_name,
                                    'price': price_val,
                                    'price_string': ad.get('price_string', ''),
                                    'price_million_per_m2': ad.get('price_million_per_m2'),
                                    'size': ad.get('size'),
                                    'living_size': ad.get('living_size'),
                                    'rooms': ad.get('rooms'),
                                    'toilets': ad.get('toilets'),
                                    'floors': ad.get('floors'),
                                    'house_type': ad.get('house_type'),
                                    'property_legal_document': ad.get('property_legal_document'),
                                    'direction': ad.get('direction'),
                                    'latitude': ad.get('latitude'),
                                    'longitude': ad.get('longitude'),
                                    'street_name': ad.get('street_name', ''),
                                    'ward_name': ad.get('ward_name', ''),
                                    'area_name': ad.get('area_name', ''), # Quận/Huyện
                                    'region_name': ad.get('region_name', 'Tp Hồ Chí Minh'),
                                    'list_time': ad.get('list_time'),
                                    'orig_list_time': ad.get('orig_list_time'),
                                    'is_company_ad': ad.get('company_ad', False),
                                    'deposit': ad.get('deposit')
                                }
                                all_ads.append(record)
                                new_in_page += 1
                                cat_count += 1
                    
                    time.sleep(0.15) # Giãn cách nhẹ
                    
                    if len(ads) < 50:
                        break
                except Exception as e:
                    time.sleep(0.5)
                    break
            
            print(f"    -> Phân khúc {price_label}: Đã lấy | Tổng tích lũy: {len(all_ads)}")
            if len(all_ads) >= target_count:
                break
                
        print(f"    ==> Hoàn thành danh mục {cg_name}: {cat_count} tin bán hợp lệ.")
        if len(all_ads) >= target_count:
            break

    df = pd.DataFrame(all_ads)
    print("\n" + "=" * 60)
    print(f"[*] Thu thập hoàn tất! Tổng số bản ghi thực tế: {len(df)}")
    print("=" * 60)
    
    # Lưu định dạng CSV
    csv_path = 'data/raw/raw_real_estate_hcm.csv'
    csv_fulltext_path = 'data/raw/01_raw_crawled_fulltext.csv'
    
    # Khử newline trong các trường text để tránh vỡ định dạng CSV
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.replace('\r', ' ', regex=False).str.replace('\n', ' ', regex=False).str.strip()
        
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df.to_csv(csv_fulltext_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Đã lưu file CSV: {csv_path} ({os.path.getsize(csv_path)/1024/1024:.2f} MB)")
    print(f"[OK] Đã lưu file CSV: {csv_fulltext_path}")
    
    return df

if __name__ == '__main__':
    fetch_real_estate_data(target_count=6500)
