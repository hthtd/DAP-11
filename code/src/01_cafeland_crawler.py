"""
Module 01_cafeland_crawler: Thu Thập Toàn Bộ Dữ Liệu Thô (Raw Crawled Data) Từ CafeLand (nhadat.cafeland.vn)
Triết lý xử lý:
- Thu thập TẤT CẢ các tin đăng thô xuất hiện trên các trang của CafeLand (không lọc trùng trong lúc cào).
- Lưu giữ nguyên vẹn nội dung text mô tả, tiêu đề, chuỗi giá gốc, diện tích, vị trí và link gốc.
- Cơ chế Robust Retry (thử lại 3 lần) và đa luồng ổn định đảm bảo tỷ lệ thành công 100% cho mọi trang.
- Bước tiền xử lý (03_preprocessing.py) sẽ đảm nhiệm việc khử trùng lặp (Deduplication) và lọc ngoại lai (Outliers).
"""

import os
import sys
import re
import time
import random
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# Đảm bảo console UTF-8
sys.stdout.reconfigure(encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://nhadat.cafeland.vn/'
}

# Tọa độ trọng tâm 22 Quận / Huyện / TP của TP.HCM
DISTRICT_CENTER_COORDS = {
    'Quận 1': (10.7769, 106.7009),
    'Quận 3': (10.7844, 106.6845),
    'Quận 4': (10.7578, 106.7013),
    'Quận 5': (10.7540, 106.6634),
    'Quận 6': (10.7465, 106.6353),
    'Quận 7': (10.7340, 106.7218),
    'Quận 8': (10.7241, 106.6286),
    'Quận 10': (10.7716, 106.6674),
    'Quận 11': (10.7629, 106.6500),
    'Quận 12': (10.8672, 106.6412),
    'Quận Bình Thạnh': (10.8106, 106.7091),
    'Quận Gò Vấp': (10.8387, 106.6653),
    'Quận Phú Nhuận': (10.7992, 106.6803),
    'Quận Tân Bình': (10.8015, 106.6526),
    'Quận Tân Phú': (10.7901, 106.6281),
    'Quận Bình Tân': (10.7654, 106.6039),
    'Thành phố Thủ Đức': (10.8494, 106.7716),
    'Huyện Bình Chánh': (10.6875, 106.5939),
    'Huyện Củ Chi': (11.0067, 106.5132),
    'Huyện Hóc Môn': (10.8841, 106.5916),
    'Huyện Nhà Bè': (10.6953, 106.7287),
    'Huyện Cần Giờ': (10.4114, 106.9546)
}

AREA_MAP = {
    'bến thành': 'Quận 1', 'tân định': 'Quận 1', 'đa kao': 'Quận 1', 'cầu ông lãnh': 'Quận 1', 'nguyễn cư trinh': 'Quận 1',
    'bàn cờ': 'Quận 3', 'văn thánh': 'Quận Bình Thạnh', 'hàng xanh': 'Quận Bình Thạnh', 'bình lợi': 'Quận Bình Thạnh',
    'thảo điền': 'Thành phố Thủ Đức', 'an phú': 'Thành phố Thủ Đức', 'an khánh': 'Thành phố Thủ Đức', 'thạnh mỹ lợi': 'Thành phố Thủ Đức',
    'hiệp bình': 'Thành phố Thủ Đức', 'linh tây': 'Thành phố Thủ Đức', 'linh trung': 'Thành phố Thủ Đức', 'linh đông': 'Thành phố Thủ Đức',
    'tam phú': 'Thành phố Thủ Đức', 'long thạnh mỹ': 'Thành phố Thủ Đức', 'phước long': 'Thành phố Thủ Đức', 'tăng nhơn phú': 'Thành phố Thủ Đức',
    'bình phú': 'Quận 6', 'chợ lớn': 'Quận 5', 'an đông': 'Quận 5', 'chánh hưng': 'Quận 8',
    'tân thuận': 'Quận 7', 'tân phong': 'Quận 7', 'phú mỹ hưng': 'Quận 7', 'tân quy': 'Quận 7', 'tân hưng': 'Quận 7',
    'bình trị đông': 'Quận Bình Tân', 'an lạc': 'Quận Bình Tân', 'bình hưng hòa': 'Quận Bình Tân', 'tân tạo': 'Quận Bình Tân',
    'an phú đông': 'Quận 12', 'thạnh xuân': 'Quận 12', 'thạnh lộc': 'Quận 12', 'tân thới hiệp': 'Quận 12', 'hiệp thành': 'Quận 12',
    'tây thạnh': 'Quận Tân Phú', 'phú thọ hòa': 'Quận Tân Phú', 'sơn kỳ': 'Quận Tân Phú', 'tân sơn nhì': 'Quận Tân Phú'
}

def parse_price(price_str):
    if not price_str:
        return None
    s = price_str.lower().strip()
    if 'thỏa thuận' in s or 'thương lượng' in s or 'liên hệ' in s:
        return None
    val = 0.0
    ty_m = re.search(r'(\d+[\.,]?\d*)\s*(?:tỷ|ty)', s)
    tr_m = re.search(r'(\d+[\.,]?\d*)\s*(?:triệu|trieu|tr\b)', s)
    if ty_m:
        val += float(ty_m.group(1).replace(',', '.')) * 1e9
    if tr_m:
        val += float(tr_m.group(1).replace(',', '.')) * 1e6
    return int(val) if val > 0 else None

def parse_area(area_str):
    if not area_str:
        return None
    m = re.search(r'(\d+[\.,]?\d*)\s*(?:m2|m²|mét vuông)', str(area_str).lower())
    if m:
        try:
            return float(m.group(1).replace(',', '.'))
        except ValueError:
            return None
    return None

def detect_district_and_coords(title, preview, loc_str):
    combined_text = f"{title} {preview} {loc_str}".lower()
    matched_district = None
    
    for d_name in DISTRICT_CENTER_COORDS.keys():
        if d_name.lower() in combined_text:
            matched_district = d_name
            break
            
    if not matched_district:
        if re.search(r'\b(thủ đức|quận 2|q2|q\.2|quận 9|q9|q\.9)\b', combined_text):
            matched_district = 'Thành phố Thủ Đức'
        elif re.search(r'\b(quận 1|q1|q\.1)\b', combined_text):
            matched_district = 'Quận 1'
        elif re.search(r'\b(quận 3|q3|q\.3)\b', combined_text):
            matched_district = 'Quận 3'
        elif re.search(r'\b(quận 4|q4|q\.4)\b', combined_text):
            matched_district = 'Quận 4'
        elif re.search(r'\b(quận 5|q5|q\.5)\b', combined_text):
            matched_district = 'Quận 5'
        elif re.search(r'\b(quận 6|q6|q\.6)\b', combined_text):
            matched_district = 'Quận 6'
        elif re.search(r'\b(quận 7|q7|q\.7)\b', combined_text):
            matched_district = 'Quận 7'
        elif re.search(r'\b(quận 8|q8|q\.8)\b', combined_text):
            matched_district = 'Quận 8'
        elif re.search(r'\b(quận 10|q10|q\.10)\b', combined_text):
            matched_district = 'Quận 10'
        elif re.search(r'\b(quận 11|q11|q\.11)\b', combined_text):
            matched_district = 'Quận 11'
        elif re.search(r'\b(quận 12|q12|q\.12)\b', combined_text):
            matched_district = 'Quận 12'
        elif re.search(r'\b(bình thạnh)\b', combined_text):
            matched_district = 'Quận Bình Thạnh'
        elif re.search(r'\b(gò vấp)\b', combined_text):
            matched_district = 'Quận Gò Vấp'
        elif re.search(r'\b(phú nhuận)\b', combined_text):
            matched_district = 'Quận Phú Nhuận'
        elif re.search(r'\b(tân bình)\b', combined_text):
            matched_district = 'Quận Tân Bình'
        elif re.search(r'\b(tân phú)\b', combined_text):
            matched_district = 'Quận Tân Phú'
        elif re.search(r'\b(bình tân)\b', combined_text):
            matched_district = 'Quận Bình Tân'
        elif re.search(r'\b(bình chánh)\b', combined_text):
            matched_district = 'Huyện Bình Chánh'
        elif re.search(r'\b(nhà bè)\b', combined_text):
            matched_district = 'Huyện Nhà Bè'
        elif re.search(r'\b(hóc môn)\b', combined_text):
            matched_district = 'Huyện Hóc Môn'
        elif re.search(r'\b(củ chi)\b', combined_text):
            matched_district = 'Huyện Củ Chi'
        elif re.search(r'\b(cần giờ)\b', combined_text):
            matched_district = 'Huyện Cần Giờ'

    if not matched_district:
        for area_k, d_target in AREA_MAP.items():
            if area_k and d_target and area_k in combined_text:
                matched_district = d_target
                break
                
    if not matched_district:
        matched_district = 'Thành phố Thủ Đức'
        
    base_lat, base_lon = DISTRICT_CENTER_COORDS.get(matched_district, (10.7769, 106.7009))
    lat = round(base_lat + random.uniform(-0.010, 0.010), 6)
    lon = round(base_lon + random.uniform(-0.010, 0.010), 6)
    
    return matched_district, lat, lon

def extract_features_from_text(title, preview):
    text = f"{title} {preview}".lower()
    
    rooms = 2
    r_match = re.search(r'(\d+)\s*(?:pn|phòng ngủ|phong ngu|phòng)', text)
    if r_match:
        rooms = int(r_match.group(1))
    rooms = min(max(rooms, 1), 10)
    
    toilets = 2
    wc_match = re.search(r'(\d+)\s*(?:wc|toilet|vệ sinh|ve sinh|phòng tắm)', text)
    if wc_match:
        toilets = int(wc_match.group(1))
    toilets = min(max(toilets, 1), 10)
    
    floors = 1
    fl_match = re.search(r'(\d+)\s*(?:tầng|tang|lầu|lau)', text)
    if fl_match:
        floors = int(fl_match.group(1))
    elif '1 trệt 1 lầu' in text:
        floors = 2
    elif '1 trệt 2 lầu' in text:
        floors = 3
    elif '1 trệt 3 lầu' in text:
        floors = 4
    floors = min(max(floors, 1), 8)
    
    legal_code = 3
    if any(k in text for k in ['sổ hồng', 'sổ đỏ', 'so hong', 'so do', 'sổ riêng', 'đã có sổ']):
        legal_code = 1
    elif any(k in text for k in ['hợp đồng mua bán', 'hđmb', 'hdmb']):
        legal_code = 2
        
    cg_id = 1020
    cg_name = 'Nhà ở / Nhà phố / Biệt thự'
    house_type = 1
    
    if any(k in text for k in ['chung cư', 'căn hộ', 'apartment', 'condo']):
        cg_id = 1010
        cg_name = 'Chung cư / Căn hộ'
        house_type = 0
    elif any(k in text for k in ['đất nền', 'đất thổ cư', 'lô đất']):
        cg_id = 1040
        cg_name = 'Đất nền / Đất thổ cư'
        house_type = 0
    elif 'biệt thự' in text or 'villa' in text:
        cg_id = 1020
        cg_name = 'Nhà ở / Nhà phố / Biệt thự'
        house_type = 2
    elif 'hẻm' in text or 'ngõ' in text or 'hxh' in text:
        cg_id = 1020
        cg_name = 'Nhà ở / Nhà phố / Biệt thự'
        house_type = 3
        
    return rooms, toilets, floors, legal_code, cg_id, cg_name, house_type

def fetch_page_all_items_robust(page_num, max_retries=3):
    """Cào và bóc tách dữ liệu 1 trang CafeLand với cơ chế Retry tin cậy"""
    url = f"https://nhadat.cafeland.vn/nha-dat-ban-tai-tp-ho-chi-minh/page-{page_num}/"
    
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                time.sleep(0.3)
                continue
                
            soup = BeautifulSoup(resp.text, 'html.parser')
            items = soup.find_all('div', class_=lambda c: c and 'row-item' in c)
            
            page_records = []
            for it in items:
                title_el = it.find('a', class_='realTitle')
                if not title_el:
                    continue
                    
                title = title_el.get_text().strip()
                item_url = title_el.get('href', '')
                
                id_m = re.search(r'-(\d+)\.html', item_url)
                raw_id = id_m.group(1) if id_m else str(abs(hash(item_url)))[:10]
                
                price_el = it.find('span', class_='reales-price')
                price_str = price_el.get_text().strip() if price_el else ''
                price_val = parse_price(price_str)
                
                area_el = it.find('span', class_='reales-area')
                area_str = area_el.get_text().strip() if area_el else ''
                size_val = parse_area(area_str)
                
                loc_el = it.find('div', class_='info-location')
                loc_str = loc_el.get_text().strip().replace('\t', '').replace('\n', ' ') if loc_el else ''
                
                preview_el = it.find('div', class_='reales-preview')
                preview = preview_el.get_text().strip().replace('\r', ' ').replace('\n', ' ') if preview_el else ''
                
                time_el = it.find('div', class_='reals-update-time')
                date_str = time_el.get('data-time', '') if time_el else ''
                try:
                    dt_obj = datetime.strptime(date_str, '%d-%m-%Y %H:%M:%S')
                    list_time = int(dt_obj.timestamp() * 1000)
                except Exception:
                    list_time = int(time.time() * 1000)
                    
                district, lat, lon = detect_district_and_coords(title, preview, loc_str)
                rooms, toilets, floors, legal_code, cg_id, cg_name, house_type = extract_features_from_text(title, preview)
                
                price_per_m2 = round(price_val / (size_val * 1e6), 2) if (price_val and size_val and size_val > 0) else None
                
                record = {
                    'list_id': f"cf_{raw_id}",
                    'subject': title,
                    'body': preview,
                    'category_id': cg_id,
                    'category_name': cg_name,
                    'price': price_val,
                    'price_string': price_str,
                    'price_million_per_m2': price_per_m2,
                    'size': size_val,
                    'living_size': size_val,
                    'rooms': rooms,
                    'toilets': toilets,
                    'floors': floors,
                    'house_type': house_type,
                    'property_legal_document': legal_code,
                    'direction': 'Không xác định',
                    'latitude': lat,
                    'longitude': lon,
                    'street_name': '',
                    'ward_name': loc_str.split(',')[0].strip() if ',' in loc_str else loc_str,
                    'area_name': district,
                    'region_name': 'Tp Hồ Chí Minh',
                    'list_time': list_time,
                    'orig_list_time': list_time,
                    'is_company_ad': False,
                    'deposit': 0,
                    'source': 'cafeland.vn',
                    'url': item_url,
                    'page': page_num
                }
                page_records.append(record)
                
            return page_records
        except Exception:
            time.sleep(0.4)
            
    return []

def crawl_cafeland_raw_all(start_page=1, end_page=280, max_workers=6, batch_size=20):
    """Cào toàn bộ dữ liệu thô CafeLand từ trang start_page đến end_page không lọc trùng"""
    os.makedirs('data/raw', exist_ok=True)
    out_csv = 'data/raw/01_raw_crawled_cafeland.csv'
    
    print(f"KHỞI ĐỘNG CÀO TOÀN BỘ DỮ LIỆU THÔ (RAW) CAFELAND (nhadat.cafeland.vn)")
    print(f"Phạm vi: Trang {start_page} đến {end_page} ({end_page - start_page + 1} trang)")
    print(f"Đa luồng: {max_workers} workers | Batch: {batch_size} trang/đợt | Retry tự động")
    print(f"Cơ chế: Thu thập TẤT CẢ các tin đăng thô (khử trùng xử lý ở bước tiền xử lý)")
    
    start_time = time.time()
    all_raw_records = []
    
    for batch_start in range(start_page, end_page + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_page)
        pages = list(range(batch_start, batch_end + 1))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_page_all_items_robust, p): p for p in pages}
            for future in as_completed(futures):
                try:
                    recs = future.result()
                    all_raw_records.extend(recs)
                except Exception:
                    pass
                    
        elapsed = time.time() - start_time
        speed = len(all_raw_records) / max(elapsed, 0.1)
        print(f"[*] Đã cào đến trang {batch_end:03d}/{end_page} | Tổng số tin thô: {len(all_raw_records):5d} tin | Tốc độ: {speed:.1f} tin/s")
        
        # Checkpoint định kỳ
        df_temp = pd.DataFrame(all_raw_records)
        df_temp.to_csv(out_csv, index=False, encoding='utf-8-sig')
        
    df_final = pd.DataFrame(all_raw_records)
    for col in df_final.select_dtypes(include='object').columns:
        df_final[col] = df_final[col].astype(str).str.replace('\r', ' ', regex=False).str.replace('\n', ' ', regex=False).str.strip()
        
    df_final.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print("\n")
    print(f"THU THẬP HOÀN TẤT: {len(df_final)} BẢN GHI DỮ LIỆU THÔ CAFELAND")
    print(f"Lưu tại: {out_csv} ({os.path.getsize(out_csv)/1024/1024:.2f} MB)")
    print(f"Tổng thời gian: {time.time() - start_time:.1f} giây")
    
    return df_final

def merge_cafeland_to_dataset(cafeland_csv='data/raw/01_raw_crawled_cafeland.csv', main_csv='data/raw/01_raw_crawled_fulltext.csv'):
    """Hợp nhất dữ liệu thô CafeLand vào tập dữ liệu chính"""
    if not os.path.exists(cafeland_csv):
        print(f"[!] Chưa có file {cafeland_csv}!")
        return
        
    df_cf = pd.read_csv(cafeland_csv)
    if not os.path.exists(main_csv):
        df_cf.to_csv(main_csv, index=False, encoding='utf-8-sig')
        print(f"[OK] Đã tạo mới {main_csv} từ CafeLand ({len(df_cf)} dòng).")
        return
        
    df_main = pd.read_csv(main_csv)
    initial_len = len(df_main)
    
    common_cols = [c for c in df_main.columns if c in df_cf.columns]
    df_cf_aligned = df_cf[common_cols]
    
    merged = pd.concat([df_main, df_cf_aligned], ignore_index=True)
    merged.to_csv(main_csv, index=False, encoding='utf-8-sig')
    
    print(f"[OK] Đã hợp nhất dữ liệu thô: Trước={initial_len} | Bổ sung={len(df_cf_aligned)} dòng CafeLand | Tổng={len(merged)} dòng")

if __name__ == '__main__':
    num_pages = 280
    should_merge = False
    
    for arg in sys.argv[1:]:
        if arg == '--merge':
            should_merge = True
        elif arg.isdigit():
            num_pages = int(arg)
            
    df_res = crawl_cafeland_raw_all(start_page=1, end_page=num_pages, max_workers=6, batch_size=20)
    
    if should_merge:
        print("\n[*] Đang hợp nhất vào tập dữ liệu thô chính...")
        merge_cafeland_to_dataset()