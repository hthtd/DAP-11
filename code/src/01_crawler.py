"""
Module 01: Điều Phối Thu Thập Dữ Liệu Bất Động Sản TP.HCM (Multi-Source Crawler)
Tích hợp 2 nguồn dữ liệu chính thống:
1. CafeLand (nhadat.cafeland.vn) -> src/01_cafeland_crawler.py
2. Chợ Tốt / Nhà Tốt (API Gateway) -> src/01_chotot_crawler.py

Kết quả hợp nhất lưu tại: data/raw/01_raw_crawled_fulltext.csv
"""

import os
import sys
import subprocess
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

def run_multi_source_crawlers(target_cafeland=7000, target_chotot=7000):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=" * 70)
    print("🌐 ĐIỀU PHỐI THU THẬP DỮ LIỆU ĐA NGUỒN: CAFELAND & CHỢ TỐT")
    print("=" * 70)
    
    cf_script = os.path.join(script_dir, '01_cafeland_crawler.py')
    ct_script = os.path.join(script_dir, '01_chotot_crawler.py')
    
    print(f"\n[1/2] Đang thu thập từ nguồn CafeLand (Target: {target_cafeland} tin)...")
    subprocess.run([sys.executable, cf_script, str(target_cafeland)], check=False)
    
    print(f"\n[2/2] Đang thu thập từ nguồn Chợ Tốt (Target: {target_chotot} tin)...")
    subprocess.run([sys.executable, ct_script], check=False)
    
    # Hợp nhất 2 nguồn dữ liệu
    cf_csv = os.path.join(raw_dir, '01_raw_crawled_cafeland.csv')
    ct_csv = os.path.join(raw_dir, '01_raw_crawled_chotot.csv')
    if not os.path.exists(ct_csv):
        ct_csv = os.path.join(raw_dir, 'raw_real_estate_hcm.csv')
        
    merged_csv = os.path.join(raw_dir, '01_raw_crawled_fulltext.csv')
    
    dfs = []
    if os.path.exists(cf_csv):
        dfs.append(pd.read_csv(cf_csv))
    if os.path.exists(ct_csv):
        dfs.append(pd.read_csv(ct_csv))
        
    if dfs:
        common_cols = list(set.intersection(*(set(df.columns) for df in dfs)))
        df_merged = pd.concat([df[common_cols] for df in dfs], ignore_index=True)
        df_merged.to_csv(merged_csv, index=False, encoding='utf-8-sig')
        print("\n" + "=" * 70)
        print(f"✅ HỢP NHẤT HOÀN TẤT: {len(df_merged)} BẢN GHI DỮ LIỆU THÔ ĐA NGUỒN")
        print(f"📁 Lưu tại: {merged_csv}")
        print("=" * 70)
    else:
        print("[!] Không tìm thấy dữ liệu nguồn để hợp nhất.")

if __name__ == '__main__':
    run_multi_source_crawlers()
