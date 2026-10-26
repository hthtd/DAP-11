"""
Module 05: Khám Phá & Trực Quan Hóa Dữ Liệu Chuyên Sâu (EDA - Exploratory Data Analysis)
Tạo hệ thống 8 biểu đồ chuẩn mực, kích thước lớn, cân đối, không bị lệch hoặc dồn cục:
- Cỡ hình rộng rãi (14x8 đến 16x11 inches), độ phân giải 300 DPI
- Cân chỉnh trục tọa độ hợp lý, loại bỏ méo mó tỷ lệ
- Ghi nhãn số liệu trực tiếp (Data Labels) trên các cột và thanh boxplot
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đồng bộ thư mục làm việc về thư mục code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình giao diện biểu đồ chuyên nghiệp & sắc nét
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font_scale=1.1, palette="deep")

def generate_eda_visualizations():
    data_path = 'data/processed/03_spatial_featured_real_estate.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Chưa tìm thấy file 03_spatial_featured_real_estate.csv! Hãy chạy 04_spatial_features.py trước.")
        
    df = pd.read_csv(data_path, low_memory=False)
    os.makedirs('reports/figures', exist_ok=True)
    
    print("=" * 60)
    print(f"[*] Bắt đầu tạo hệ thống 8 biểu đồ EDA kích thước lớn, cân đối ({len(df)} mẫu)...")
    print("=" * 60)
    
    # -------------------------------------------------------------
    # 1. BIỂU ĐỒ 1: Phân phối Biến mục tiêu (Price Billion) & Log Transform
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Lọc phân khúc thông dụng <= 35 tỷ để trực quan hóa cân đối, không bị nén vào góc
    focus_price = df[df['price_billion'] <= 35]
    sns.histplot(focus_price['price_billion'], kde=True, ax=axes[0], color='#1f77b4', bins=35, edgecolor='black', alpha=0.6)
    median_p = df['price_billion'].median()
    mean_p = df['price_billion'].mean()
    axes[0].axvline(median_p, color='#e74c3c', linestyle='--', linewidth=2, label=f'Trung vị: {median_p:.2f} Tỷ VNĐ')
    axes[0].axvline(mean_p, color='#f39c12', linestyle=':', linewidth=2, label=f'Trung bình: {mean_p:.2f} Tỷ VNĐ')
    axes[0].set_title('Phân phối Giá BĐS Phổ Biến (≤ 35 Tỷ - Lệch phải tự nhiên)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Tổng Giá Bán (Tỷ VNĐ)', fontsize=12)
    axes[0].set_ylabel('Số Lượng Tin Đăng', fontsize=12)
    axes[0].legend(loc='upper right', frameon=True)
    
    # Biểu đồ Log Transform
    sns.histplot(np.log1p(df['price_billion']), kde=True, ax=axes[1], color='#2ca02c', bins=35, edgecolor='black', alpha=0.6)
    axes[1].set_title('Phân phối Chuẩn Hóa log(1 + Giá) (Chuẩn bị cho Mô hình ML)', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Giá Trị log(1 + Giá Tỷ VNĐ)', fontsize=12)
    axes[1].set_ylabel('Số Lượng Tin Đăng', fontsize=12)
    
    plt.tight_layout()
    fig.savefig('reports/figures/01_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 1: Phân phối Giá bán & Biến đổi Log -> reports/figures/01_price_distribution.png")
    
    # -------------------------------------------------------------
    # 2. BIỂU ĐỒ 2: Tương quan Diện tích vs Tổng Giá theo Loại hình BĐS
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 8))
    # Lấy phân khúc cốt lõi: Diện tích <= 300m2 và Giá <= 40 Tỷ để phân bố trải đều đẹp mắt
    sample_df = df[(df['area'] >= 15) & (df['area'] <= 300) & (df['price_billion'] <= 40)].copy()
    
    sns.scatterplot(
        data=sample_df, x='area', y='price_billion', hue='property_type',
        alpha=0.65, s=55, palette='tab10', ax=ax
    )
    sns.regplot(
        data=sample_df, x='area', y='price_billion', scatter=False,
        ax=ax, color='#2c3e50', line_kws={'linestyle': '--', 'linewidth': 2.5, 'label': 'Đường xu hướng hồi quy chung'}
    )
    ax.set_title('Tương Quan Giữa Diện Tích (m²) và Tổng Giá Bán (Tỷ VNĐ) Theo Loại Hình BĐS', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Diện Tích Sử Dụng (m²)', fontsize=13)
    ax.set_ylabel('Tổng Giá Rao Bán (Tỷ VNĐ)', fontsize=13)
    ax.legend(title='Loại Hình BĐS', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=11)
    
    plt.tight_layout()
    fig.savefig('reports/figures/02_area_vs_price_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 2: Scatter plot Diện tích vs Tổng giá -> reports/figures/02_area_vs_price_scatter.png")

    # -------------------------------------------------------------
    # 3. BIỂU ĐỒ 3: Phân bố Giá theo Quận / Huyện (Xếp theo Trung vị giá)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 11))
    district_medians = df.groupby('district')['price_billion'].median().sort_values(ascending=False)
    district_order = district_medians.index
    
    palette_colors = sns.color_palette("mako_r", len(district_order))
    sns.boxplot(
        data=df[df['price_billion'] <= 30], y='district', x='price_billion', order=district_order,
        palette=palette_colors, ax=ax, showfliers=False, width=0.65
    )
    
    # Ghi nhãn giá trung vị trực tiếp bên cạnh mỗi quận
    for i, dist in enumerate(district_order):
        med_val = district_medians[dist]
        ax.text(med_val + 0.3, i, f'{med_val:.1f} Tỷ', va='center', fontsize=10, fontweight='bold', color='#1a252f')
        
    ax.set_title('Phân Bố Mức Giá BĐS Theo Từng Quận / Huyện Tại TP.HCM (Sắp xếp theo Trung vị)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Tổng Giá Bán (Tỷ VNĐ)', fontsize=13)
    ax.set_ylabel('Quận / Huyện', fontsize=13)
    
    plt.tight_layout()
    fig.savefig('reports/figures/03_district_price_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 3: Boxplot Giá theo Quận/Huyện -> reports/figures/03_district_price_boxplot.png")

    # -------------------------------------------------------------
    # 4. BIỂU ĐỒ 4: Cơ cấu Loại hình BĐS và Tình trạng Pháp lý
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    type_counts = df['property_type'].value_counts()
    bars1 = axes[0].barh(type_counts.index, type_counts.values, color='#3498db', edgecolor='black', alpha=0.85, height=0.6)
    axes[0].set_title('Cơ Cấu Tin Đăng Theo Loại Hình Bất Động Sản', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Số Lượng Tin Đăng', fontsize=12)
    # Thêm nhãn số lượng & %
    total_samples = len(df)
    for bar in bars1:
        w = bar.get_width()
        axes[0].text(w + 50, bar.get_y() + bar.get_height()/2, f'{int(w):,} ({w/total_samples*100:.1f}%)', va='center', fontsize=10, fontweight='bold')
    axes[0].set_xlim(0, max(type_counts.values) * 1.25)
    
    legal_counts = df['legal_status'].value_counts()
    bars2 = axes[1].barh(legal_counts.index, legal_counts.values, color='#e67e22', edgecolor='black', alpha=0.85, height=0.6)
    axes[1].set_title('Cơ Cấu Tin Đăng Theo Tình Trạng Pháp Lý', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Số Lượng Tin Đăng', fontsize=12)
    for bar in bars2:
        w = bar.get_width()
        axes[1].text(w + 50, bar.get_y() + bar.get_height()/2, f'{int(w):,} ({w/total_samples*100:.1f}%)', va='center', fontsize=10, fontweight='bold')
    axes[1].set_xlim(0, max(legal_counts.values) * 1.25)
    
    plt.tight_layout()
    fig.savefig('reports/figures/04_property_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 4: Phân bố Loại hình & Pháp lý -> reports/figures/04_property_type_distribution.png")

    # -------------------------------------------------------------
    # 5. BIỂU ĐỒ 5: Ma trận Tương quan Đa biến (Correlation Heatmap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10))
    numeric_cols = [
        'price_billion', 'area', 'rooms', 'toilets', 'floors', 
        'dist_cbd_km', 'dist_nearest_metro_km', 'dist_nearest_hospital_km', 'dist_nearest_university_km'
    ]
    vn_labels = [
        'Tổng Giá (Tỷ)', 'Diện Tích (m²)', 'Số Phòng Ngủ', 'Số WC', 'Số Tầng',
        'Cách CBD Q1', 'Cách Ga Metro', 'Cách Bệnh Viện', 'Cách Trường ĐH'
    ]
    corr_df = df[numeric_cols].copy()
    corr_df.columns = vn_labels
    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.8, vmax=0.8,
        square=True, linewidths=1.0, cbar_kws={"shrink": .8, "label": "Hệ số tương quan Pearson (r)"},
        annot_kws={"size": 11, "weight": "bold"}, ax=ax
    )
    ax.set_title('Ma Trận Tương Quan Tuyến Tính Đa Biến Giữa Giá & Các Đặc Trưng GIS / Thuộc Tính', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    fig.savefig('reports/figures/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 5: Correlation Heatmap -> reports/figures/05_correlation_heatmap.png")

    # -------------------------------------------------------------
    # 6. BIỂU ĐỒ 6: Mật độ Không gian & Vị trí BĐS TP.HCM (Hexbin Cân Đối)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10))
    # Lọc tọa độ hợp lệ nằm trong vùng trung tâm & mở rộng TP.HCM
    map_df = df[(df['latitude'] >= 10.60) & (df['latitude'] <= 10.95) & (df['longitude'] >= 106.50) & (df['longitude'] <= 106.90)]
    
    hb = ax.hexbin(
        map_df['longitude'], map_df['latitude'], gridsize=45, cmap='YlOrRd',
        mincnt=1, bins='log', edgecolors='gray', linewidths=0.2
    )
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Số Lượng Bất Động Sản (Thang Log10)', fontsize=12)
    
    ax.set_title('Bản Đồ Mật Độ Phân Bố Không Gian Bất Động Sản TP.HCM (Tỷ lệ cân xứng)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Kinh Độ (Longitude)', fontsize=13)
    ax.set_ylabel('Vĩ Độ (Latitude)', fontsize=13)
    ax.set_aspect('equal', adjustable='box') # Giữ tỷ lệ bản đồ địa lý chuẩn xác, không bị méo hình
    
    plt.tight_layout()
    fig.savefig('reports/figures/06_h3_grid_spatial_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 6: Hexbin Spatial Density Map -> reports/figures/06_h3_grid_spatial_density.png")

    # -------------------------------------------------------------
    # 7. BIỂU ĐỒ 7: Tác động của Khoảng cách Tiện ích tới Mức giá
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    df_poi_focus = df[df['price_billion'] <= 30].copy()
    
    # CBD
    sns.scatterplot(data=df_poi_focus, x='dist_cbd_km', y='price_billion', alpha=0.35, color='#e74c3c', s=30, ax=axes[0, 0])
    sns.regplot(data=df_poi_focus, x='dist_cbd_km', y='price_billion', scatter=False, color='#2c3e50', line_kws={'linewidth': 2}, ax=axes[0, 0])
    axes[0, 0].set_title('1. Khoảng Cách Tới CBD Quận 1 vs Giá (Gần trung tâm giá tăng vượt bậc)', fontweight='bold', fontsize=13)
    axes[0, 0].set_xlabel('Khoảng Cách Tới Chợ Bến Thành (km)', fontsize=11)
    axes[0, 0].set_ylabel('Giá Bán (Tỷ VNĐ)', fontsize=11)

    # Metro
    sns.scatterplot(data=df_poi_focus, x='dist_nearest_metro_km', y='price_billion', alpha=0.35, color='#2980b9', s=30, ax=axes[0, 1])
    sns.regplot(data=df_poi_focus, x='dist_nearest_metro_km', y='price_billion', scatter=False, color='#2c3e50', line_kws={'linewidth': 2}, ax=axes[0, 1])
    axes[0, 1].set_title('2. Khoảng Cách Tới Ga Metro Tuyến 1 vs Giá BĐS', fontweight='bold', fontsize=13)
    axes[0, 1].set_xlabel('Khoảng Cách Tới Ga Metro Gần Nhất (km)', fontsize=11)
    axes[0, 1].set_ylabel('Giá Bán (Tỷ VNĐ)', fontsize=11)

    # Hospital
    sns.scatterplot(data=df_poi_focus, x='dist_nearest_hospital_km', y='price_billion', alpha=0.35, color='#27ae60', s=30, ax=axes[1, 0])
    sns.regplot(data=df_poi_focus, x='dist_nearest_hospital_km', y='price_billion', scatter=False, color='#2c3e50', line_kws={'linewidth': 2}, ax=axes[1, 0])
    axes[1, 0].set_title('3. Khoảng Cách Tới Bệnh Viện Tuyến Đầu vs Giá BĐS', fontweight='bold', fontsize=13)
    axes[1, 0].set_xlabel('Khoảng Cách Tới Bệnh Viện Gần Nhất (km)', fontsize=11)
    axes[1, 0].set_ylabel('Giá Bán (Tỷ VNĐ)', fontsize=11)

    # University
    sns.scatterplot(data=df_poi_focus, x='dist_nearest_university_km', y='price_billion', alpha=0.35, color='#8e44ad', s=30, ax=axes[1, 1])
    sns.regplot(data=df_poi_focus, x='dist_nearest_university_km', y='price_billion', scatter=False, color='#2c3e50', line_kws={'linewidth': 2}, ax=axes[1, 1])
    axes[1, 1].set_title('4. Khoảng Cách Tới Trường Đại Học Lớn vs Giá BĐS', fontweight='bold', fontsize=13)
    axes[1, 1].set_xlabel('Khoảng Cách Tới Trường ĐH Gần Nhất (km)', fontsize=11)
    axes[1, 1].set_ylabel('Giá Bán (Tỷ VNĐ)', fontsize=11)

    plt.tight_layout()
    fig.savefig('reports/figures/07_distance_to_poi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 7: Khoảng cách tiện ích POI vs Giá -> reports/figures/07_distance_to_poi_distribution.png")

    # -------------------------------------------------------------
    # 8. BIỂU ĐỒ 8: Tác động của Số phòng ngủ & Số tầng tới Giá trị BĐS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    room_data = df[(df['rooms'] >= 1) & (df['rooms'] <= 8)].copy()
    room_agg = room_data.groupby('rooms')['price_billion'].median()
    bars_r = axes[0].bar([str(int(r)) for r in room_agg.index], room_agg.values, color='#3498db', edgecolor='black', alpha=0.85, width=0.55)
    axes[0].set_title('Trung Vị Giá BĐS Theo Số Lượng Phòng Ngủ', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel('Số Phòng Ngủ (Phòng)', fontsize=12)
    axes[0].set_ylabel('Trung Vị Giá (Tỷ VNĐ)', fontsize=12)
    for bar in bars_r:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f} Tỷ', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, max(room_agg.values) * 1.18)

    floor_data = df[(df['floors'] >= 1) & (df['floors'] <= 6)].copy()
    floor_agg = floor_data.groupby('floors')['price_billion'].median()
    bars_f = axes[1].bar([str(int(f)) for f in floor_agg.index], floor_agg.values, color='#9b59b6', edgecolor='black', alpha=0.85, width=0.55)
    axes[1].set_title('Trung Vị Giá BĐS Theo Số Tầng Nhà', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Số Tầng Nhà', fontsize=12)
    axes[1].set_ylabel('Trung Vị Giá (Tỷ VNĐ)', fontsize=12)
    for bar in bars_f:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{h:.1f} Tỷ', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, max(floor_agg.values) * 1.18)
    
    plt.tight_layout()
    fig.savefig('reports/figures/08_rooms_floors_price_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 8: Barplot Số phòng/Số tầng vs Giá -> reports/figures/08_rooms_floors_price_bar.png")
    
    print("\n" + "=" * 60)
    print(f"[*] ĐÃ XUẤT THÀNH CÔNG 8 BIỂU ĐỒ EDA ĐẠT CHUẨN VÀO: reports/figures/")
    print("=" * 60)

if __name__ == '__main__':
    generate_eda_visualizations()
