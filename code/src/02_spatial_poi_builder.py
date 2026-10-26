"""
Module 03_spatial_poi_builder: Xây dựng cơ sở dữ liệu POI (Points of Interest) quy mô lớn tại TP.HCM
Bao gồm hơn 75+ địa điểm trọng điểm phân chia thành 6 nhóm tiện ích:
1. Trung tâm Hành chính & Tài chính (CBD, Landmark, Sân bay)
2. Hệ thống Giao thông huyết mạch (14 Ga Metro Tuyến 1, Ga Tuyến 2, Bến xe, Ga Sài Gòn)
3. Trung tâm Thương mại & Bán lẻ lớn (Vincom, Takashimaya, Crescent Mall, Aeon Mall, Gigamall...)
4. Bệnh viện tuyến Trung ương & Quốc tế lớn
5. Trường Đại học & Làng Đại học Quốc gia
6. Công viên trung tâm & Điểm nhấn cảnh quan đô thị
"""

import json
import os
import sys
import geopandas as gpd
from shapely.geometry import Point

# Đồng bộ thư mục làm việc về thư mục code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')

POIS = [
    # -------------------------------------------------------------
    # 1. TRUNG TÂM HÀNH CHÍNH, TÀI CHÍNH & ĐIỂM NGHẼN HẠ TẦNG
    # -------------------------------------------------------------
    {"name": "CBD Quận 1 (Chợ Bến Thành / Nguyễn Huệ)", "type": "CBD", "lat": 10.7725, "lon": 106.6980},
    {"name": "KĐT Mới Thủ Thiêm (Trung tâm Tài chính mới)", "type": "CBD", "lat": 10.7735, "lon": 106.7180},
    {"name": "Landmark 81 & Tân Cảng", "type": "CBD", "lat": 10.7950, "lon": 106.7218},
    {"name": "Trung tâm Đô thị Phú Mỹ Hưng (Q7)", "type": "CBD", "lat": 10.7295, "lon": 106.7085},
    {"name": "Sân bay Quốc tế Tân Sơn Nhất", "type": "Airport", "lat": 10.8185, "lon": 106.6588},
    {"name": "Ga Xe Lửa Sài Gòn (Quận 3)", "type": "Transit", "lat": 10.7818, "lon": 106.6782},
    {"name": "Bến xe Miền Tây (Bình Tân)", "type": "Transit", "lat": 10.7410, "lon": 106.6185},
    {"name": "Bến xe Miền Đông Cũ (Bình Thạnh)", "type": "Transit", "lat": 10.8142, "lon": 106.7095},
    
    # -------------------------------------------------------------
    # 2. TOÀN BỘ 14 GA METRO TUYẾN SỐ 1 (BẾN THÀNH - SUỐI TIÊN)
    # -------------------------------------------------------------
    {"name": "Ga Metro 01: Bến Thành", "type": "Metro", "lat": 10.7717, "lon": 106.6983},
    {"name": "Ga Metro 02: Nhà Hát Thành Phố", "type": "Metro", "lat": 10.7766, "lon": 106.7032},
    {"name": "Ga Metro 03: Ba Son", "type": "Metro", "lat": 10.7836, "lon": 106.7082},
    {"name": "Ga Metro 04: Văn Thánh", "type": "Metro", "lat": 10.7963, "lon": 106.7168},
    {"name": "Ga Metro 05: Tân Cảng", "type": "Metro", "lat": 10.7995, "lon": 106.7225},
    {"name": "Ga Metro 06: Thảo Điền", "type": "Metro", "lat": 10.8035, "lon": 106.7352},
    {"name": "Ga Metro 07: An Phú", "type": "Metro", "lat": 10.8039, "lon": 106.7447},
    {"name": "Ga Metro 08: Rạch Chiếc", "type": "Metro", "lat": 10.8143, "lon": 106.7583},
    {"name": "Ga Metro 09: Phước Long", "type": "Metro", "lat": 10.8248, "lon": 106.7645},
    {"name": "Ga Metro 10: Bình Thái", "type": "Metro", "lat": 10.8354, "lon": 106.7708},
    {"name": "Ga Metro 11: Thủ Đức", "type": "Metro", "lat": 10.8465, "lon": 106.7770},
    {"name": "Ga Metro 12: Khu Công Nghệ Cao", "type": "Metro", "lat": 10.8576, "lon": 106.7865},
    {"name": "Ga Metro 13: Đại Học Quốc Gia", "type": "Metro", "lat": 10.8752, "lon": 106.8012},
    {"name": "Ga Metro 14: Bến Xe Miền Đông Mới", "type": "Metro", "lat": 10.8805, "lon": 106.8123},

    # -------------------------------------------------------------
    # 3. TRUNG TÂM THƯƠNG MẠI & PHỨC HỢP MUA SẮM (SHOPPING MALLS)
    # -------------------------------------------------------------
    {"name": "Vincom Center Đồng Khởi (Q1)", "type": "Shopping", "lat": 10.7780, "lon": 106.7018},
    {"name": "Takashimaya / Saigon Centre (Q1)", "type": "Shopping", "lat": 10.7736, "lon": 106.7008},
    {"name": "Crescent Mall Phú Mỹ Hưng (Q7)", "type": "Shopping", "lat": 10.7290, "lon": 106.7215},
    {"name": "SC VivoCity (Quận 7)", "type": "Shopping", "lat": 10.7325, "lon": 106.7060},
    {"name": "Lotte Mart Nam Sài Gòn (Q7)", "type": "Shopping", "lat": 10.7420, "lon": 106.7015},
    {"name": "Vạn Hạnh Mall (Quận 10)", "type": "Shopping", "lat": 10.7705, "lon": 106.6698},
    {"name": "Gigamall Phạm Văn Đồng (TP Thủ Đức)", "type": "Shopping", "lat": 10.8282, "lon": 106.7245},
    {"name": "Vincom Mega Mall Thảo Điền", "type": "Shopping", "lat": 10.8038, "lon": 106.7435},
    {"name": "Vincom Mega Mall Grand Park (Q9 cũ)", "type": "Shopping", "lat": 10.8465, "lon": 106.8240},
    {"name": "Aeon Mall Tân Phú Celadon", "type": "Shopping", "lat": 10.8005, "lon": 106.6175},
    {"name": "Aeon Mall Bình Tân", "type": "Shopping", "lat": 10.7435, "lon": 106.6110},
    {"name": "Emart Phan Văn Trị (Gò Vấp)", "type": "Shopping", "lat": 10.8272, "lon": 106.6890},
    {"name": "Emart Sala Đại Quang Minh (Thủ Thiêm)", "type": "Shopping", "lat": 10.7680, "lon": 106.7220},

    # -------------------------------------------------------------
    # 4. HỆ THỐNG BỆNH VIỆN TUYẾN ĐẦU & QUỐC TẾ
    # -------------------------------------------------------------
    {"name": "Bệnh viện Chợ Rẫy (Q5)", "type": "Hospital", "lat": 10.7578, "lon": 106.6596},
    {"name": "Bệnh viện Đại học Y Dược (Q5)", "type": "Hospital", "lat": 10.7561, "lon": 106.6619},
    {"name": "Bệnh viện Từ Dũ (Q1)", "type": "Hospital", "lat": 10.7686, "lon": 106.6853},
    {"name": "Bệnh viện Hùng Vương (Q5)", "type": "Hospital", "lat": 10.7545, "lon": 106.6625},
    {"name": "Bệnh viện Nhi Đồng 1 (Q10)", "type": "Hospital", "lat": 10.7708, "lon": 106.6710},
    {"name": "Bệnh viện Nhi Đồng 2 (Q1)", "type": "Hospital", "lat": 10.7812, "lon": 106.7020},
    {"name": "Bệnh viện Nhi Đồng Thành Phố (Bình Chánh)", "type": "Hospital", "lat": 10.6865, "lon": 106.5680},
    {"name": "Bệnh viện Nhân Dân 115 (Q10)", "type": "Hospital", "lat": 10.7744, "lon": 106.6622},
    {"name": "Bệnh viện Tim Tâm Đức (Q7)", "type": "Hospital", "lat": 10.7308, "lon": 106.7198},
    {"name": "Bệnh viện FV Pháp Việt (Q7)", "type": "Hospital", "lat": 10.7303, "lon": 106.7214},
    {"name": "Bệnh viện Nhân Dân Gia Định (Bình Thạnh)", "type": "Hospital", "lat": 10.8021, "lon": 106.6953},
    {"name": "Bệnh viện Ung Bướu CS1 (Bình Thạnh)", "type": "Hospital", "lat": 10.8015, "lon": 106.6945},
    {"name": "Bệnh viện Ung Bướu CS2 (TP Thủ Đức)", "type": "Hospital", "lat": 10.8702, "lon": 106.8291},
    {"name": "Bệnh viện Quân Y 175 (Gò Vấp)", "type": "Hospital", "lat": 10.8178, "lon": 106.6806},
    {"name": "Bệnh viện Thống Nhất (Tân Bình)", "type": "Hospital", "lat": 10.7915, "lon": 106.6545},
    {"name": "Bệnh viện TP Thủ Đức", "type": "Hospital", "lat": 10.8540, "lon": 106.7533},
    {"name": "Bệnh viện Đa khoa Quốc tế Vinmec Central Park", "type": "Hospital", "lat": 10.7935, "lon": 106.7210},
    {"name": "Bệnh viện Quốc tế City (Bình Tân)", "type": "Hospital", "lat": 10.7425, "lon": 106.6120},

    # -------------------------------------------------------------
    # 5. TRƯỜNG ĐẠI HỌC & TRUNG TÂM GIÁO DỤC TRỌNG ĐIỂM
    # -------------------------------------------------------------
    {"name": "ĐHQG TP.HCM - Làng Đại học (Thủ Đức)", "type": "University", "lat": 10.8700, "lon": 106.8030},
    {"name": "ĐH Bách Khoa TP.HCM (CS1 Q10)", "type": "University", "lat": 10.7726, "lon": 106.6597},
    {"name": "ĐH Bách Khoa TP.HCM (CS2 Dĩ An)", "type": "University", "lat": 10.8800, "lon": 106.8055},
    {"name": "ĐH Khoa học Tự nhiên (Q5)", "type": "University", "lat": 10.7628, "lon": 106.6825},
    {"name": "ĐH Khoa học Xã hội & Nhân văn (Q1)", "type": "University", "lat": 10.7865, "lon": 106.7025},
    {"name": "ĐH Kinh tế TP.HCM (UEH CS A - Q3)", "type": "University", "lat": 10.7828, "lon": 106.6958},
    {"name": "ĐH Kinh tế TP.HCM (UEH CS Nam Sài Gòn)", "type": "University", "lat": 10.7090, "lon": 106.6710},
    {"name": "ĐH Y Dược TP.HCM (Q5)", "type": "University", "lat": 10.7566, "lon": 106.6622},
    {"name": "ĐH Quốc tế RMIT Việt Nam (Q7)", "type": "University", "lat": 10.7301, "lon": 106.6936},
    {"name": "ĐH Tôn Đức Thắng (Q7)", "type": "University", "lat": 10.7327, "lon": 106.6992},
    {"name": "ĐH Cảnh sát Nhân dân (Q7)", "type": "University", "lat": 10.7355, "lon": 106.7115},
    {"name": "ĐH Sư Phạm Kỹ Thuật TP.HCM (Thủ Đức)", "type": "University", "lat": 10.8507, "lon": 106.7719},
    {"name": "ĐH Nông Lâm TP.HCM (Thủ Đức)", "type": "University", "lat": 10.8705, "lon": 106.7915},
    {"name": "ĐH Công nghiệp TP.HCM - IUH (Gò Vấp)", "type": "University", "lat": 10.8222, "lon": 106.6875},
    {"name": "ĐH HUTECH (Bình Thạnh)", "type": "University", "lat": 10.8016, "lon": 106.7144},
    {"name": "ĐH Ngoại Thương CS2 (Bình Thạnh)", "type": "University", "lat": 10.8055, "lon": 106.7148},
    {"name": "ĐH Giao thông Vận tải CS2 (Thủ Đức)", "type": "University", "lat": 10.8450, "lon": 106.7820},
    {"name": "ĐH Sài Gòn (Q5)", "type": "University", "lat": 10.7595, "lon": 106.6815},
    {"name": "ĐH Văn Lang (CS3 Gò Vấp)", "type": "University", "lat": 10.8305, "lon": 106.6980},

    # -------------------------------------------------------------
    # 6. CÔNG VIÊN & KHÔNG GIAN CÔNG CỘNG XANH (PARKS & LANDMARKS)
    # -------------------------------------------------------------
    {"name": "Công viên Tao Đàn (Q1)", "type": "Park", "lat": 10.7745, "lon": 106.6930},
    {"name": "Thảo Cầm Viên Sài Gòn (Q1)", "type": "Park", "lat": 10.7875, "lon": 106.7050},
    {"name": "Công viên Vinhomes Central Park (Bình Thạnh)", "type": "Park", "lat": 10.7930, "lon": 106.7230},
    {"name": "Công viên Gia Định (Gò Vấp / Phú Nhuận)", "type": "Park", "lat": 10.8145, "lon": 106.6740},
    {"name": "Công viên Hoàng Văn Thụ (Tân Bình)", "type": "Park", "lat": 10.8005, "lon": 106.6625},
    {"name": "Công viên Cầu Ánh Sao / Hồ Bán Nguyệt (Q7)", "type": "Park", "lat": 10.7275, "lon": 106.7190},
    {"name": "Khu Công Viên Sala (KĐT Sala Thủ Thiêm)", "type": "Park", "lat": 10.7695, "lon": 106.7205},
    {"name": "Công viên Đầm Sen (Quận 11)", "type": "Park", "lat": 10.7675, "lon": 106.6385},
    {"name": "Khu Du Lịch Văn Hóa Suối Tiên (TP Thủ Đức)", "type": "Park", "lat": 10.8655, "lon": 106.8020}
]

def generate_poi_geojson():
    os.makedirs('data/spatial_poi', exist_ok=True)
    
    geometry = [Point(p['lon'], p['lat']) for p in POIS]
    gdf = gpd.GeoDataFrame(POIS, geometry=geometry, crs="EPSG:4326")
    
    geojson_path = 'data/spatial_poi/hcm_pois.geojson'
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"[OK] Đã lưu thành công {len(gdf)} điểm tiện ích không gian (POI) vào: {geojson_path}")
    print("Danh sách phân loại:")
    print(gdf['type'].value_counts().to_string())
    return gdf

if __name__ == '__main__':
    generate_poi_geojson()
