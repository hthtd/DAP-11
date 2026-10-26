"""
Module 05: Huấn Luyện, So Sánh Mô Hình & Giải Thích Bằng SHAP (Modeling & Evaluation)
- Model A (Baseline): Thuộc tính cơ bản + Địa giới hành chính (Quận/Huyện)
- Model B (Proposed): Thuộc tính cơ bản + Uber H3 Grid + Khoảng cách tiện ích đô thị (CBD, Metro, BV, ĐH) + Spatial CV
- Đánh giá trên tập Kiểm thử Test 2026 bằng các chỉ số: MAE, RMSE, R2, MAPE và % Cải thiện
- Giải thích mô hình minh bạch bằng SHAP (SHapley Additive exPlanations)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold, GroupKFold
import lightgbm as lgb
import xgboost as xgb
import shap

# Đồng bộ thư mục làm việc về thư mục code
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.getcwd() != BASE_DIR:
    os.chdir(BASE_DIR)

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Segoe UI']
plt.rcParams['axes.unicode_minus'] = False

def train_and_evaluate_models():
    data_path = 'data/processed/03_spatial_featured_real_estate.csv'
    if not os.path.exists(data_path):
        data_path = 'data/processed/processed_spatial_real_estate.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError("Chưa tìm thấy dữ liệu đặc trưng không gian! Hãy chạy 03_spatial_features.py trước.")
        
    df = pd.read_csv(data_path, low_memory=False)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)
    
    print("=" * 60)
    print(f"[*] Bắt đầu huấn luyện và so sánh Model A vs Model B trên {len(df)} BĐS...")
    print("=" * 60)
    
    train_df = df[df['split_group'] == 'Train (2024-2025)'].copy().reset_index(drop=True)
    test_df = df[df['split_group'] == 'Test (2026)'].copy().reset_index(drop=True)
    
    print(f"[+] Tập Train: {len(train_df)} mẫu | Tập Test 2026: {len(test_df)} mẫu")
    
    # -----------------------------------------------------------------
    # ĐỊNH NGHĨA BIẾN VÀ TIỀN MÃ HÓA
    # -----------------------------------------------------------------
    # Biến phân loại chung
    categorical_cols = ['property_type', 'legal_status', 'district', 'h3_index']
    for col in categorical_cols:
        if col in df.columns:
            train_df[col] = train_df[col].astype('category')
            test_df[col] = test_df[col].astype('category')
            
    # MODEL A FEATURES (Quận/Huyện)
    features_a = ['area', 'rooms', 'toilets', 'floors', 'property_type', 'legal_status', 'district']
    
    # MODEL B FEATURES (H3 Grid + Khoảng cách POI + Lat/Long)
    features_b = [
        'area', 'rooms', 'toilets', 'floors', 'property_type', 'legal_status', 
        'h3_index', 'dist_cbd_km', 'dist_nearest_metro_km', 'dist_nearest_hospital_km', 'dist_nearest_university_km',
        'latitude', 'longitude'
    ]
    
    y_train = train_df['price_billion'].values
    y_test = test_df['price_billion'].values
    
    # Biến đổi log target để tối ưu huấn luyện
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    # -----------------------------------------------------------------
    # HUẤN LUYỆN MODEL A (BASELINE: QUẬN / HUYỆN)
    # -----------------------------------------------------------------
    print("\n--- [1] Đang huấn luyện Model A (Baseline: Quận/Huyện) ---")
    X_train_a = train_df[features_a].copy()
    X_test_a = test_df[features_a].copy()
    
    model_a = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model_a.fit(X_train_a, y_train_log)
    
    y_pred_a_log = model_a.predict(X_test_a)
    y_pred_a = np.expm1(y_pred_a_log)
    
    mae_a = mean_absolute_error(y_test, y_pred_a)
    rmse_a = np.sqrt(mean_squared_error(y_test, y_pred_a))
    r2_a = r2_score(y_test, y_pred_a)
    mape_a = mean_absolute_percentage_error(y_test, y_pred_a) * 100
    
    print(f"    -> Model A MAE:  {mae_a:.3f} Tỷ VNĐ")
    print(f"    -> Model A RMSE: {rmse_a:.3f} Tỷ VNĐ")
    print(f"    -> Model A R2:   {r2_a:.4f}")
    print(f"    -> Model A MAPE: {mape_a:.2f}%")

    # -----------------------------------------------------------------
    # HUẤN LUYỆN MODEL B (PROPOSED: H3 GRID + SPATIAL POI + SPATIAL CV)
    # -----------------------------------------------------------------
    print("\n--- [2] Đang huấn luyện Model B (Proposed: H3 Grid + Spatial POI) ---")
    X_train_b = train_df[features_b].copy()
    X_test_b = test_df[features_b].copy()
    
    # Huấn luyện với LightGBM hỗ trợ Categorical H3 Grid tự nhiên
    model_b = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.03,
        num_leaves=45,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model_b.fit(X_train_b, y_train_log)
    
    y_pred_b_log = model_b.predict(X_test_b)
    y_pred_b = np.expm1(y_pred_b_log)
    
    mae_b = mean_absolute_error(y_test, y_pred_b)
    rmse_b = np.sqrt(mean_squared_error(y_test, y_pred_b))
    r2_b = r2_score(y_test, y_pred_b)
    mape_b = mean_absolute_percentage_error(y_test, y_pred_b) * 100
    
    # % Cải thiện của Model B so với Model A
    improve_mae = ((mae_a - mae_b) / mae_a) * 100
    improve_rmse = ((rmse_a - rmse_b) / rmse_a) * 100
    improve_r2 = ((r2_b - r2_a) / abs(r2_a)) * 100
    
    print(f"    -> Model B MAE:  {mae_b:.3f} Tỷ VNĐ (Cải thiện {improve_mae:+.2f}%)")
    print(f"    -> Model B RMSE: {rmse_b:.3f} Tỷ VNĐ (Cải thiện {improve_rmse:+.2f}%)")
    print(f"    -> Model B R2:   {r2_b:.4f} (Cải thiện {improve_r2:+.2f}%)")
    print(f"    -> Model B MAPE: {mape_b:.2f}%")

    # -----------------------------------------------------------------
    # SPATIAL CROSS-VALIDATION (5 FOLDS) TRÊN TRAIN SET
    # -----------------------------------------------------------------
    print("\n--- [3] Thực hiện 5-Fold Spatial Cross-Validation cho Model B ---")
    gkf = GroupKFold(n_splits=5)
    groups = train_df['h3_index'].astype(str)
    
    cv_maes, cv_rmses, cv_r2s = [], [], []
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_train_b, y_train_log, groups=groups)):
        X_tr, y_tr = X_train_b.iloc[trn_idx], y_train_log[trn_idx]
        X_va, y_va = X_train_b.iloc[val_idx], y_train[val_idx]
        
        fold_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=35, random_state=42, verbose=-1)
        fold_model.fit(X_tr, y_tr)
        va_pred = np.expm1(fold_model.predict(X_va))
        
        cv_maes.append(mean_absolute_error(y_va, va_pred))
        cv_rmses.append(np.sqrt(mean_squared_error(y_va, va_pred)))
        cv_r2s.append(r2_score(y_va, va_pred))
        
    print(f"    -> 5-Fold Spatial CV MAE:  {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f} Tỷ")
    print(f"    -> 5-Fold Spatial CV RMSE: {np.mean(cv_rmses):.3f} ± {np.std(cv_rmses):.3f} Tỷ")
    print(f"    -> 5-Fold Spatial CV R2:   {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    # -----------------------------------------------------------------
    # LƯU MÔ HÌNH VÀ KẾT QUẢ SO SÁNH
    # -----------------------------------------------------------------
    joblib.dump(model_a, 'models/model_a_baseline.pkl')
    joblib.dump(model_b, 'models/model_b_spatial.pkl')
    
    # Metadata đặc trưng
    feature_meta = {
        'features_a': features_a,
        'features_b': features_b,
        'categorical_cols': categorical_cols
    }
    with open('models/feature_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)
        
    metrics = {
        'model_a': {'mae': float(mae_a), 'rmse': float(rmse_a), 'r2': float(r2_a), 'mape': float(mape_a)},
        'model_b': {'mae': float(mae_b), 'rmse': float(rmse_b), 'r2': float(r2_b), 'mape': float(mape_b)},
        'spatial_cv_model_b': {
            'mae_mean': float(np.mean(cv_maes)), 'mae_std': float(np.std(cv_maes)),
            'rmse_mean': float(np.mean(cv_rmses)), 'rmse_std': float(np.std(cv_rmses)),
            'r2_mean': float(np.mean(cv_r2s)), 'r2_std': float(np.std(cv_r2s))
        },
        'improvement_pct': {
            'mae_reduction_pct': float(improve_mae),
            'rmse_reduction_pct': float(improve_rmse),
            'r2_increase_pct': float(improve_r2)
        }
    }
    with open('models/metrics_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------
    # BIỂU ĐỒ 9: SO SÁNH SAI SỐ MODEL A VS MODEL B
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cột 1: Bar so sánh MAE & RMSE có data labels
    x_pos = np.arange(2)
    w = 0.35
    b1 = axes[0].bar(x_pos - w/2, [mae_a, mae_b], width=w, label='MAE (Sai số tuyệt đối)', color='#e74c3c', edgecolor='black', alpha=0.85)
    b2 = axes[0].bar(x_pos + w/2, [rmse_a, rmse_b], width=w, label='RMSE (Căn bậc hai sai số)', color='#2980b9', edgecolor='black', alpha=0.85)
    
    axes[0].set_title(f'So Sánh Sai Số Định Giá (MAE Giảm {improve_mae:.1f}%, RMSE Giảm {improve_rmse:.1f}%)', fontsize=14, fontweight='bold', pad=12)
    axes[0].set_ylabel('Sai Số Định Giá (Tỷ VNĐ)', fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(['Model A (Quận / Huyện)', 'Model B (H3 Grid + 81 POI)'], fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    
    for bar in b1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.3f} Tỷ', ha='center', fontsize=10, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.05, f'{h:.3f} Tỷ', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, max(rmse_a, rmse_b) * 1.22)
    
    # Cột 2: Scatter Predicted vs Actual (Cân đối trong phân khúc <= 35 Tỷ)
    mask_eval = (y_test <= 35) & (y_pred_b <= 35)
    axes[1].scatter(y_test[mask_eval], y_pred_b[mask_eval], alpha=0.45, color='#27ae60', s=35, label='Dự đoán Model B')
    axes[1].plot([0, 35], [0, 35], color='#c0392b', linestyle='--', linewidth=2.5, label='Đường chuẩn lý tưởng (y = x)')
    axes[1].set_title(f'Tương Quan Giá Thực Tế vs Giá Dự Đoán Model B (R² = {r2_b:.4f})', fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel('Giá Thực Tế (Tỷ VNĐ)', fontsize=12)
    axes[1].set_ylabel('Giá Dự Đoán Bởi Model B (Tỷ VNĐ)', fontsize=12)
    axes[1].set_xlim(0, 35)
    axes[1].set_ylim(0, 35)
    axes[1].legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    fig.savefig('reports/figures/09_model_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 9: So sánh Sai số Mô hình -> reports/figures/09_model_error_comparison.png")

    # -----------------------------------------------------------------
    # GIẢI THÍCH MÔ HÌNH BẰNG SHAP (TREE-SHAP)
    # -----------------------------------------------------------------
    print("\n--- [4] Tính toán SHAP Values giải thích mô hình ---")
    sample_shap = X_test_b.sample(min(300, len(X_test_b)), random_state=42)
    explainer = shap.TreeExplainer(model_b)
    shap_values = explainer.shap_values(sample_shap)
    
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    
    # Biểu đồ 10: SHAP Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    shap.summary_plot(shap_values, sample_shap, max_display=12, show=False)
    plt.title('Tầm Quan Trọng & Mức Độ Đóng Góp Của Các Biến Đặc Trưng (TreeSHAP Summary)', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/10_shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[+] Biểu đồ 10: SHAP Summary Beeswarm Plot -> reports/figures/10_shap_summary_plot.png")
    
    print("\n" + "=" * 60)
    print(f"[*] HUẤN LUYỆN & ĐÁNH GIÁ THÀNH CÔNG:")
    print(f"    - Model A (Baseline): MAE={mae_a:.3f}, RMSE={rmse_a:.3f}, R2={r2_a:.4f}")
    print(f"    - Model B (Proposed): MAE={mae_b:.3f}, RMSE={rmse_b:.3f}, R2={r2_b:.4f}")
    print(f"    - Cải thiện MAE: {improve_mae:.2f}% | Cải thiện RMSE: {improve_rmse:.2f}%")
    print("=" * 60)

if __name__ == '__main__':
    train_and_evaluate_models()
