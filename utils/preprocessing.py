import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

# Pengelompokan fitur ke dalam kategori dalam bahasa Indonesia untuk HR
FEATURE_GROUPS: Dict[str, str] = {
    'Age': 'Demografi',
    'DistanceFromHome': 'Keseimbangan Kerja-Hidup',
    'Education': 'Karir & Pendidikan',
    'EnvironmentSatisfaction': 'Lingkungan Kerja',
    'JobInvolvement': 'Keterlibatan Kerja',
    'JobSatisfaction': 'Lingkungan Kerja',
    'MonthlyIncome': 'Kompensasi',
    'NumCompaniesWorked': 'Riwayat Karir',
    'PercentSalaryHike': 'Kompensasi',
    'RelationshipSatisfaction': 'Lingkungan Kerja',
    'StockOptionLevel': 'Kompensasi',
    'TotalWorkingYears': 'Riwayat Karir',
    'TrainingTimesLastYear': 'Pengembangan Karyawan',
    'WorkLifeBalance': 'Keseimbangan Kerja-Hidup',
    'YearsAtCompany': 'Retensi & Loyalitas',
    'YearsSinceLastPromotion': 'Retensi & Loyalitas',
    'BusinessTravel_Travel_Frequently': 'Keseimbangan Kerja-Hidup',
    'BusinessTravel_Travel_Rarely': 'Keseimbangan Kerja-Hidup',
    'Department_Research & Development': 'Departemen',
    'Department_Sales': 'Departemen',
    'EducationField_Life Sciences': 'Karir & Pendidikan',
    'EducationField_Marketing': 'Karir & Pendidikan',
    'EducationField_Medical': 'Karir & Pendidikan',
    'EducationField_Other': 'Karir & Pendidikan',
    'EducationField_Technical Degree': 'Karir & Pendidikan',
    'Gender_Male': 'Demografi',
    'MaritalStatus_Married': 'Demografi',
    'MaritalStatus_Single': 'Demografi',
    'OverTime_Yes': 'Keterlibatan Kerja',
    'JobRole_Human Resources': 'Peran Pekerjaan',
    'JobRole_Laboratory Technician': 'Peran Pekerjaan',
    'JobRole_Manager': 'Peran Pekerjaan',
    'JobRole_Manufacturing Director': 'Peran Pekerjaan',
    'JobRole_Research Director': 'Peran Pekerjaan',
    'JobRole_Research Scientist': 'Peran Pekerjaan',
    'JobRole_Sales Executive': 'Peran Pekerjaan',
    'JobRole_Sales Representative': 'Peran Pekerjaan'
}

FEATURE_INDONESIAN_MAP: Dict[str, str] = {
    'Age': 'Usia',
    'DistanceFromHome': 'Jarak dari Rumah',
    'Education': 'Tingkat Pendidikan',
    'EnvironmentSatisfaction': 'Kepuasan Lingkungan',
    'JobInvolvement': 'Keterlibatan Kerja',
    'JobSatisfaction': 'Kepuasan Kerja',
    'MonthlyIncome': 'Pendapatan Bulanan',
    'NumCompaniesWorked': 'Jumlah Perusahaan Sebelumnya',
    'PercentSalaryHike': 'Persentase Kenaikan Gaji',
    'RelationshipSatisfaction': 'Kepuasan Hubungan',
    'StockOptionLevel': 'Tingkat Opsi Saham',
    'TotalWorkingYears': 'Total Masa Kerja (Tahun)',
    'TrainingTimesLastYear': 'Jumlah Pelatihan Tahun Lalu',
    'WorkLifeBalance': 'Keseimbangan Kerja-Hidup',
    'YearsAtCompany': 'Lama Bekerja di Perusahaan Ini',
    'YearsSinceLastPromotion': 'Tahun Sejak Promosi Terakhir',
    'BusinessTravel': 'Perjalanan Bisnis',
    'Department': 'Departemen',
    'EducationField': 'Bidang Pendidikan',
    'Gender': 'Jenis Kelamin',
    'MaritalStatus': 'Status Pernikahan',
    'OverTime': 'Lembur',
    'JobRole': 'Peran Pekerjaan'
}

VALUE_INDONESIAN_MAP: Dict[str, str] = {
    'Travel_Frequently': 'Sering',
    'Travel_Rarely': 'Jarang',
    'Non-Travel': 'Tidak Pernah',
    'Research & Development': 'Penelitian & Pengembangan',
    'Sales': 'Penjualan',
    'Human Resources': 'Sumber Daya Manusia',
    'Life Sciences': 'Ilmu Hayati',
    'Medical': 'Medis',
    'Marketing': 'Pemasaran',
    'Technical Degree': 'Gelar Teknis',
    'Other': 'Lainnya',
    'Male': 'Laki-laki',
    'Female': 'Perempuan',
    'Married': 'Menikah',
    'Single': 'Lajang',
    'Divorced': 'Cerai',
    'Yes': 'Ya',
    'No': 'Tidak',
    'Sales Executive': 'Eksekutif Penjualan',
    'Research Scientist': 'Ilmuwan Peneliti',
    'Laboratory Technician': 'Teknisi Laboratorium',
    'Manufacturing Director': 'Direktur Manufaktur',
    'Healthcare Representative': 'Perwakilan Layanan Kesehatan',
    'Manager': 'Manajer',
    'Sales Representative': 'Perwakilan Penjualan',
    'Research Director': 'Direktur Penelitian'
}

def scale_features(df_features: pd.DataFrame, scaler: Any) -> np.ndarray:
    """Melakukan standardisasi fitur input menggunakan scaler yang telah dilatih."""
    return scaler.transform(df_features)

def calculate_local_contributions(
    df_raw_row: pd.DataFrame, 
    X_scaled: np.ndarray, 
    scaler: Any, 
    best_lr: Any
) -> pd.DataFrame:
    """
    Menghitung kontribusi lokal fitur terhadap risiko attrisi menggunakan koefisien 
    Regresi Logistik. Menghasilkan kontribusi positif dan negatif.
    """
    feature_names = df_raw_row.columns.tolist()
    coefs = best_lr.coef_[0]
    
    contributions = X_scaled[0] * coefs
    
    records = []
    for col, val_scaled, coef, contrib in zip(feature_names, X_scaled[0], coefs, contributions):
        raw_val = df_raw_row.iloc[0][col]
        # Lewati jika nilai dummy binary adalah 0.0 (tidak aktif)
        if '_' in col and raw_val == 0.0:
            continue
            
        group = FEATURE_GROUPS.get(col, 'Lainnya')
        
        # Penamaan deskriptif yang ramah untuk HR
        friendly_name = col
        if '_' in col:
            parts = col.split('_')
            feat_name = FEATURE_INDONESIAN_MAP.get(parts[0], parts[0])
            val_name = VALUE_INDONESIAN_MAP.get(parts[1], parts[1])
            friendly_name = f"{feat_name}: {val_name}"
        else:
            friendly_name = FEATURE_INDONESIAN_MAP.get(col, col)
            
        records.append({
            'Feature': col,
            'Label': friendly_name,
            'Category': group,
            'RawValue': raw_val,
            'ScaledValue': val_scaled,
            'Coefficient': coef,
            'Contribution': contrib
        })
        
    df_contrib = pd.DataFrame(records)
    df_contrib['AbsContribution'] = df_contrib['Contribution'].abs()
    df_contrib = df_contrib.sort_values(by='Contribution', ascending=False).reset_index(drop=True)
    return df_contrib
