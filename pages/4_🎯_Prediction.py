import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.model_loader import load_ml_artifacts, load_ann_model
from utils.feature_builder import build_feature_vector
from utils.preprocessing import scale_features, calculate_local_contributions
from utils.plots import plot_prediction_gauge, plot_feature_importance_bar
from utils.styles import inject_custom_css

# Konfigurasi Halaman
st.set_page_config(page_title="Diagnosis Risiko Attrisi", page_icon="🎯", layout="wide")

# Suntikkan gaya CSS kustom
inject_custom_css()

st.markdown("<div class='app-title'>🎯 Diagnosis Risiko Attrisi Karyawan</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Masukkan profil karyawan untuk menguji tingkat risiko pengunduran diri dan memperoleh rencana retensi.</div>", unsafe_allow_html=True)

# Muat Model & Aset
with st.spinner("Menginisialisasi model dan memuat aset preproses..."):
    try:
        artifacts = load_ml_artifacts()
        ann_model = load_ann_model()
        scaler = artifacts['scaler']
        best_lr = artifacts['best_lr']
        best_xgb = artifacts['best_xgb']
        best_rf = artifacts['best_rf']
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# Layout Form Input
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("📋 Form Profiler Input Karyawan")

# Membuat tab input
tab_demo, tab_job, tab_comp = st.tabs([
    "👤 Demografi & Masa Kerja",
    "🏢 Peran & Lingkungan Kerja",
    "💰 Kompensasi & Keterlibatan Kerja"
])

# Definisi Input di dalam tab
with tab_demo:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia", min_value=18, max_value=65, value=35, step=1)
        gender_map = {"Female": "Perempuan", "Male": "Laki-laki"}
        gender = st.selectbox("Jenis Kelamin", options=list(gender_map.keys()), format_func=lambda x: gender_map[x])
        marital_map = {"Single": "Lajang", "Married": "Menikah", "Divorced": "Cerai"}
        marital_status = st.selectbox("Status Pernikahan", options=list(marital_map.keys()), format_func=lambda x: marital_map[x])
    with col2:
        distance_from_home = st.number_input("Jarak dari Rumah (km)", min_value=1, max_value=30, value=8, step=1)
        education = st.selectbox("Tingkat Pendidikan", options=[1, 2, 3, 4, 5], 
                                 format_func=lambda x: {1: "1: SMA", 2: "2: Diploma", 3: "3: S1 (Sarjana)", 4: "4: S2 (Magister)", 5: "5: S3 (Doktor)"}[x], index=2)
        num_companies = st.number_input("Jumlah Perusahaan Sebelumnya", min_value=0, max_value=10, value=2, step=1)
    with col3:
        total_working_years = st.number_input("Total Masa Kerja (Tahun)", min_value=0, max_value=40, value=10, step=1)
        years_at_company = st.number_input("Lama Bekerja di Perusahaan Ini (Tahun)", min_value=0, max_value=40, value=5, step=1)
        years_since_promotion = st.number_input("Tahun Sejak Promosi Terakhir", min_value=0, max_value=15, value=1, step=1)

with tab_job:
    col1, col2, col3 = st.columns(3)
    with col1:
        dept_map = {
            "Research & Development": "Penelitian & Pengembangan",
            "Sales": "Penjualan",
            "Human Resources": "Sumber Daya Manusia"
        }
        department = st.selectbox("Departemen", options=list(dept_map.keys()), format_func=lambda x: dept_map[x])
        
        job_role_map = {
            "Sales Executive": "Eksekutif Penjualan",
            "Research Scientist": "Ilmuwan Peneliti",
            "Laboratory Technician": "Teknisi Laboratorium",
            "Manufacturing Director": "Direktur Manufaktur",
            "Healthcare Representative": "Perwakilan Layanan Kesehatan",
            "Manager": "Manajer",
            "Sales Representative": "Perwakilan Penjualan",
            "Research Director": "Direktur Penelitian",
            "Human Resources": "Sumber Daya Manusia"
        }
        job_role = st.selectbox("Peran Pekerjaan", options=list(job_role_map.keys()), format_func=lambda x: job_role_map[x])
    with col2:
        travel_map = {
            "Travel_Rarely": "Jarang",
            "Travel_Frequently": "Sering",
            "Non-Travel": "Tidak Pernah"
        }
        business_travel = st.selectbox("Perjalanan Bisnis", options=list(travel_map.keys()), format_func=lambda x: travel_map[x])
        
        field_map = {
            "Life Sciences": "Ilmu Hayati",
            "Medical": "Medis",
            "Marketing": "Pemasaran",
            "Technical Degree": "Gelar Teknis",
            "Human Resources": "Sumber Daya Manusia",
            "Other": "Lainnya"
        }
        education_field = st.selectbox("Bidang Pendidikan", options=list(field_map.keys()), format_func=lambda x: field_map[x])
    with col3:
        training_times = st.number_input("Jumlah Pelatihan Tahun Lalu", min_value=0, max_value=6, value=2, step=1)
        env_satisfaction = st.selectbox("Kepuasan Lingkungan Kerja", options=[1, 2, 3, 4], 
                                        format_func=lambda x: {1: "1: Rendah", 2: "2: Sedang", 3: "3: Tinggi", 4: "4: Sangat Tinggi"}[x], index=2)
        job_satisfaction = st.selectbox("Kepuasan Kerja", options=[1, 2, 3, 4], 
                                        format_func=lambda x: {1: "1: Rendah", 2: "2: Sedang", 3: "3: Tinggi", 4: "4: Sangat Tinggi"}[x], index=2)

with tab_comp:
    col1, col2, col3 = st.columns(3)
    with col1:
        monthly_income = st.number_input("Pendapatan Bulanan ($)", min_value=1000, max_value=25000, value=5000, step=100)
        percent_salary_hike = st.number_input("Persentase Kenaikan Gaji (%)", min_value=11, max_value=25, value=14, step=1)
    with col2:
        stock_option_level = st.selectbox("Tingkat Opsi Saham", options=[0, 1, 2, 3], index=1)
        relationship_sat = st.selectbox("Kepuasan Hubungan (Rekan Kerja)", options=[1, 2, 3, 4], 
                                        format_func=lambda x: {1: "1: Rendah", 2: "2: Sedang", 3: "3: Tinggi", 4: "4: Sangat Tinggi"}[x], index=2)
    with col3:
        overtime_map = {"No": "Tidak", "Yes": "Ya"}
        overtime = st.selectbox("Lembur Wajib", options=list(overtime_map.keys()), format_func=lambda x: overtime_map[x])
        work_life_balance = st.selectbox("Keseimbangan Kerja-Hidup (WLB)", options=[1, 2, 3, 4], 
                                         format_func=lambda x: {1: "1: Buruk", 2: "2: Cukup", 3: "3: Baik", 4: "4: Terbaik"}[x], index=2)
        job_involvement = st.selectbox("Keterlibatan Kerja", options=[1, 2, 3, 4], 
                                       format_func=lambda x: {1: "1: Rendah", 2: "2: Sedang", 3: "3: Tinggi", 4: "4: Sangat Tinggi"}[x], index=2)

st.markdown("</div>", unsafe_allow_html=True)

# Memilih Model Prediktif
selected_model_name = st.selectbox("Pilih Model Klasifikasi untuk Diagnosis:", [
    "Keras ANN (Direkomendasikan - Recall / Sensitivitas Tinggi)", 
    "Tuned XGBoost (Seimbang)", 
    "Logistic Regression (Interpretabilitas Tinggi)"
])

# Tombol Eksekusi
if st.button("🚀 Jalankan Diagnosis Attrisi Karyawan", use_container_width=True):
    # Menyusun kamus input (menggunakan kunci asli Bahasa Inggris untuk model)
    raw_inputs = {
        'Age': age, 'DistanceFromHome': distance_from_home, 'Education': education,
        'EnvironmentSatisfaction': env_satisfaction, 'JobInvolvement': job_involvement,
        'JobSatisfaction': job_satisfaction, 'MonthlyIncome': monthly_income,
        'NumCompaniesWorked': num_companies, 'PercentSalaryHike': percent_salary_hike,
        'RelationshipSatisfaction': relationship_sat, 'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times,
        'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
        'YearsSinceLastPromotion': years_since_promotion, 'BusinessTravel': business_travel,
        'Department': department, 'EducationField': education_field, 'Gender': gender,
        'MaritalStatus': marital_status, 'OverTime': overtime, 'JobRole': job_role
    }
    
    # Rekonstruksi vektor fitur (37 fitur)
    df_vector = build_feature_vector(raw_inputs)
    
    # Standardisasi Fitur
    X_scaled = scale_features(df_vector, scaler)
    
    # Menjalankan Prediksi berdasarkan model pilihan
    if "Keras ANN" in selected_model_name:
        prob = float(ann_model.predict(X_scaled, verbose=0)[0][0])
    elif "XGBoost" in selected_model_name:
        prob = float(best_xgb.predict_proba(X_scaled)[0][1])
    else:
        prob = float(best_lr.predict_proba(X_scaled)[0][1])
        
    # Menentukan Kategori Risiko
    if prob < 0.35:
        risk_category = "RISIKO RENDAH"
        risk_color = "#10B981" # Hijau Emerald
    elif prob < 0.70:
        risk_category = "RISIKO SEDANG"
        risk_color = "#F59E0B" # Oranye Amber
    else:
        risk_category = "RISIKO TINGGI"
        risk_color = "#EF4444" # Merah Rose
        
    # Menghitung kontribusi lokal fitur (surrogate model Regresi Logistik)
    df_contrib = calculate_local_contributions(df_vector, X_scaled, scaler, best_lr)
    
    # Memilah faktor pendorong dan faktor protektif
    drivers = df_contrib[df_contrib['Contribution'] > 0.02].sort_values(by='Contribution', ascending=False)
    protective = df_contrib[df_contrib['Contribution'] < -0.02].sort_values(by='Contribution', ascending=True)
    
    # Menampilkan Hasil Diagnosis
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.header("🎯 Hasil Diagnosis Risiko Attrisi Karyawan")
    
    # Chart gauge
    fig_gauge = plot_prediction_gauge(prob)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Teks ringkasan diagnosis
    st.markdown(f"""
        <div style="text-align: center; margin-top: -15px; margin-bottom: 25px;">
            <p style="font-size: 1.15rem; font-weight: 700; margin-bottom: 5px;">
                Penilaian Risiko: <span style="color: {risk_color}; font-weight: bold;">{risk_category}</span>
            </p>
            <p style="font-size: 0.95rem; color: #64748B; max-width: 500px; margin: 0 auto; line-height: 1.5;">
                Karyawan ini terprediksi memiliki kemungkinan sebesar <b>{prob*100:.1f}%</b> untuk mengundurkan diri. Nilai baseline rata-rata historis adalah 16.1%.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    # Faktor Pendorong & Faktor Protektif
    st.subheader("🔍 Faktor Pendorong & Faktor Protektif")
    
    st.markdown("##### ⚠️ Pendorong Risiko (Meningkatkan Kerentanan)")
    if not drivers.empty:
        for idx, r in drivers.head(4).iterrows():
            st.markdown(f"- **{r['Label']}**: menambah nilai kontribusi sebesar `+{r['Contribution']:.3f}`.")
    else:
        st.markdown("- *Tidak ditemukan pendorong risiko signifikan.*")
        
    st.markdown("##### 🛡️ Faktor Protektif (Menjaga Kestabilan)")
    if not protective.empty:
        for idx, r in protective.head(4).iterrows():
            st.markdown(f"- **{r['Label']}**: mengurangi nilai kontribusi sebesar `{r['Contribution']:.3f}`.")
    else:
        st.markdown("- *Tidak ditemukan faktor protektif signifikan.*")
        
    st.write("---")
    
    # Rencana Retensi HR
    st.subheader("📋 Rencana Retensi HR yang Dapat Ditindaklanjuti")
    
    has_recs = False
    
    # 1. Overtime
    if overtime == "Yes":
        has_recs = True
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>⚠️ Kontrol Lembur & Pencegahan Kejenuhan (Burnout)</div>
                <div class='plan-card-body'>
                    Lembur wajib aktif terdeteksi. Batasi jam lembur mingguan, tawarkan jam kerja fleksibel, atau berikan hari libur pengganti untuk meredakan kelelahan kerja fisik dan mental.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # 2. Compensation
    if monthly_income < 4000:
        has_recs = True
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>💰 Penyesuaian Kompensasi & Gaji Pokok</div>
                <div class='plan-card-body'>
                    Gaji bulanan karyawan berada di tier bawah (${0:,.2f}). Segera lakukan evaluasi kesetaraan gaji internal dan pertimbangkan opsi penyesuaian gaji pokok atau penyaluran bonus insentif.
                </div>
            </div>
        """.format(monthly_income), unsafe_allow_html=True)
        
    # 3. Satisfaction
    if job_satisfaction <= 2 or env_satisfaction <= 2:
        has_recs = True
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>⭐ Evaluasi Kepuasan Lingkungan Kerja</div>
                <div class='plan-card-body'>
                    Skor kepuasan kerja/lingkungan di bawah batas optimal. Jadwalkan sesi 1-on-1 tatap muka untuk mendengar keluhan spesifik, menata ulang pembagian kerja, atau tawarkan opsi mutasi horizontal.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # 4. Promotion Gap
    if years_since_promotion >= 3:
        has_recs = True
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>📈 Pemetaan & Rencana Karir Karyawan</div>
                <div class='plan-card-body'>
                    Karyawan belum menerima kenaikan promosi selama {0} tahun terakhir. Rancang peta jalan karir jangka menengah yang konkret atau libatkan dalam inisiatif kepemimpinan proyek baru.
                </div>
            </div>
        """.format(years_since_promotion), unsafe_allow_html=True)
        
    # 5. Work Life Balance
    if work_life_balance <= 2:
        has_recs = True
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>🧘 Keseimbangan Kerja-Hidup (Work-Life Balance)</div>
                <div class='plan-card-body'>
                    Keseimbangan kerja-hidup tergolong buruk. Terapkan batasan yang jelas agar karyawan tidak menerima tugas kantor setelah jam kerja usai, tinjau beban kerja saat ini, dan dorong penggunaan cuti tahunan.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    if not has_recs:
        st.markdown("""
            <div class='plan-card'>
                <div class='plan-card-title'>ℹ️ Pertahankan Pola Manajemen Sehat</div>
                <div class='plan-card-body'>
                    Seluruh metrik dan indikator profil karyawan menunjukkan status yang aman dan sehat. Lanjutkan evaluasi rutin standar dan apresiasi kinerja secara berkala.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
