# Platform Analisis Attrisi Karyawan

Proyek sains data akademis profesional untuk mendiagnosis dan memprediksi risiko attrisi (pengunduran diri) karyawan. Proyek ini memodelkan attrisi menggunakan data historis HR dan menampilkannya dalam aplikasi Streamlit interaktif multi-halaman.

---

## Struktur Proyek

```text
├── data/
│   ├── Employee-Attrition.csv              # IBM HR Analytics Attrition Dataset
│   └── employee_attrition_preprocessed.csv # Preprocessed dataset untuk pelatihan model
│
├── docs/
│   └── Mini Project Data Science.docx      # Laporan dokumentasi proyek
│
├── model/
│   ├── best_model.keras                    # Model Jaringan Saraf Tiruan (Keras ANN) yang telah dilatih
│   └── mp1_artifacts.pkl                   # Model ML & objek standardisasi (joblib/pickle)
│
├── pages/
│   ├── # File Tree: Analytics-Employee.md  # Dokumentasi pohon direktori
│   ├── 1_📈_Visualization.py               # Galeri analisis visual eksploratif (EDA)
│   ├── 2_🧠_ANN_Model.py                   # Penjelasan teknis topologi model ANN
│   ├── 3_⚔️_Model_Comparison.py            # Komparasi performa antar-model ML/DL
│   └── 4_🎯_Prediction.py                  # Diagnosis risiko pengunduran diri karyawan individu
│
├── utils/
│   ├── feature_builder.py                  # Fungsi perakit vektor fitur masukan model
│   ├── model_loader.py                     # Modul pemuatan model & penanganan deserialisasi Keras
│   ├── plots.py                            # Kumpulan fungsi pembuat grafik interaktif
│   ├── preprocessing.py                    # Kalkulator preproses data & kontribusi lokal fitur
│   └── styles.py                           # Modul injeksi kode CSS kustom aplikasi
│
├── app.py                                  # Halaman beranda platform
├── requirements.txt                        # Daftar dependensi Python
├── test.py                                 # Skrip penguji kesehatan server
└── README.md                               # Dokumentasi proyek (file ini)
```

---

## Variabel Target & Fitur Model

Model prediktif menggunakan variabel target biner serta 23 fitur dasar (yang kemudian di-preproses menjadi 37 dimensi fitur masukan model).

### Variabel Target
*   **`Attrition`** (Status Attrisi): Menunjukkan apakah karyawan keluar dari perusahaan (`Yes` / Ya) atau tetap bertahan (`No` / Tidak).

### Fitur Numerik & Ordinal (16 Fitur)
1.  **`Age` (Usia)**: Usia karyawan (rentang 18 - 65 tahun).
2.  **`DistanceFromHome` (Jarak dari Rumah)**: Jarak rumah karyawan ke kantor dalam kilometer (km).
3.  **`Education` (Tingkat Pendidikan)**: Tingkat pendidikan formal (1: SMA, 2: Diploma, 3: S1/Sarjana, 4: S2/Magister, 5: S3/Doktor).
4.  **`EnvironmentSatisfaction` (Kepuasan Lingkungan)**: Kepuasan terhadap suasana kantor (1: Rendah, 2: Sedang, 3: Tinggi, 4: Sangat Tinggi).
5.  **`JobInvolvement` (Keterlibatan Kerja)**: Tingkat keterikatan pada pekerjaan (1: Rendah, 2: Sedang, 3: Tinggi, 4: Sangat Tinggi).
6.  **`JobSatisfaction` (Kepuasan Kerja)**: Kepuasan terhadap tugas kerja (1: Rendah, 2: Sedang, 3: Tinggi, 4: Sangat Tinggi).
7.  **`MonthlyIncome` (Pendapatan Bulanan)**: Gaji bersih per bulan dalam USD ($).
8.  **`NumCompaniesWorked` (Jumlah Perusahaan Sebelumnya)**: Banyaknya perusahaan tempat bekerja sebelum bergabung.
9.  **`PercentSalaryHike` (Persentase Kenaikan Gaji)**: Persentase kenaikan gaji tahunan (%).
10. **`RelationshipSatisfaction` (Kepuasan Hubungan)**: Hubungan interpersonal dengan rekan kerja (1: Rendah, 2: Sedang, 3: Tinggi, 4: Sangat Tinggi).
11. **`StockOptionLevel` (Tingkat Opsi Saham)**: Hak opsi kepemilikan saham perusahaan (skala 0 - 3).
12. **`TotalWorkingYears` (Total Masa Kerja)**: Total pengalaman kerja profesional dalam tahun.
13. **`TrainingTimesLastYear` (Jumlah Pelatihan Tahun Lalu)**: Jumlah pelatihan yang diikuti tahun lalu.
14. **`WorkLifeBalance` (Keseimbangan Kerja-Hidup)**: Evaluasi WLB (1: Buruk, 2: Cukup, 3: Baik, 4: Terbaik).
15. **`YearsAtCompany` (Lama Bekerja di Perusahaan Ini)**: Durasi bekerja di perusahaan saat ini dalam tahun.
16. **`YearsSinceLastPromotion` (Tahun Sejak Promosi Terakhir)**: Jumlah tahun sejak kenaikan pangkat terakhir.

### Fitur Kategorikal (One-Hot Encoded)
17. **`BusinessTravel` (Perjalanan Bisnis)**: Kategori frekuensi dinas luar (`Travel_Rarely` / Jarang, `Travel_Frequently` / Sering, `Non-Travel` / Tidak Pernah).
18. **`Department` (Departemen)**: Departemen karyawan (`Research & Development`, `Sales`, `Human Resources`).
19. **`EducationField` (Bidang Pendidikan)**: Jurusan pendidikan (`Life Sciences`, `Medical`, `Marketing`, `Technical Degree`, `Human Resources`, `Other`).
20. **`Gender` (Jenis Kelamin)**: Gender karyawan (`Male` / Laki-laki, `Female` / Perempuan).
21. **`MaritalStatus` (Status Pernikahan)**: Status perkawinan (`Married` / Menikah, `Single` / Lajang, `Divorced` / Cerai).
22. **`OverTime` (Lembur)**: Karyawan bekerja lembur atau tidak (`Yes` / Ya, `No` / Tidak).
23. **`JobRole` (Peran Pekerjaan)**: Posisi jabatan kerja karyawan (misalnya `Sales Executive`, `Research Scientist`, `Laboratory Technician`, `Manager`, dll).

---

## Evaluasi Model & Hasil

Karena ketidakseimbangan kelas target (~16% data historis attrisi), metrik **Recall** dan **F1-Score** dijadikan parameter utama evaluasi model bisnis retensi HR. Model Keras ANN dilatih khusus dengan pembobotan kelas (*class-weight*) untuk memprioritaskan sensitivitas penangkapan karyawan yang ingin keluar.

### Perbandingan Kinerja Model (Data Uji)

| Model | Akurasi | Presisi | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 86.39% | 0.6400 | 0.3404 | 0.4444 | 0.8147 |
| **Random Forest** | 84.69% | 0.5833 | 0.1489 | 0.2373 | 0.8173 |
| **XGBoost** | 80.61% | 0.4194 | 0.5532 | 0.4771 | 0.7749 |
| **ANN (Deep Learning)** | **81.97%** | **0.4545** | **0.6383** | **0.5310** | **0.7918** |

### Temuan Utama
- **Model ANN mengungguli model ML klasik** pada metrik bisnis sensitif yaitu Recall (63.83%) dan F1-Score (0.5310), menjadikannya model yang paling direkomendasikan untuk platform HR.
- **Lembur Wajib (Overtime) dan Pendapatan Rendah** terbukti menjadi dua pendorong utama keputusan pengunduran diri karyawan.

---

## Panduan Instalasi & Setup

1.  **Aktifkan Lingkungan Kerja Conda:**
    ```bash
    conda activate employee
    ```

2.  **Pasang Dependensi Python:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Server Dashboard Streamlit:**
    ```bash
    streamlit run app.py
    ```

4.  **Buka di Browser Anda:**
    Akses URL lokal `http://localhost:8501` untuk menjelajahi platform.
# analytics-employee
