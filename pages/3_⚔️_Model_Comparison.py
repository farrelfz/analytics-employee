import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.styles import inject_custom_css

# Konfigurasi Halaman
st.set_page_config(page_title="Matriks Perbandingan Model", page_icon="⚔️", layout="wide")

# Suntikkan gaya CSS kustom
inject_custom_css()

st.markdown("<div class='app-title'>⚔️ Matriks Komparasi Model Prediktif</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Tolok ukur performa machine learning tradisional terhadap deep feedforward neural network.</div>", unsafe_allow_html=True)

# Membuat DataFrame Metrik Evaluasi dalam Bahasa Indonesia
metrics_data = [
    {"Model": "Logistic Regression", "Akurasi": 86.39, "Presisi": 0.6400, "Recall": 0.3404, "F1-Score": 0.4444, "ROC-AUC": 0.8147},
    {"Model": "Random Forest", "Akurasi": 84.69, "Presisi": 0.5833, "Recall": 0.1489, "F1-Score": 0.2373, "ROC-AUC": 0.8173},
    {"Model": "XGBoost", "Akurasi": 80.61, "Presisi": 0.4194, "Recall": 0.5532, "F1-Score": 0.4771, "ROC-AUC": 0.7749},
    {"Model": "ANN", "Akurasi": 81.97, "Presisi": 0.4545, "Recall": 0.6383, "F1-Score": 0.5310, "ROC-AUC": 0.7918}
]
df_metrics = pd.DataFrame(metrics_data)

# 1. Tabel Performa Evaluasi (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Matriks Performa Model Evaluasi (Data Uji)")

st.dataframe(
    df_metrics.style.format({
        "Akurasi": "{:.2f}%",
        "Presisi": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
        "ROC-AUC": "{:.4f}"
    }).highlight_max(subset=["Recall", "F1-Score", "ROC-AUC"], color="#D1FAE5")
      .highlight_max(subset=["Akurasi", "Presisi"], color="#DBEAFE"),
    use_container_width=True,
    hide_index=True
)
st.caption("Keterangan warna: Sorotan hijau menandakan skor tertinggi pada metrik bisnis sensitif (Recall/F1/AUC). Sorotan biru menandakan skor tertinggi pada Akurasi/Presisi.")
st.markdown("</div>", unsafe_allow_html=True)

# 2. Wawasan Analisis Data Science (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Analisis Mendalam Performa Model")

tab_explain1, tab_explain2, tab_explain3 = st.tabs([
    "⚖️ Ilusi Akurasi & Imbalance", 
    "📈 Keunggulan Recall ANN", 
    "💼 Implikasi Strategi HR"
])

with tab_explain1:
    st.write(
        "**Ketidakseimbangan Data & Ilusi Akurasi:**\n"
        "- **Akurasi Tinggi Semu:** Model **Logistic Regression** mencapai tingkat akurasi tertinggi sebesar **86.39%**. "
        "Namun, ini adalah bentuk ilusi akurasi karena adanya ketidakseimbangan kelas target (~16% attrisi). "
        "Model yang secara naif menebak 'Tidak Keluar' untuk seluruh karyawan akan mendapat akurasi 84%, tetapi tidak berguna untuk keputusan bisnis.\n"
        "- **Kelemahan Deteksi:** Logistic Regression memiliki **Recall yang rendah (34.04%)**, artinya model ini melewatkan 66% dari total karyawan yang sebenarnya akan resign."
    )

with tab_explain2:
    st.write(
        "**Mengapa Jaringan Saraf Tiruan (ANN) Unggul pada Recall:**\n"
        "- **Recall Tertinggi:** Model **ANN** meraih nilai **Recall tertinggi (63.83%)** dan **F1-Score tertinggi (0.5310)**.\n"
        "- **Mekanisme Pembobotan Kelas:** Melalui pembobotan kelas target (*class-weight*) yang disesuaikan saat proses fitting, jaringan saraf dipaksa mempelajari pola-pola kritis dari kelas minoritas (karyawan keluar), menekan seminimal mungkin kejadian salah deteksi (*False Negatives*)."
    )

with tab_explain3:
    st.write(
        "**Implikasi Bisnis pada Strategi Retensi HR:**\n"
        "- **Recall Adalah Kunci:** Dalam isu turnover karyawan, **Recall adalah metrik paling krusial**. Jika model gagal mendeteksi karyawan yang ingin keluar (*False Negative*), karyawan tersebut akan resign dan perusahaan harus menanggung biaya rekrutmen ulang yang mahal.\n"
        "- **Keuntungan Preventif:** Sebaliknya, jika model salah menandai karyawan yang sebenarnya setia (*False Positive*/Alarm Palsu), perusahaan hanya kehilangan waktu untuk melakukan sesi diskusi HR tatap muka—tindakan yang justru produktif untuk meningkatkan keharmonisan kerja. Oleh karena itu, **model ANN adalah model paling direkomendasikan untuk digunakan di platform operasional**."
    )
st.markdown("</div>", unsafe_allow_html=True)

# 3. Grafik Batang Perbandingan Metrik (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Visualisasi Metrik Evaluasi")
selected_metric = st.selectbox("Pilih Metrik untuk Visualisasi:", ["Recall", "F1-Score", "ROC-AUC", "Akurasi", "Presisi"])

fig_bar = px.bar(
    df_metrics,
    x='Model',
    y=selected_metric,
    color='Model',
    color_discrete_map={
        'ANN': '#F43F5E',                # Rose (Vibrant Winner)
        'XGBoost': '#6366F1',            # Indigo
        'Logistic Regression': '#0EA5E9', # Sky Blue
        'Random Forest': '#94A3B8'       # Slate
    },
    text_auto=True
)
fig_bar.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=40, r=40, t=10, b=20),
    showlegend=False,
    yaxis_title=selected_metric,
    xaxis_title=""
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
