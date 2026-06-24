import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
from utils.styles import inject_custom_css

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Attrisi Karyawan",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suntikkan gaya CSS kustom
inject_custom_css()

# Header Aplikasi
st.markdown("<div class='app-title'>🏢 Platform Analisis Attrisi Karyawan</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>HR Intelligence Suite & Alat Diagnosis Deep Learning</div>", unsafe_allow_html=True)

# Muat Dataset untuk KPI & Dashboard
@st.cache_data
def load_hr_data(file_path: str = 'data/Employee-Attrition.csv') -> pd.DataFrame:
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            logger.error(f"Dataset path not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_hr_data()

if df.empty:
    st.error("File data tidak ditemukan di path `data/Employee-Attrition.csv`. Silakan periksa kembali struktur folder Anda.")
else:
    # Terjemahkan kolom Attrition & Gender untuk tampilan chart yang konsisten
    df_plot = df.copy()
    df_plot['Attrisi'] = df_plot['Attrition'].map({'Yes': 'Ya', 'No': 'Tidak'})
    df_plot['Jenis Kelamin'] = df_plot['Gender'].map({'Female': 'Perempuan', 'Male': 'Laki-laki'})
    
    # Hitung metrik HR untuk KPI
    total_employees = len(df)
    attrition_yes = (df['Attrition'] == 'Yes').sum()
    retention_count = total_employees - attrition_yes
    attrition_rate = (attrition_yes / total_employees) * 100
    overtime_rate = (df['OverTime'] == 'Yes').mean() * 100

    # Tampilkan Grid Metrik Utama (5 KPIs dalam 5 kolom)
    st.markdown(f"""
        <div class='metric-grid'>
            <div class='metric-card'>
                <div class='metric-title'>Total Karyawan</div>
                <div class='metric-value'>{total_employees:,}</div>
            </div>
            <div class='metric-card alert-card'>
                <div class='metric-title'>Total Attrisi</div>
                <div class='metric-value'>{attrition_yes:,}</div>
            </div>
            <div class='metric-card success-card'>
                <div class='metric-title'>Karyawan Bertahan</div>
                <div class='metric-value'>{retention_count:,}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-title'>Tingkat Attrisi</div>
                <div class='metric-value'>{attrition_rate:.1f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Menampilkan Panduan Platform & Informasi secara terbuka (Natural)
    st.markdown("""
        <div class='content-container' style='background-color:#FAF5FF; border-left: 5px solid #818CF8; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem;'>
                <!-- Kolom 1: Ikhtisar Platform -->
                <div style='padding-right: 0.5rem;'>
                    <h4 style='color:#4F46E5 !important; margin-top:0; margin-bottom:0.75rem; font-family:"Plus Jakarta Sans", sans-serif; font-weight:700;'>ℹ️ Ikhtisar Platform</h4>
                    <p style='font-size:0.92rem; line-height:1.6; color:#475569; margin-bottom:1rem; font-family:"Inter", sans-serif;'>
                        Platform ini berfungsi sebagai alat pendukung keputusan profesional yang dirancang untuk manajer HR dan praktisi data science.
                        Dengan menggunakan metrik HR standar industri dan klasifikasi deep learning, platform ini menerjemahkan data personel menjadi wawasan yang dapat ditindaklanjuti.
                    </p>
                    <div style='background-color:#FFFFFF; padding:1rem; border-radius:8px; border:1px solid #E2E8F0; font-size:0.88rem; font-family:"Inter", sans-serif; color:#475569;'>
                        <b style='color:#0F172A;'>Bagian Utama Platform:</b>
                        <ul style='margin-top:0.5rem; margin-bottom:0; padding-left:1.2rem; line-height:1.5;'>
                            <li style='margin-bottom:0.3rem;'><b style='color:#0F172A;'>Dashboard:</b> Halaman utama ini menyajikan visualisasi tingkat tinggi tren karyawan.</li>
                            <li style='margin-bottom:0.3rem;'><b style='color:#0F172A;'>Galeri Visualisasi:</b> Eksplorasi grafik interaktif mendalam memetakan beban kerja, demografi, dan kesejahteraan karyawan.</li>
                            <li style='margin-bottom:0.3rem;'><b style='color:#0F172A;'>Arsitektur ANN:</b> Pahami rancangan jaringan saraf tiruan (ANN) yang digunakan untuk mendeteksi risiko pengunduran diri.</li>
                            <li style='margin-bottom:0.3rem;'><b style='color:#0F172A;'>Komparasi Model:</b> Bandingkan performa klasifikasi tradisional (Regresi Logistik, Random Forest, XGBoost) dengan ANN.</li>
                            <li style='margin-bottom:0;'><b style='color:#0F172A;'>Diagnosis Prediktif:</b> Uji profil karyawan individu secara langsung untuk menilai risiko resign serta mendapatkan rekomendasi rencana retensi.</li>
                        </ul>
                    </div>
                </div>
                <!-- Kolom 2: Panduan Interaktif -->
                <div>
                    <h4 style='color:#4F46E5 !important; margin-top:0; margin-bottom:0.75rem; font-family:"Plus Jakarta Sans", sans-serif; font-weight:700;'>🎯 Panduan Interaktif</h4>
                    <p style='font-size:0.92rem; line-height:1.6; color:#475569; margin-bottom:1rem; font-family:"Inter", sans-serif;'>Aplikasi ini dirancang untuk menjembatani keputusan bisnis HR dengan analisis data science yang mendalam:</p>
                    <ul style='padding-left:1.2rem; line-height:1.6; font-family:"Inter", sans-serif; color:#475569;'>
                        <li style='margin-bottom:0.5rem;'><b style='color:#0F172A;'>Fokus Bisnis:</b> Memahami indikator utama pengunduran diri karyawan seperti <i>Jam Lembur Wajib</i>, <i>Keseimbangan Kerja-Hidup</i>, dan <i>Keadilan Gaji (Compa-Ratio)</i>.</li>
                        <li style='margin-bottom:0;'><b style='color:#0F172A;'>Fokus Data Science:</b> Mengamati performa model, trade-off bobot kelas (Presisi vs. Recall), dan visualisasi kontribusi fitur lokal dengan nilai koefisien model.</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write("---")

    # Grid Grafik Distribusi & Demografi (2 Kolom)
    col1, col2 = st.columns(2)
    color_map = {'Ya': '#F43F5E', 'Tidak': '#3B82F6'} # Coral Red & Royal Blue
    
    with col1:
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("Distribusi Attrisi Berdasarkan Departemen")
        
        dept_data = df_plot.groupby(['Department', 'Attrisi']).size().reset_index(name='Jumlah')
        dept_total = df_plot.groupby('Department').size().reset_index(name='Total')
        dept_data = dept_data.merge(dept_total, on='Department')
        dept_data['Persentase'] = (dept_data['Jumlah'] / dept_data['Total']) * 100
        
        fig_dept = px.bar(
            dept_data, 
            x='Department', 
            y='Jumlah', 
            color='Attrisi',
            barmode='group',
            color_discrete_map=color_map,
            text=dept_data.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']:.1f}%)", axis=1),
            category_orders={'Attrisi': ['Tidak', 'Ya']},
            labels={'Department': 'Departemen', 'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig_dept.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title="",
            height=280
        )
        fig_dept.update_traces(textposition='outside', textfont_size=9)
        st.plotly_chart(fig_dept, use_container_width=True)
        
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi Departemen:</b> Departemen <b>Penjualan (Sales)</b> menunjukkan tingkat persentase attrisi tertinggi 
                (sekitar 20.6%), diikuti oleh HR (19.0%). Penelitian & Pengembangan (R&D) menunjukkan stabilitas terbaik dengan tingkat attrisi 13.8%.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("Demografi Karyawan (Usia vs. Jenis Kelamin)")
        
        df_demo = df_plot.copy()
        df_demo['Kelompok Usia'] = pd.cut(df_demo['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
        demo_data = df_demo.groupby(['Kelompok Usia', 'Jenis Kelamin', 'Attrisi']).size().reset_index(name='Jumlah')
        
        fig_demo = px.bar(
            demo_data,
            x='Kelompok Usia',
            y='Jumlah',
            color='Attrisi',
            facet_col='Jenis Kelamin',
            barmode='stack',
            color_discrete_map=color_map,
            category_orders={'Attrisi': ['Tidak', 'Ya']},
            labels={'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig_demo.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            xaxis_title="Kelompok Usia",
            height=280
        )
        st.plotly_chart(fig_demo, use_container_width=True)
        
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi Demografis:</b> Karyawan muda pada rentang usia 18-25 tahun menunjukkan proporsi attrisi yang paling tinggi di kedua jenis kelamin, yang mengindikasikan adanya friksi adaptasi awal karir.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Distribusi Peran Pekerjaan (Lebar Penuh)
    st.markdown("<div class='content-container'>", unsafe_allow_html=True)
    st.subheader("Rincian Attrisi Berdasarkan Peran Pekerjaan")
    
    role_data = df_plot.groupby(['JobRole', 'Attrisi']).size().reset_index(name='Jumlah')
    role_total = df_plot.groupby('JobRole').size().reset_index(name='Total')
    role_data = role_data.merge(role_total, on='JobRole')
    role_data['Tingkat'] = (role_data['Jumlah'] / role_data['Total']) * 100
    
    role_indonesian_map = {
        'Sales Executive': 'Eksekutif Penjualan',
        'Research Scientist': 'Ilmuwan Peneliti',
        'Laboratory Technician': 'Teknisi Laboratorium',
        'Manufacturing Director': 'Direktur Manufaktur',
        'Healthcare Representative': 'Perwakilan Layanan Kesehatan',
        'Manager': 'Manajer',
        'Sales Representative': 'Perwakilan Penjualan',
        'Research Director': 'Direktur Penelitian',
        'Human Resources': 'Sumber Daya Manusia'
    }
    role_data['Peran Pekerjaan'] = role_data['JobRole'].map(role_indonesian_map)
    
    role_rates = role_data[role_data['Attrisi'] == 'Ya'].sort_values(by='Tingkat', ascending=False)
    role_order = role_rates['Peran Pekerjaan'].tolist()
    
    fig_role = px.bar(
        role_data,
        y='Peran Pekerjaan',
        x='Jumlah',
        color='Attrisi',
        barmode='stack',
        orientation='h',
        color_discrete_map=color_map,
        category_orders={'Peran Pekerjaan': role_order, 'Attrisi': ['Tidak', 'Ya']},
        text=role_data.apply(lambda r: f"{r['Tingkat']:.1f}%" if r['Attrisi'] == 'Ya' else "", axis=1),
        labels={'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
    )
    
    fig_role.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=150, r=40, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title="Jumlah Karyawan",
        yaxis_title="",
        height=380
    )
    fig_role.update_traces(textposition='inside', insidetextanchor='end', textfont_size=9)
    st.plotly_chart(fig_role, use_container_width=True)
    
    st.markdown("""
        <div class='highlight-box'>
            <b>Interpretasi Peran Pekerjaan:</b> Posisi operasional lapangan seperti <b>Perwakilan Penjualan (Sales Representative)</b> (39.8% attrisi), 
            <b>Teknisi Laboratorium (Laboratory Technician)</b> (23.9%), dan <b>Spesialis SDM (HR Specialist)</b> (23.1%) menunjukkan tingkat pergantian karyawan yang sangat tinggi. Sebaliknya, posisi eksekutif dan manajerial menunjukkan retensi yang luar biasa baik (biasanya di bawah 5%).
        </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")

    # Pratinjau Dataset Karyawan
    st.markdown("### 📋 Pratinjau Dataset Karyawan")
    st.dataframe(df.head(5), use_container_width=True)
