import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from utils.styles import inject_custom_css
from utils.plots import plot_correlation_heatmap

# Konfigurasi Halaman
st.set_page_config(page_title="Visualisasi Eksploratif HR", page_icon="📈", layout="wide")

# Suntikkan gaya CSS kustom
inject_custom_css()

st.markdown("<div class='app-title'>📈 Galeri Visualisasi Eksploratif</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Analisis interaktif memetakan beban kerja, demografi, dan profil kompensasi karyawan.</div>", unsafe_allow_html=True)

# Muat data
@st.cache_data
def load_data(file_path: str = 'data/Employee-Attrition.csv') -> pd.DataFrame:
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("Dataset tidak ditemukan. Harap letakkan file `Employee-Attrition.csv` di folder `data/`.")
else:
    # Terjemahkan data agar labels di grafik berbahasa Indonesia
    df_plot = df.copy()
    df_plot['Attrisi'] = df_plot['Attrition'].map({'Yes': 'Ya', 'No': 'Tidak'})
    df_plot['Jenis Kelamin'] = df_plot['Gender'].map({'Female': 'Perempuan', 'Male': 'Laki-laki'})
    df_plot['Lembur Wajib'] = df_plot['OverTime'].map({'Yes': 'Ya', 'No': 'Tidak'})
    
    # Konfigurasi Tab
    t1, t2, t3, t4, t5 = st.tabs([
        "👤 Demografi & Ikhtisar",
        "🏢 Analisis Departemen & Peran",
        "💰 Analisis Gaji & Masa Kerja",
        "⭐ Kepuasan & Kesejahteraan",
        "🔗 Dinamika Korelasi"
    ])
    
    # Tema Grafik
    chart_theme = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=320
    )
    
    color_map = {'Ya': '#F43F5E', 'Tidak': '#3B82F6'}
    
    # ------------------ TAB 1: DEMOGRAFI & IKHTISAR ------------------
    with t1:
        st.markdown("### Demografi Personel dan Ikhtisar Attrisi")
        
        # 1. Attrisi Pie
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("1. Proporsi Attrisi Karyawan")
        attr_counts = df_plot['Attrisi'].value_counts()
        fig1 = go.Figure(data=[go.Pie(
            labels=attr_counts.index, 
            values=attr_counts.values, 
            hole=.4,
            marker=dict(colors=['#3B82F6', '#F43F5E']),
            textinfo='label+percent'
        )])
        fig1.update_layout(**chart_theme)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Tingkat attrisi dasar (baseline) perusahaan adalah sebesar <b>16.1%</b> (237 karyawan keluar dari total 1.470 karyawan). Ini adalah acuan historis penting untuk model prediktif kita.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 2. Histogram Usia
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("2. Distribusi Usia Karyawan Berdasarkan Attrisi")
        fig2 = px.histogram(
            df_plot, 
            x="Age", 
            color="Attrisi", 
            marginal="box",
            barmode="overlay",
            color_discrete_map=color_map,
            opacity=0.75,
            labels={'Age': 'Usia Karyawan', 'count': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig2.update_layout(**chart_theme)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Pengunduran diri memuncak pada rentang usia produktif awal yaitu <b>25-35 tahun</b>. Median usia karyawan yang keluar adalah 32 tahun, sedangkan karyawan yang bertahan memiliki median usia 36 tahun.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 3. Jenis Kelamin vs Attrisi
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("3. Distribusi Attrisi Berdasarkan Jenis Kelamin")
        gender_data = df_plot.groupby(['Jenis Kelamin', 'Attrisi']).size().reset_index(name='Jumlah')
        gender_total = df_plot.groupby('Jenis Kelamin').size().reset_index(name='Total')
        gender_data = gender_data.merge(gender_total, on='Jenis Kelamin')
        gender_data['Persentase'] = (gender_data['Jumlah'] / gender_data['Total']) * 100
        
        fig3 = px.bar(
            gender_data,
            x='Jenis Kelamin',
            y='Jumlah',
            color='Attrisi',
            barmode='group',
            color_discrete_map=color_map,
            text=gender_data.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']:.1f}%)", axis=1),
            labels={'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig3.update_layout(**chart_theme)
        fig3.update_traces(textposition='outside', textfont_size=9)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Staf laki-laki mengalami tingkat keluar yang sedikit lebih tinggi (<b>17.0%</b>) dibandingkan staf perempuan (<b>14.8%</b>). Namun, perbedaan ini relatif kecil dibandingkan dengan faktor lingkungan kerja lainnya.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
            
    # ------------------ TAB 2: ANALISIS DEPARTEMEN & PERAN ------------------
    with t2:
        st.markdown("### Profil Departemen dan Intensitas Kerja")
        
        # 4. Departemen
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("4. Distribusi Departemen")
        dept_data = df_plot.groupby(['Department', 'Attrisi']).size().reset_index(name='Jumlah')
        dept_total = df_plot.groupby('Department').size().reset_index(name='Total')
        dept_data = dept_data.merge(dept_total, on='Department')
        dept_data['Persentase'] = (dept_data['Jumlah'] / dept_data['Total']) * 100
        
        fig4 = px.bar(
            dept_data,
            x='Department',
            y='Jumlah',
            color='Attrisi',
            barmode='stack',
            color_discrete_map=color_map,
            text=dept_data.apply(lambda r: f"{r['Persentase']:.1f}%" if r['Attrisi'] == 'Ya' else "", axis=1),
            labels={'Department': 'Departemen', 'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig4.update_layout(**chart_theme)
        fig4.update_traces(textposition='inside', insidetextanchor='end', textfont_size=9)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Departemen <b>Penjualan (Sales)</b> mencatat tingkat attrisi tertinggi (20.6%). Sementara Departemen Penelitian & Pengembangan (R&D) merupakan wilayah kerja paling stabil dengan attrisi terendah (13.8%).
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 5. Lembur Wajib
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("5. Pengaruh Lembur Wajib Terhadap Attrisi")
        ot_data = df_plot.groupby(['Lembur Wajib', 'Attrisi']).size().reset_index(name='Jumlah')
        ot_total = df_plot.groupby('Lembur Wajib').size().reset_index(name='Total')
        ot_data = ot_data.merge(ot_total, on='Lembur Wajib')
        ot_data['Persentase'] = (ot_data['Jumlah'] / ot_data['Total']) * 100
        
        fig5 = px.bar(
            ot_data,
            x='Lembur Wajib',
            y='Jumlah',
            color='Attrisi',
            barmode='group',
            color_discrete_map=color_map,
            text=ot_data.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']:.1f}%)", axis=1),
            labels={'Lembur Wajib': 'Lembur Wajib', 'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig5.update_layout(**chart_theme)
        fig5.update_traces(textposition='outside', textfont_size=9)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> <b>Kebijakan lembur wajib berkorelasi sangat kuat dengan lonjakan attrisi (mencapai 30.5%)</b>. Karyawan yang diwajibkan lembur memiliki kecenderungan keluar hampir 3 kali lipat lebih besar dibanding staf biasa yang tidak lembur (10.4%).
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 3: ANALISIS GAJI & MASA KERJA ------------------
    with t3:
        st.markdown("### Pemetaan Finansial & Masa Kerja Organisasional")
        
        # 6. Gaji
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("6. Distribusi Pendapatan Bulanan")
        fig6 = px.box(
            df_plot,
            x='Attrisi',
            y='MonthlyIncome',
            color='Attrisi',
            points="outliers",
            color_discrete_map=color_map,
            labels={'MonthlyIncome': 'Pendapatan Bulanan ($)', 'Attrisi': 'Status Attrisi'}
        )
        fig6.update_layout(**chart_theme)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Karyawan yang mengundurkan diri memiliki nilai median pendapatan bulanan yang jauh lebih rendah (<b>$3.200</b>) dibandingkan dengan rekan kerja yang aktif bertahan (<b>$5.200</b>). Kelompok pendapatan rendah merupakan kelompok paling rentan terhadap risiko resign.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 7. Lama Kerja
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("7. Lama Bekerja di Perusahaan")
        fig7 = px.box(
            df_plot,
            x='Attrisi',
            y='YearsAtCompany',
            color='Attrisi',
            points="outliers",
            color_discrete_map=color_map,
            labels={'YearsAtCompany': 'Lama Kerja di Perusahaan (Tahun)', 'Attrisi': 'Status Attrisi'}
        )
        fig7.update_layout(**chart_theme)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Tingkat pengunduran diri memuncak pada <b>3 tahun pertama masa kerja (median 3 tahun)</b>, merefleksikan pentingnya penyesuaian proses onboarding. Risiko ini cenderung stabil dan menurun drastis setelah melewati masa kerja 5 tahun.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 4: KEPUASAN & KESEJAHTERAAN ------------------
    with t4:
        st.markdown("### Hubungan Kepuasan Kerja & Keseimbangan Hidup")
        
        # 8. Kepuasan Kerja
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("8. Pengaruh Kepuasan Kerja Terhadap Attrisi")
        satis_data = df_plot.groupby(['JobSatisfaction', 'Attrisi']).size().reset_index(name='Jumlah')
        satis_total = df_plot.groupby('JobSatisfaction').size().reset_index(name='Total')
        satis_data = satis_data.merge(satis_total, on='JobSatisfaction')
        satis_data['Persentase'] = (satis_data['Jumlah'] / satis_data['Total']) * 100
        
        fig8 = px.bar(
            satis_data,
            x='JobSatisfaction',
            y='Jumlah',
            color='Attrisi',
            barmode='group',
            color_discrete_map=color_map,
            text=satis_data.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']:.1f}%)", axis=1),
            labels={'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig8.update_layout(
            xaxis=dict(
                title="Tingkat Kepuasan Kerja (1=Rendah, 4=Sangat Tinggi)",
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Rendah (1)', 'Sedang (2)', 'Tinggi (3)', 'Sangat Tinggi (4)']
            ),
            **chart_theme
        )
        fig8.update_traces(textposition='outside', textfont_size=9)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Tingkat kepuasan kerja yang buruk (Level 1) menghasilkan tingkat attrisi tertinggi (22.8%). Sebaliknya, karyawan yang merasakan kepuasan sangat tinggi (Level 4) hanya memiliki tingkat attrisi 11.3%.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 9. WLB
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        st.subheader("9. Analisis Keseimbangan Kerja-Hidup (Work-Life Balance)")
        wlb_data = df_plot.groupby(['WorkLifeBalance', 'Attrisi']).size().reset_index(name='Jumlah')
        wlb_total = df_plot.groupby('WorkLifeBalance').size().reset_index(name='Total')
        wlb_data = wlb_data.merge(wlb_total, on='WorkLifeBalance')
        wlb_data['Persentase'] = (wlb_data['Jumlah'] / wlb_data['Total']) * 100
        
        fig9 = px.bar(
            wlb_data,
            x='WorkLifeBalance',
            y='Jumlah',
            color='Attrisi',
            barmode='group',
            color_discrete_map=color_map,
            text=wlb_data.apply(lambda r: f"{r['Jumlah']} ({r['Persentase']:.1f}%)", axis=1),
            labels={'Jumlah': 'Jumlah Karyawan', 'Attrisi': 'Attrisi'}
        )
        fig9.update_layout(
            xaxis=dict(
                title="Keseimbangan Kerja-Hidup (1=Buruk, 4=Terbaik)",
                tickmode='array',
                tickvals=[1, 2, 3, 4],
                ticktext=['Buruk (1)', 'Cukup (2)', 'Baik (3)', 'Terbaik (4)']
            ),
            **chart_theme
        )
        fig9.update_traces(textposition='outside', textfont_size=9)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> Karyawan dengan tingkat keseimbangan hidup yang <b>Buruk (Level 1)</b> menunjukkan persentase attrisi ekstrem sebesar <b>31.2%</b>. Ini menegaskan bahwa WLB adalah salah satu fondasi utama program retensi HR.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 5: DINAMIKA KORELASI ------------------
    with t5:
        st.markdown("### Matriks Korelasi Pearson Fitur Numerik")
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        fig10 = plot_correlation_heatmap(df)
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("""
            <div class='highlight-box'>
                <b>Interpretasi:</b> 
                Matriks korelasi ini menggambarkan kekuatan hubungan linear antar variabel numerik. Hubungan positif yang kuat terlihat jelas pada variabel masa kerja (misalnya <i>YearsAtCompany</i> dengan <i>TotalWorkingYears</i>). Model machine learning kami memanfaatkan korelasi tersembunyi ini untuk memetakan profil risiko karyawan secara menyeluruh.
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
