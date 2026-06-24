import streamlit as st
import os
from utils.styles import inject_custom_css

# Konfigurasi Halaman
st.set_page_config(page_title="Model ANN Deep Learning", page_icon="🧠", layout="wide")

# Suntikkan gaya CSS kustom
inject_custom_css()

st.markdown("<div class='app-title'>🧠 Arsitektur Jaringan Saraf Tiruan (ANN)</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Penjelasan teknis tentang model klasifikasi deep artificial neural network (ANN) yang digunakan.</div>", unsafe_allow_html=True)

# 1. Topologi Layer Model (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Topologi Layer Jaringan Saraf")
st.write(
    "Diagram alir berikut menunjukkan jalur data sekuensial dari dimensi fitur input "
    "hingga menghasilkan nilai probabilitas attrisi karyawan."
)

# Rendering diagram Graphviz dalam Bahasa Indonesia
dot_code = """
digraph G {
    rankdir=LR;
    node [shape=box, style="filled,rounded", fillcolor="#EEF2FF", color="#6366F1", fontname="Plus Jakarta Sans", fontsize=11, width=1.5, height=0.6];
    edge [color="#64748B", arrowhead=vee, arrowsize=0.8];
    
    in [label="Fitur Input\\n(37 Dimensi)", shape=ellipse, fillcolor="#E0E7FF", color="#4338CA"];
    d1 [label="Layer Dense 1\\n64 Neuron (ReLU)"];
    d2 [label="Layer Dense 2\\n32 Neuron (ReLU)"];
    d3 [label="Layer Dense 3\\n16 Neuron (ReLU)"];
    d4 [label="Layer Dense 4\\n8 Neuron (ReLU)"];
    out [label="Probabilitas Output\\n1 Neuron (Sigmoid)", shape=ellipse, fillcolor="#FEE2E2", color="#EF4444"];
    
    in -> d1 -> d2 -> d3 -> d4 -> out;
}
"""
st.graphviz_chart(dot_code)
st.markdown("</div>", unsafe_allow_html=True)

# 2. Ringkasan Parameter Model (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Ringkasan Parameter Model Keras")

st.markdown(
    """
    <table class='tech-table' style='width:100%; border-collapse:collapse; margin-top:1rem;'>
        <thead>
            <tr style='background-color:#F8FAFC;'>
                <th style='border:1px solid #E2E8F0; padding:10px; text-align:left; color:#1E1B4B;'>Layer</th>
                <th style='border:1px solid #E2E8F0; padding:10px; text-align:left; color:#1E1B4B;'>Tipe</th>
                <th style='border:1px solid #E2E8F0; padding:10px; text-align:left; color:#1E1B4B;'>Dimensi Output</th>
                <th style='border:1px solid #E2E8F0; padding:10px; text-align:left; color:#1E1B4B;'>Jumlah Parameter</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>input_layer</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Input</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 37)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>0</td>
            </tr>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>dense</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Dense</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 64)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>2,432</td>
            </tr>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>dense_1</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Dense</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 32)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>2,080</td>
            </tr>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>dense_2</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Dense</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 16)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>528</td>
            </tr>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>dense_3</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Dense</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 8)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>136</td>
            </tr>
            <tr>
                <td style='border:1px solid #E2E8F0; padding:10px;'><b>dense_4</b></td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>Dense (Output)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>(None, 1)</td>
                <td style='border:1px solid #E2E8F0; padding:10px;'>9</td>
            </tr>
        </tbody>
    </table>
    
    <br>
    
    <b>Statistik Kunci:</b>
    <ul>
        <li><b>Total Parameter Model:</b> 15.557</li>
        <li><b>Bobot yang Dapat Dilatih (Trainable Parameters):</b> 5.185</li>
        <li><b>Metode Optimasi (Optimizer):</b> Adam (learning_rate=0.001)</li>
        <li><b>Penghentian Dini (Early Stopping):</b> Mengawasi nilai <code>val_loss</code> dengan batas kesabaran (patience) 15 epoch, menghentikan pelatihan secara otomatis ketika performa tidak menunjukkan perbaikan demi menghindari overfitting.</li>
    </ul>
    """, 
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)

# 3. Konsep Desain Model (Untuk Pembelajaran)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Konsep Desain Jaringan Saraf")

tab_explain1, tab_explain2, tab_explain3 = st.tabs([
    "⚙️ Kenapa Memilih ANN?", 
    "🛡️ Regularisasi Dropout", 
    "📉 Fungsi Loss & Optimasi"
])

with tab_explain1:
    st.write(
        "**Alasan Penggunaan Artificial Neural Network (ANN):**\n"
        "- **Hubungan Non-Linear Kompleks:** Model berbasis pohon keputusan memotong fitur secara kaku, sedangkan jaringan saraf tiruan mampu membangun batas keputusan yang halus dan non-linear di ruang multidimensi.\n"
        "- **Optimasi Nilai Recall:** Kami menggunakan **class-weight adjustment** selama pelatihan. Hal ini memaksa ANN untuk fokus memaksimalkan nilai **Recall (63.83%)** sehingga kita dapat menjaring sebanyak mungkin karyawan yang berisiko keluar."
    )
    
with tab_explain2:
    st.write(
        "**Peran Penting Layer Dropout:**\n"
        "- **Mengatasi Overfitting:** Dataset tabular dengan jumlah baris terbatas (seperti 1.470 baris) sangat rentan terhadap overfitting di jaringan saraf.\n"
        "- **Deaktivasi Neuron Acak:** Dengan menonaktifkan sebagian neuron secara acak (misalnya 20%) di setiap langkah pelatihan, model dipaksa mempelajari pola yang bersifat umum daripada sekadar menghafal baris data individual."
    )
    
with tab_explain3:
    st.write(
        "**Mekanisme Loss dan Optimasi:**\n"
        "- **Binary Crossentropy:** Fungsi loss standar untuk klasifikasi biner (0 atau 1). Fungsi ini memberikan penalti besar jika model salah memprediksi dengan tingkat keyakinan tinggi, sehingga mematangkan kalibrasi probabilitas output.\n"
        "- **Adam Optimizer:** Secara dinamis mengatur learning rate untuk setiap parameter selama pelatihan, mempercepat konvergensi menuju solusi optimal."
    )
st.markdown("</div>", unsafe_allow_html=True)

# 4. Kelebihan & Kekurangan (Lebar Penuh)
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.subheader("Kelebihan & Kekurangan Model")

st.markdown("""
    **Kelebihan:**
    - **Recall Tinggi (63.83%):** Sukses mendeteksi hampir 2/3 dari total karyawan yang berisiko keluar, memberikan waktu bagi HR untuk bertindak.
    - **Probabilitas Halus:** Output berupa probabilitas kontinu dari 0 hingga 1, ideal sebagai instrumen risk-profiling.
    
    **Kekurangan:**
    - **Presisi Rendah (45.45%):** Menghasilkan alarm palsu (false alarm) yang lebih sering dibandingkan Regresi Logistik standar.
    - **Karakter Black-Box:** Nilai bobot jaringan sangat rapat dan sulit diinterpretasikan secara langsung. Kami mengatasinya dengan menggunakan koefisien Regresi Logistik sebagai penaksir pengaruh fitur lokal (surrogate model).
""")
st.markdown("</div>", unsafe_allow_html=True)
