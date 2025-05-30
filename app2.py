import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Judul Aplikasi
st.set_page_config(page_title="Prediksi Harga Mobile 2023", layout="wide")
st.title("📱 Prediksi Harga Mobile Phone 2023")
st.markdown("""
Aplikasi ini memprediksi harga _mobile phone_ berdasarkan fitur-fiturnya menggunakan model Regresi Linear dan Decision Tree.
Masukkan fitur-fitur _mobile phone_ di bawah ini untuk melihat estimasi harganya.
""")
st.markdown("---")

# Fungsi untuk memuat model dan kolom
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Pastikan file model ada di direktori yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

@st.cache_data
def load_columns_list(columns_path):
    try:
        columns = joblib.load(columns_path)
        return columns
    except FileNotFoundError:
        st.error(f"File daftar kolom tidak ditemukan di {columns_path}. Pastikan file ini ada.")
        return None
    except Exception as e:
        st.error(f"Error memuat daftar kolom: {e}")
        return None

# Memuat model
linear_model = load_model('linear_regression_model.pkl')
tree_model = load_model('decision_tree_model.pkl')
xtrain_columns = load_columns_list('xtrain_columns.pkl')

# Jika model atau kolom gagal dimuat, hentikan eksekusi lebih lanjut di bagian input
if not linear_model or not tree_model or not xtrain_columns:
    st.warning("Model atau daftar kolom tidak berhasil dimuat. Aplikasi tidak dapat melanjutkan.")
    st.stop()


# Input dari pengguna di sidebar atau expander
with st.sidebar:
    st.header("Masukkan Fitur Mobile Phone:")

    # Kategori prosesor berdasarkan data training (dengan 'Exynos' sebagai baseline yang di-drop)
    processor_options = ['Exynos', 'Google', 'Huawei', 'IOS', 'Mediatek', 'Other', 'Snapdragon'] # Urutkan agar konsisten

    # Definisikan input
    ram_options = sorted([2, 3, 4, 6, 8, 12]) # Dari df['RAM'].value_counts()
    rom_options = sorted([16, 32, 64, 128, 256, 512]) # Dari df['ROM'].value_counts()
    battery_options = sorted([0, 1000, 1500, 2000, 2500, 4000, 4500, 5000, 5500]) # Dari pemrosesan kolom Battery

    col1, col2 = st.columns(2)
    with col1:
        ram = st.selectbox('RAM (GB)', options=ram_options, index=ram_options.index(4))
        rom = st.selectbox('ROM/Storage (GB)', options=rom_options, index=rom_options.index(128))
        battery = st.selectbox('Kapasitas Baterai (mAh)', options=battery_options, index=battery_options.index(5000))
        processor = st.selectbox('Merek Prosesor', options=processor_options, index=processor_options.index('Snapdragon'))

    with col2:
        size_cam_blkg = st.number_input('Ukuran Kamera Belakang Utama (MP)', min_value=1, max_value=200, value=50, step=1)
        total_cam_blkg = st.selectbox('Jumlah Kamera Belakang', options=[1, 2, 3, 4, 5], index=2) # Asumsi 1-5 kamera
        size_cam_dpn = st.number_input('Ukuran Kamera Depan Utama (MP)', min_value=0, max_value=100, value=12, step=1) # 0 jika tidak ada
        total_cam_dpn = st.selectbox('Jumlah Kamera Depan', options=[1, 2], index=0) # Asumsi 1-2 kamera

# Tombol untuk prediksi
predict_button = st.sidebar.button('Prediksi Harga', use_container_width=True, type="primary")

st.markdown("---")
st.subheader("Hasil Prediksi Harga (dalam Rupee India):")

if predict_button:
    if linear_model and tree_model and xtrain_columns:
        # Membuat DataFrame untuk input
        input_data = {}

        # Fitur Numerik Dasar (sesuai urutan kolom di X_train sebelum dummy processor)
        # Urutan di xtrain_columns: ['RAM', 'ROM', 'Battery', 'Size Cam Blkg', 'Total Cam Blkg', 'Size Cam Dpn', 'Total Cam Dpn', 'Upd_Processor_Google', ..., 'Upd_Processor_Snapdragon']
        
        input_data['RAM'] = ram
        input_data['ROM'] = rom
        input_data['Battery'] = battery
        input_data['Size Cam Blkg'] = size_cam_blkg
        input_data['Total Cam Blkg'] = total_cam_blkg
        input_data['Size Cam Dpn'] = size_cam_dpn
        input_data['Total Cam Dpn'] = total_cam_dpn

        # One-hot encoding untuk prosesor
        # Asumsi 'Exynos' adalah kategori baseline (dihilangkan saat drop_first=True)
        # dan urutan kategori lainnya adalah Google, Huawei, IOS, Mediatek, Other, Snapdragon
        processor_dummies = {
            'Upd_Processor_Google': 0,
            'Upd_Processor_Huawei': 0,
            'Upd_Processor_IOS': 0,
            'Upd_Processor_Mediatek': 0,
            'Upd_Processor_Other': 0,
            'Upd_Processor_Snapdragon': 0
        }
        
        if processor != 'Exynos': # Jika bukan baseline
            dummy_col_name = f"Upd_Processor_{processor}"
            if dummy_col_name in processor_dummies:
                 processor_dummies[dummy_col_name] = 1
            # else:
            #     st.warning(f"Kategori prosesor '{processor}' tidak dikenali untuk pembuatan dummy variable.")

        input_data.update(processor_dummies)
        
        # Membuat DataFrame dengan urutan kolom yang benar
        try:
            input_df = pd.DataFrame([input_data], columns=xtrain_columns)
        except Exception as e:
            st.error(f"Error saat membuat DataFrame input: {e}")
            st.error(f"Data yang coba dimasukkan: {input_data}")
            st.error(f"Kolom yang diharapkan: {xtrain_columns}")
            st.stop()


        # Melakukan prediksi
        try:
            pred_linear = linear_model.predict(input_df)[0]
            pred_tree = tree_model.predict(input_df)[0]

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="Prediksi Regresi Linear", value=f"₹ {pred_linear:,.2f}")
            with col_res2:
                st.metric(label="Prediksi Decision Tree", value=f"₹ {pred_tree:,.2f}")

            st.markdown("---")
            with st.expander("Lihat Data Input yang Diproses"):
                st.write("Data input yang digunakan untuk prediksi (setelah encoding):")
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.error("Pastikan model dan daftar kolom dimuat dengan benar, dan input sesuai.")

    else:
        st.warning("Model atau file kolom belum dimuat. Tidak dapat melakukan prediksi.")

st.markdown("""
<hr>
<p style='text-align: center; font-size: small;'>
    Anggota Kelompok :<br>
    Zaky Al Fatih Nata Imam - 1301223172<br>
    I Dewa Putu Rangga Putra Dharma - 1301220427<br>
    Zaidan Rasyid - 1301223134
</p>
""", unsafe_allow_html=True)