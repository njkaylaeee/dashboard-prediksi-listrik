# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== Sidebar Navigasi ==================
st.sidebar.image("assets/listrik_logo.png", width=150)
st.sidebar.markdown("### ⚡ Prediksi Tagihan Listrik")
st.sidebar.caption("Proyek Data Mining FMIPA UNNES")

menu = st.sidebar.radio("Navigasi", [
    "🏠 Halaman Awal", 
    "📈 Evaluasi Model", 
    "🧾 Formulir Prediksi"
])

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_listrik.csv")
    df = df.dropna()
    df['Estimated_Bill'] = df['Global_active_power'] * 1450 * 0.001
    df['Waktu'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    return df

df = load_data()

# ================== Halaman Awal ==================
if menu == "🏠 Halaman Awal":
    st.title("🏠 Analisis Data Konsumsi Listrik")
    st.markdown("""
    Dataset ini berisi informasi konsumsi energi listrik rumah tangga.
    Anda dapat memilih variabel untuk melihat distribusi serta korelasi antar variabel.
    """)

    kategori = st.selectbox("Pilih Variabel untuk Distribusi", options=['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])

    fig1 = px.histogram(df, x=kategori, color_discrete_sequence=['indianred'], nbins=40, title=f'Distribusi {kategori}')
    st.plotly_chart(fig1, use_container_width=True)

    # Pie Chart Kategori Konsumsi
    bins = [0, 1.5, 3.5, 10]
    labels = ['Hemat', 'Normal', 'Tinggi']
    df['Kategori'] = pd.cut(df['Global_active_power'], bins=bins, labels=labels)
    st.subheader("📊 Proporsi Kategori Konsumsi")
    fig4 = px.pie(df, names='Kategori', title='Distribusi Konsumsi Listrik Rumah Tangga', color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("📌 Penjelasan Kolom"):
        st.markdown("""
        - **Global_active_power**: Konsumsi energi total dalam kilowatt (kW)
        - **Sub_metering_1**: Penggunaan dapur/listrik kecil
        - **Sub_metering_2**: Penggunaan peralatan rumah tangga
        - **Sub_metering_3**: Penggunaan listrik besar/pemanas
        - **Estimated_Bill**: Estimasi tagihan listrik berdasarkan konsumsi
        """)

    st.subheader("🔗 Korelasi Antar Variabel")
    fig2, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='RdBu', fmt='.2f', ax=ax)
    st.pyplot(fig2)

# ================== Evaluasi Model ==================
elif menu == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Model Prediktif")

    X = df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
    y = df['Estimated_Bill']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Regresi Linier": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.subheader(f"Model: {name}")
        st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5, color='mediumblue')
        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Prediksi")
        ax.set_title(f"Prediksi vs Aktual - {name}")
        st.pyplot(fig)

    st.subheader("📊 Tren Prediksi vs Aktual Seiring Waktu")
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    df_sorted = df.sort_values(by='Waktu')
    df_sorted['Prediksi'] = model.predict(df_sorted[X.columns])

    fig3 = px.line(df_sorted, x='Waktu', y=['Estimated_Bill', 'Prediksi'],
                   title='Perbandingan Prediksi dan Aktual Tagihan Listrik',
                   labels={'value': 'Tagihan Listrik (Rp)', 'Waktu': 'Tanggal'},
                   template='plotly_white')
    fig3.update_traces(mode='lines+markers', hovertemplate=None)
    fig3.update_layout(hovermode='x unified')
    st.plotly_chart(fig3, use_container_width=True)

# ================== Formulir Prediksi ==================
elif menu == "🧾 Formulir Prediksi":
    st.title("🧾 Formulir Prediksi dan Ringkasan Penggunaan")
    st.markdown("Masukkan nilai-nilai konsumsi untuk mendapatkan estimasi tagihan listrik.")

    nama = st.text_input("Nama Pengguna")
    gap = st.number_input("Daya Aktif Global (kW)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    sm1 = st.number_input("Sub Metering 1 (Dapur)", min_value=0.0, max_value=30.0, step=1.0)
    sm2 = st.number_input("Sub Metering 2 (Rumah Tangga)", min_value=0.0, max_value=30.0, step=1.0)
    sm3 = st.number_input("Sub Metering 3 (Pemanas)", min_value=0.0, max_value=30.0, step=1.0)

    if st.button("🔍 Prediksi"):
        if not nama:
            st.warning("Silakan isi nama pengguna.")
        else:
            model = GradientBoostingRegressor()
            model.fit(df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']], df['Estimated_Bill'])
            hasil = model.predict([[gap, sm1, sm2, sm3]])[0]

            if hasil < 2000:
                rekom = "Sangat Hemat"
            elif hasil < 5000:
                rekom = "Wajar"
            else:
                rekom = "Boros - Cek Penggunaan"

            st.success(f"Hai, {nama}! Estimasi Tagihan: Rp {hasil:,.2f}")
            st.markdown(f"**Rekomendasi**: {rekom}")
            st.markdown(f"**Input**: GAP={gap}, SM1={sm1}, SM2={sm2}, SM3={sm3}")

            if "riwayat" not in st.session_state:
                st.session_state.riwayat = []

            st.session_state.riwayat.append({
                "Nama": nama,
                "Global_active_power": gap,
                "Sub_metering_1": sm1,
                "Sub_metering_2": sm2,
                "Sub_metering_3": sm3,
                "Estimasi_Tagihan": hasil,
                "Rekomendasi": rekom
            })

    st.subheader("📚 Riwayat Prediksi")
    if "riwayat" in st.session_state and st.session_state.riwayat:
        st.dataframe(pd.DataFrame(st.session_state.riwayat))
    else:
        st.info("Belum ada riwayat prediksi.")
