# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== Sidebar ==================
menu = st.sidebar.selectbox("Navigasi Dashboard", [
    "📊 Analisis Konsumsi Energi",
    "🤖 Evaluasi Model Prediktif",
    "📜 Riwayat Penggunaan Form",
    "📂 Ringkasan Konsumsi & Formulir Prediksi"
])

# ================== Load Dataset ==================
@st.cache_data
def load_data():
    df = pd.read_csv("data_listrik.csv")
    df = df.dropna()
    df['Estimated_Bill'] = df['Global_active_power'] * 1450 * 0.001
    df['Waktu'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    return df

df = load_data()

# ================== Halaman 1: Eksplorasi Data ==================
if menu == "📊 Analisis Konsumsi Energi":
    st.title("📊 Analisis Konsumsi Energi Rumah Tangga")
    st.markdown("""
    Halaman ini menyajikan visualisasi dan statistik dari data penggunaan listrik rumah tangga.
    """)

    st.subheader("📌 Statistik Ringkasan")
    st.write(df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Estimated_Bill']].describe())

    st.subheader("🔍 Distribusi Konsumsi Energi")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Global_active_power'], bins=50, kde=True, ax=ax1, color='darkcyan')
    ax1.set_xlabel("Daya Aktif Global (kW)")
    st.pyplot(fig1)

    st.subheader("🏠 Proporsi Penggunaan Peralatan")
    sub_avg = df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    fig2, ax2 = plt.subplots()
    ax2.pie(sub_avg, labels=sub_avg.index, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.subheader("🔗 Korelasi Antar Variabel")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    st.pyplot(fig3)

# ================== Halaman 2: Prediksi ==================
elif menu == "🤖 Evaluasi Model Prediktif":
    st.title("🤖 Evaluasi Model Prediktif")
    st.markdown("""
    Halaman ini menampilkan performa berbagai model Machine Learning untuk prediksi tagihan listrik.
    """)

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
        st.write("📉 MAE:", mean_absolute_error(y_test, y_pred))
        st.write("📉 RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
        st.write("📈 Skor R²:", r2_score(y_test, y_pred))
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_xlabel("Nilai Aktual")
        ax.set_ylabel("Nilai Prediksi")
        ax.set_title(f"Prediksi vs Aktual - {name}")
        st.pyplot(fig)

    st.subheader("📈 Grafik Prediksi Tagihan Seiring Waktu")
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    df_sorted = df.sort_values(by='Waktu')
    df_sorted['Prediksi'] = model.predict(df_sorted[X.columns])

    fig_plotly = px.line(df_sorted, x='Waktu', y=['Estimated_Bill', 'Prediksi'],
                         labels={'value': 'Tagihan Listrik (Rp)', 'Waktu': 'Tanggal'},
                         title='Prediksi vs Aktual Estimasi Tagihan Listrik Seiring Waktu')
    st.plotly_chart(fig_plotly, use_container_width=True)

# ================== Halaman 3: Riwayat Input ==================
elif menu == "📜 Riwayat Penggunaan Form":
    st.title("📜 Riwayat Penggunaan Form Prediksi")
    st.markdown("""
    Daftar input prediksi yang telah dilakukan selama sesi penggunaan dashboard.
    """)
    if "riwayat" not in st.session_state:
        st.session_state.riwayat = []
    st.write(pd.DataFrame(st.session_state.riwayat))

# ================== Halaman 4: Ringkasan & Prediksi ==================
elif menu == "📂 Ringkasan Konsumsi & Formulir Prediksi":
    st.title("📂 Ringkasan Konsumsi & Formulir Prediksi")
    st.markdown("""
    Lihat data konsumsi terbaru dan gunakan formulir untuk memperkirakan tagihan listrik berdasarkan input.
    """)

    st.dataframe(df.head(100))

    st.subheader("📝 Formulir Input Prediksi")
    nama = st.text_input("Masukkan Nama Anda")

    if nama:
        st.success(f"Hai, {nama}! Selamat datang di halaman prediksi 👋")

    col1, col2 = st.columns(2)

    with col1:
        gap = st.slider("Daya Aktif Global (kW)", 0.0, 10.0, 1.0, 0.1)
        sm1 = st.slider("Meter Sub 1 (Dapur/Listrik Kecil)", 0.0, 30.0, 0.0, 1.0)
    with col2:
        sm2 = st.slider("Meter Sub 2 (Listrik Rumah Tangga)", 0.0, 30.0, 0.0, 1.0)
        sm3 = st.slider("Meter Sub 3 (Pemanas Air/Listrik Berat)", 0.0, 30.0, 0.0, 1.0)

    if st.button("Prediksi Tagihan Listrik"):
        if not nama:
            st.warning("Silakan isi nama terlebih dahulu.")
        else:
            model = GradientBoostingRegressor()
            X = df[['Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
            y = df['Estimated_Bill']
            model.fit(X, y)
            hasil = model.predict([[gap, sm1, sm2, sm3]])[0]

            st.markdown(f"""
            <div style='padding: 1rem; background-color: #d0f0c0; border-left: 5px solid green;'>
                <h4>💡 Estimasi Tagihan Listrik:</h4>
                <h2>Rp {hasil:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.riwayat.append({
                "Nama": nama,
                "Global_active_power": gap,
                "Sub_metering_1": sm1,
                "Sub_metering_2": sm2,
                "Sub_metering_3": sm3,
                "Estimasi_Tagihan": hasil
            })
