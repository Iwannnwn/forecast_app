import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ==========================================
# 🚀 LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_lstm_model_businessday.h5", compile=False)
    return model

@st.cache_resource
def load_scalers():
    """Load pre-trained scalers. If not exist, create dummy scalers."""
    try:
        with open("feature_scaler.pkl", "rb") as f:
            feature_scaler = pickle.load(f)
        with open("target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)
        return feature_scaler, target_scaler
    except FileNotFoundError:
        st.warning("⚠️ Scaler files tidak ditemukan. Menggunakan default scaler.")
        # Create default scalers (perlu disesuaikan dengan training data)
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        return feature_scaler, target_scaler

model = load_model()
feature_scaler, target_scaler = load_scalers()

# ==========================================
# ⚙️ CONFIG
# ==========================================
SEQ_LENGTH = 30  # panjang urutan waktu (window) - SESUAI TRAINING MODEL
st.set_page_config(page_title="Prediksi Mixtro", page_icon="📊", layout="wide")
st.title("📊 Prediksi Permintaan Produk Mixtro (PT Petrokimia Gresik)")
st.write("Model LSTM dengan 28 fitur waktu, lag, rolling, dan outlier pattern.")

# ==========================================
# 📂 UPLOAD DATA
# ==========================================
uploaded_file = st.file_uploader(
    "📥 Upload data historis (CSV dengan kolom: tanggal, kuantitas, business_day)",
    type=["csv"]
)

if uploaded_file is not None:
    # ==========================================
    # 🔹 Membaca dan menyiapkan data awal
    # ==========================================
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]

    if not {"tanggal", "kuantitas", "business_day"}.issubset(df.columns):
        st.error("❌ Kolom wajib: tanggal, kuantitas, business_day")
        st.stop()

    # Parsing tanggal sesuai format dd/mm/yyyy
    df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    df["Kuantitas_capped"] = df["kuantitas"]

    st.write(f"📅 Jumlah data mentah terbaca: **{len(df)} baris**")

    # ==========================================
    # 🔧 FEATURE ENGINEERING (selaras dengan training)
    # ==========================================
    st.info("🔧 Melakukan feature engineering otomatis sesuai model training...")

    # Fitur waktu dasar
    df["day_of_week"] = df["tanggal"].dt.dayofweek
    df["month"] = df["tanggal"].dt.month
    df["quarter"] = df["tanggal"].dt.quarter
    df["is_month_end"] = (df["tanggal"].dt.day > 25).astype(int)
    df["is_quarter_end"] = df["tanggal"].dt.month.isin([3, 6, 9, 12]).astype(int)

    # Log transform kuantitas
    df["Kuantitas_log"] = np.log1p(df["kuantitas"])

    # Outlier detection sederhana
    q1, q3 = df["kuantitas"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_outlier"] = ((df["kuantitas"] < lower) | (df["kuantitas"] > upper)).astype(int)
    df["outlier_magnitude"] = np.where(df["is_outlier"] == 1, abs(df["kuantitas"] - df["kuantitas"].median()), 0)
    df["outlier_rolling_count"] = df["is_outlier"].rolling(window=30, min_periods=1).sum()

    # Cyclical encoding
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["kuantitas"].shift(lag)

    # Rolling statistics
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df["kuantitas"].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df["kuantitas"].rolling(window=window, min_periods=1).std()
        df[f"rolling_min_{window}"] = df["kuantitas"].rolling(window=window, min_periods=1).min()
        df[f"rolling_max_{window}"] = df["kuantitas"].rolling(window=window, min_periods=1).max()

    # Difference features
    df["diff_1"] = df["kuantitas"].diff()
    df["diff_7"] = df["kuantitas"].diff(7)

    # Cleaning NaN & Inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)

    st.success(f"✅ Jumlah data setelah feature engineering: {len(df)} baris (target minimal {SEQ_LENGTH})")

    # ==========================================
    # 📊 Kolom fitur numerik sesuai model training
    # ==========================================
    feature_cols = [
        'Kuantitas_log', 'is_outlier', 'outlier_magnitude', 'outlier_rolling_count',
        'day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end',
        'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
        'diff_1', 'diff_7'
    ]

    # ==========================================
    # 📈 PREDIKSI
    # ==========================================
    if len(df) < SEQ_LENGTH:
        st.warning(
            f"⚠️ Data hanya {len(df)} baris setelah feature engineering. "
            f"Butuh minimal {SEQ_LENGTH} hari data historis penuh untuk prediksi stabil."
        )
    else:
        last_seq = df[feature_cols].values[-SEQ_LENGTH:]
        
        # ==========================================
        # 🔧 SCALING INPUT DATA (CRITICAL!)
        # ==========================================
        try:
            last_seq_scaled = feature_scaler.transform(last_seq)
            last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)  # (1, 30, 26)
            
            # ==========================================
            # 📈 PREDIKSI
            # ==========================================
            pred_scaled = model.predict(last_seq_scaled)
            
            # ==========================================
            # 🔄 INVERSE TRANSFORM (CRITICAL!)
            # ==========================================
            pred_value = target_scaler.inverse_transform(pred_scaled)[0, 0]
            
        except Exception as scaling_error:
            st.error(f"❌ Error saat scaling/prediksi: {scaling_error}")
            # Fallback: prediksi tanpa scaling (hasil akan berbeda!)
            st.warning("⚠️ Menggunakan prediksi tanpa scaling (hasil mungkin tidak akurat)")
            last_seq_unscaled = np.expand_dims(last_seq, axis=0)
            pred_value = float(model.predict(last_seq_unscaled).flatten()[0])
        
        next_date = df["tanggal"].max() + timedelta(days=1)

        # ===========================
        # HASIL PREDIKSI
        # ===========================
        st.subheader("📊 Hasil Prediksi")
        st.success(f"Prediksi permintaan untuk {next_date.strftime('%Y-%m-%d')}: **{pred_value:,.2f} unit**")

        # ===========================
        # VISUALISASI
        # ===========================
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["tanggal"], df["kuantitas"], label="Data Historis", marker="o")
        ax.scatter(next_date, pred_value, color="red", label="Prediksi Hari Berikutnya", zorder=5)
        ax.legend()
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Permintaan (unit)")
        ax.set_title("Prediksi Permintaan Produk Mixtro")
        st.pyplot(fig)

        # ===========================
        # DOWNLOAD HASIL
        # ===========================
        result_df = pd.DataFrame({
            "tanggal": [next_date],
            "prediksi_kuantitas": [pred_value]
        })
        st.download_button(
            "⬇️ Download hasil prediksi",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu.")
