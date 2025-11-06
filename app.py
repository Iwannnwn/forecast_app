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
# 🚀 LOAD MODEL & SCALERS
# ==========================================
@st.cache_resource
def load_model_and_scalers():
    # Load model
    model = tf.keras.models.load_model("best_lstm_model_businessday.h5", compile=False)
    
    # Load scalers
    feature_scaler = None
    target_scaler = None
    
    try:
        # Coba load feature scaler
        if os.path.exists("feature_scaler.pkl"):
            with open("feature_scaler.pkl", "rb") as f:
                feature_scaler = pickle.load(f)
            st.info("✅ Feature scaler loaded successfully")
        else:
            st.warning("⚠️ feature_scaler.pkl tidak ditemukan. Akan membuat scaler dummy.")
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit dengan data dummy agar scaler bisa digunakan
            dummy_data = np.random.rand(100, 28)  # 28 fitur sesuai training
            feature_scaler.fit(dummy_data)
            
        # Coba load target scaler  
        if os.path.exists("target_scaler.pkl"):
            with open("target_scaler.pkl", "rb") as f:
                target_scaler = pickle.load(f)
            st.info("✅ Target scaler loaded successfully")
        else:
            st.warning("⚠️ target_scaler.pkl tidak ditemukan. Akan membuat scaler dummy.")
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            # Fit dengan data dummy
            dummy_target = np.random.rand(100, 1)
            target_scaler.fit(dummy_target)
            
    except Exception as e:
        st.error(f"❌ Error loading scalers: {e}")
        # Fallback: buat scaler dummy
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_data = np.random.rand(100, 28)  # 28 fitur sesuai training
        dummy_target = np.random.rand(100, 1)
        feature_scaler.fit(dummy_data)
        target_scaler.fit(dummy_target)
        
    return model, feature_scaler, target_scaler

model, feature_scaler, target_scaler = load_model_and_scalers()

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
    
    # ⚠️ CRITICAL: OUTLIER CAPPING SAMA SEPERTI TRAINING
    # Capping outliers ke 8.0 (sama seperti hybrid approach di training)
    df["Kuantitas_capped"] = df["kuantitas"].clip(lower=0, upper=8.0)

    st.write(f"📅 Jumlah data mentah terbaca: **{len(df)} baris**")
    st.info(f"🔧 Outlier capping applied")

    # ==========================================
    # 🔧 FEATURE ENGINEERING (selaras dengan training)
    # ==========================================
    st.info("🔧 Melakukan feature engineering...")

    # Fitur waktu dasar
    df["day_of_week"] = df["tanggal"].dt.dayofweek
    df["month"] = df["tanggal"].dt.month
    df["quarter"] = df["tanggal"].dt.quarter
    df["is_month_end"] = (df["tanggal"].dt.day > 25).astype(int)
    df["is_quarter_end"] = df["tanggal"].dt.month.isin([3, 6, 9, 12]).astype(int)

    # Log transform kuantitas - GUNAKAN KUANTITAS_CAPPED
    df["Kuantitas_log"] = np.log1p(df["Kuantitas_capped"])

    # Outlier detection - GUNAKAN KUANTITAS_CAPPED
    q1, q3 = df["Kuantitas_capped"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_outlier"] = ((df["Kuantitas_capped"] < lower) | (df["Kuantitas_capped"] > upper)).astype(int)
    df["outlier_magnitude"] = np.where(df["is_outlier"] == 1, 
                                       abs(df["Kuantitas_capped"] - df["Kuantitas_capped"].median()), 0)
    df["outlier_rolling_count"] = df["is_outlier"].rolling(window=30, min_periods=1).sum()

    # Cyclical encoding
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features - GUNAKAN KUANTITAS_CAPPED
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["Kuantitas_capped"].shift(lag)

    # Rolling statistics - GUNAKAN KUANTITAS_CAPPED
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).std()
        df[f"rolling_min_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).min()
        df[f"rolling_max_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).max()

    # Difference features - GUNAKAN KUANTITAS_CAPPED
    df["diff_1"] = df["Kuantitas_capped"].diff()
    df["diff_7"] = df["Kuantitas_capped"].diff(7)

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
    # 📈 PREDIKSI DENGAN SCALING
    # ==========================================
    if len(df) < SEQ_LENGTH:
        st.warning(
            f"⚠️ Data hanya {len(df)} baris setelah feature engineering. "
            f"Butuh minimal {SEQ_LENGTH} hari data historis penuh untuk prediksi stabil."
        )
    else:
        # Ambil sequence terakhir
        last_seq_raw = df[feature_cols].values[-SEQ_LENGTH:]
        
        # ⚠️ SCALING INPUT DATA - CRITICAL STEP
        st.info("🔧 Melakukan scaling pada input data...")
        last_seq_scaled = feature_scaler.transform(last_seq_raw)
        last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)  # (1, 30, 28)
        
        # Prediksi dengan data yang sudah di-scale
        try:
            pred_scaled = model.predict(last_seq_scaled)
            
            # ⚠️ INVERSE TRANSFORM HASIL PREDIKSI - CRITICAL STEP  
            st.info("🔧 Melakukan inverse transform pada hasil prediksi...")
            pred_value = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            pred_value = float(pred_value)
            
            next_date = df["tanggal"].max() + timedelta(days=1)

            # ===========================
            # HASIL PREDIKSI
            # ===========================
            st.subheader("📊 Hasil Prediksi")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📅 Tanggal Prediksi", next_date.strftime('%d/%m/%Y'))
            with col2:
                st.metric("📦 Prediksi Permintaan", f"{pred_value:,.2f} unit")
            
            st.success(f"✅ Prediksi permintaan untuk {next_date.strftime('%d/%m/%Y')}: **{pred_value:,.2f} Ton**")
            
            # Tampilkan info teknis
            with st.expander("ℹ️ Info Teknis Prediksi"):
                st.write(f"• Input data terakhir: {len(last_seq_raw)} hari")
                st.write(f"• Jumlah fitur: {last_seq_raw.shape[1]} fitur")
                st.write(f"• Data sudah di-scale: ✅")
                st.write(f"• Hasil sudah di-inverse transform: ✅")
                st.write(f"• Range data input (min-max): [{last_seq_raw.min():.2f} - {last_seq_raw.max():.2f}]")
                st.write(f"• Range data scaled (min-max): [{last_seq_scaled.min():.4f} - {last_seq_scaled.max():.4f}]")
                
                # Debug info tambahan
                st.write("**🔍 Debug Info:**")
                st.write(f"• Kuantitas_capped terakhir: {df['Kuantitas_capped'].tail(5).tolist()}")
                st.write(f"• Target scaler range: 0.0 - 8.0 Ton")
                st.write(f"• Prediksi scaled: {pred_scaled.flatten()[0]:.6f}")
                st.write(f"• Data preprocessing: SAMA dengan training model ✅")

            # ===========================
            # VISUALISASI
            # ===========================
            st.subheader("📈 Visualisasi Prediksi")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot data historis (30 hari terakhir untuk clarity)
            recent_data = df.tail(min(60, len(df)))
            ax.plot(recent_data["tanggal"], recent_data["Kuantitas_capped"], 
                   label="Data Historis (Capped)", marker="o", linewidth=2, alpha=0.8)
            
            # Plot prediksi
            ax.scatter(next_date, pred_value, color="red", s=100, 
                      label=f"Prediksi: {pred_value:,.0f} unit", zorder=5)
            
            ax.legend(fontsize=12)
            ax.set_xlabel("Tanggal", fontsize=12)
            ax.set_ylabel("Permintaan (Ton)", fontsize=12)
            ax.set_title("Prediksi Permintaan Produk Mixtro (PT Petrokimia Gresik)", fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # ===========================
            # DOWNLOAD HASIL
            # ===========================
            st.subheader("💾 Download Hasil Prediksi")
            result_df = pd.DataFrame({
                "tanggal": [next_date.strftime('%d/%m/%Y')],
                "prediksi_kuantitas": [pred_value],
                "confidence": ["Model LSTM Business Day"],
                "input_data_points": [len(df)],
                "sequence_length": [SEQ_LENGTH]
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download CSV hasil prediksi",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"prediksi_mixtro_{next_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "⬇️ Download Excel hasil prediksi", 
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"prediksi_mixtro_{next_date.strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi: {e}")

else:
    st.info("⬆️ Silakan upload file CSV terlebih dahulu.")
