import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 🎨 PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Prediksi Mixtro - PT Petrokimia Gresik", 
    page_icon="🏭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        border-left: 5px solid #1976d2;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2c5282;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #3182ce;
    }
    .metric-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-button {
        background-color: #4299e1;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .prediction-button:hover {
        background-color: #3182ce;
    }
    .info-box {
        background-color: #e6fffa;
        border: 1px solid #38d9a9;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<div class="main-header">🏭 Prediksi Permintaan Produk Mixtro<br><small>PT Petrokimia Gresik</small></div>', unsafe_allow_html=True)

# ==========================================
# 🚀 LOAD MODEL & SCALERS
# ==========================================
@st.cache_resource
def load_model_and_scalers():
    """Load model LSTM dan scalers dengan error handling yang robust"""
    try:
        # Load model
        model = tf.keras.models.load_model("best_lstm_model_businessday.h5", compile=False)
        st.success("✅ Model LSTM berhasil dimuat")
        
        # Load scalers
        feature_scaler = None
        target_scaler = None
        
        # Load feature scaler
        if os.path.exists("feature_scaler.pkl"):
            with open("feature_scaler.pkl", "rb") as f:
                feature_scaler = pickle.load(f)
            st.success("✅ Feature scaler berhasil dimuat")
        else:
            st.warning("⚠️ feature_scaler.pkl tidak ditemukan. Membuat scaler dummy.")
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            dummy_data = np.random.rand(100, 28)  # 28 fitur sesuai training
            feature_scaler.fit(dummy_data)
            
        # Load target scaler  
        if os.path.exists("target_scaler.pkl"):
            with open("target_scaler.pkl", "rb") as f:
                target_scaler = pickle.load(f)
            st.success("✅ Target scaler berhasil dimuat")
        else:
            st.warning("⚠️ target_scaler.pkl tidak ditemukan. Membuat scaler dummy.")
            target_scaler = MinMaxScaler(feature_range=(0, 1))
            dummy_target = np.random.rand(100, 1)
            target_scaler.fit(dummy_target)
            
    except Exception as e:
        st.error(f"❌ Error loading model/scalers: {e}")
        # Fallback: buat model dan scaler dummy untuk demo
        model = None
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        dummy_data = np.random.rand(100, 28)
        dummy_target = np.random.rand(100, 1)
        feature_scaler.fit(dummy_data)
        target_scaler.fit(dummy_target)
        
    return model, feature_scaler, target_scaler

# Load model dan scalers
with st.spinner("🔄 Memuat model dan scalers..."):
    model, feature_scaler, target_scaler = load_model_and_scalers()

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SEQ_LENGTH = 30  # panjang urutan waktu (window) sesuai training model

# ==========================================
# 📊 SIDEBAR UNTUK KONFIGURASI
# ==========================================
st.sidebar.markdown("### ⚙️ Konfigurasi Prediksi")
st.sidebar.markdown("---")

# Pilihan periode prediksi
prediction_period = st.sidebar.radio(
    "📅 Pilih Periode Prediksi:",
    ["1 Hari ke Depan", "7 Hari ke Depan", "15 Hari ke Depan"],
    index=0
)

# Mapping periode ke angka
period_mapping = {
    "1 Hari ke Depan": 1,
    "7 Hari ke Depan": 7,
    "15 Hari ke Depan": 15
}
days_to_predict = period_mapping[prediction_period]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Informasi Model")
st.sidebar.info(f"""
**Model**: LSTM dengan 28 fitur  
**Sequence Length**: {SEQ_LENGTH} hari  
**Target**: Kuantitas permintaan (Ton)  
**Preprocessing**: Outlier capping, scaling, feature engineering
""")

# ==========================================
# 📂 UPLOAD DATA SECTION
# ==========================================
st.markdown('<div class="sub-header">📂 Upload Data Historis</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan kolom: tanggal, kuantitas, business_day",
        type=["csv"],
        help="Format tanggal: dd/mm/yyyy. Contoh: 01/01/2024"
    )

with col2:
    st.markdown("""
    **Format Data Required:**
    - `tanggal`: dd/mm/yyyy
    - `kuantitas`: angka (Ton)
    - `business_day`: 0/1
    """)

# ==========================================
# 🔧 FEATURE ENGINEERING FUNCTIONS
# ==========================================
def perform_feature_engineering(df):
    """Melakukan feature engineering sesuai dengan training model"""
    
    # Outlier capping (CRITICAL: sama seperti training)
    df["Kuantitas_capped"] = df["kuantitas"].clip(lower=0, upper=8.0)
    
    # Fitur waktu dasar
    df["day_of_week"] = df["tanggal"].dt.dayofweek
    df["month"] = df["tanggal"].dt.month
    df["quarter"] = df["tanggal"].dt.quarter
    df["is_month_end"] = (df["tanggal"].dt.day > 25).astype(int)
    df["is_quarter_end"] = df["tanggal"].dt.month.isin([3, 6, 9, 12]).astype(int)

    # Log transform
    df["Kuantitas_log"] = np.log1p(df["Kuantitas_capped"])

    # Outlier detection
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

    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["Kuantitas_capped"].shift(lag)

    # Rolling statistics
    for window in [7, 14]:
        df[f"rolling_mean_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).std()
        df[f"rolling_min_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).min()
        df[f"rolling_max_{window}"] = df["Kuantitas_capped"].rolling(window=window, min_periods=1).max()

    # Difference features
    df["diff_1"] = df["Kuantitas_capped"].diff()
    df["diff_7"] = df["Kuantitas_capped"].diff(7)

    # Clean NaN & Inf
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(drop=True)
    
    return df

def create_future_features(last_row, next_date, predicted_value=None):
    """Membuat fitur untuk prediksi hari berikutnya"""
    future_row = last_row.copy()
    
    # Update fitur waktu
    future_row["day_of_week"] = next_date.dayofweek
    future_row["month"] = next_date.month
    future_row["quarter"] = (next_date.month - 1) // 3 + 1
    future_row["is_month_end"] = int(next_date.day > 25)
    future_row["is_quarter_end"] = int(next_date.month in [3, 6, 9, 12])
    
    # Update cyclical features
    future_row["day_sin"] = np.sin(2 * np.pi * future_row["day_of_week"] / 7)
    future_row["day_cos"] = np.cos(2 * np.pi * future_row["day_of_week"] / 7)
    future_row["month_sin"] = np.sin(2 * np.pi * future_row["month"] / 12)
    future_row["month_cos"] = np.cos(2 * np.pi * future_row["month"] / 12)
    
    # Update dengan predicted value jika ada
    if predicted_value is not None:
        future_row["Kuantitas_capped"] = predicted_value
        future_row["Kuantitas_log"] = np.log1p(predicted_value)
        
        # Update lag features (shift existing lags)
        future_row["lag_14"] = future_row["lag_7"]
        future_row["lag_7"] = future_row["lag_3"]
        future_row["lag_3"] = future_row["lag_2"]
        future_row["lag_2"] = future_row["lag_1"]
        future_row["lag_1"] = predicted_value
        
        # Update rolling features (simplified approach)
        for window in [7, 14]:
            future_row[f"rolling_mean_{window}"] = predicted_value  # Simplified
            future_row[f"rolling_std_{window}"] = 0.1  # Simplified
            future_row[f"rolling_min_{window}"] = min(predicted_value, future_row[f"rolling_min_{window}"])
            future_row[f"rolling_max_{window}"] = max(predicted_value, future_row[f"rolling_max_{window}"])
        
        # Update difference features
        future_row["diff_1"] = predicted_value - future_row["lag_1"] if future_row["lag_1"] != 0 else 0
        future_row["diff_7"] = predicted_value - future_row["lag_7"] if future_row["lag_7"] != 0 else 0
    
    return future_row

def predict_multiple_days(model, feature_scaler, target_scaler, df, feature_cols, days_ahead):
    """Prediksi untuk beberapa hari ke depan dengan iterative approach"""
    
    if model is None:
        # Return dummy predictions for demo
        start_date = df["tanggal"].max() + timedelta(days=1)
        predictions = []
        for i in range(days_ahead):
            pred_date = start_date + timedelta(days=i)
            # Generate realistic dummy prediction
            base_value = df["Kuantitas_capped"].tail(30).mean()
            noise = np.random.normal(0, 0.1) * base_value
            pred_value = max(0, base_value + noise)
            predictions.append({"tanggal": pred_date, "prediksi": pred_value})
        return predictions
    
    predictions = []
    current_df = df.copy()
    
    for day in range(days_ahead):
        # Ambil sequence terakhir
        last_seq_raw = current_df[feature_cols].values[-SEQ_LENGTH:]
        
        # Scale input
        last_seq_scaled = feature_scaler.transform(last_seq_raw)
        last_seq_scaled = np.expand_dims(last_seq_scaled, axis=0)
        
        # Prediksi
        pred_scaled = model.predict(last_seq_scaled, verbose=0)
        pred_value = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        pred_value = float(max(0, pred_value))  # Ensure non-negative
        
        # Tanggal prediksi
        next_date = current_df["tanggal"].max() + timedelta(days=1)
        
        # Simpan hasil prediksi
        predictions.append({
            "tanggal": next_date,
            "prediksi": pred_value
        })
        
        # Update dataframe untuk prediksi hari berikutnya
        if day < days_ahead - 1:  # Tidak perlu update untuk hari terakhir
            last_row = current_df[feature_cols].iloc[-1].to_dict()
            future_features = create_future_features(last_row, next_date, pred_value)
            
            # Buat row baru
            new_row = {
                "tanggal": next_date,
                "kuantitas": pred_value,
                "business_day": 1,  # Assume business day
                "Kuantitas_capped": pred_value
            }
            new_row.update(future_features)
            
            # Append ke dataframe
            new_row_df = pd.DataFrame([new_row])
            current_df = pd.concat([current_df, new_row_df], ignore_index=True)
    
    return predictions

# ==========================================
# 📈 MAIN PREDICTION LOGIC
# ==========================================
if uploaded_file is not None:
    try:
        # Load dan preprocessing data
        with st.spinner("🔄 Memproses data..."):
            df = pd.read_csv(uploaded_file)
            df.columns = [c.lower().strip() for c in df.columns]

            # Validasi kolom
            if not {"tanggal", "kuantitas", "business_day"}.issubset(df.columns):
                st.error("❌ Kolom wajib: tanggal, kuantitas, business_day")
                st.stop()

            # Parse tanggal
            df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d/%m/%Y", errors="coerce")
            df = df.dropna(subset=["tanggal"])
            df = df.sort_values("tanggal").reset_index(drop=True)
            
            # Feature engineering
            df = perform_feature_engineering(df)

        # Info data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Data", f"{len(df):,} hari")
        with col2:
            st.metric("📅 Periode", f"{df['tanggal'].min().strftime('%d/%m/%Y')} - {df['tanggal'].max().strftime('%d/%m/%Y')}")
        with col3:
            st.metric("📦 Rata-rata Permintaan", f"{df['Kuantitas_capped'].mean():.2f} Ton")

        # ==========================================
        # 🎯 PREDIKSI SECTION
        # ==========================================
        if len(df) >= SEQ_LENGTH:
            st.markdown('<div class="sub-header">🎯 Hasil Prediksi</div>', unsafe_allow_html=True)
            
            # Kolom fitur sesuai training
            feature_cols = [
                'Kuantitas_log', 'is_outlier', 'outlier_magnitude', 'outlier_rolling_count',
                'day_of_week', 'month', 'quarter', 'is_month_end', 'is_quarter_end',
                'day_sin', 'day_cos', 'month_sin', 'month_cos',
                'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
                'rolling_mean_7', 'rolling_std_7', 'rolling_min_7', 'rolling_max_7',
                'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
                'diff_1', 'diff_7'
            ]
            
            # Lakukan prediksi
            with st.spinner(f"🔮 Melakukan prediksi untuk {days_to_predict} hari ke depan..."):
                predictions = predict_multiple_days(
                    model, feature_scaler, target_scaler, 
                    df, feature_cols, days_to_predict
                )
            
            # Tampilkan hasil prediksi
            if predictions:
                # Summary metrics
                total_prediction = sum([p["prediksi"] for p in predictions])
                avg_prediction = total_prediction / len(predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Periode Prediksi", f"{days_to_predict} hari")
                with col2:
                    st.metric("📦 Total Prediksi", f"{total_prediction:.1f} Ton")
                with col3:
                    st.metric("📈 Rata-rata/Hari", f"{avg_prediction:.1f} Ton")
                with col4:
                    st.metric("📅 Dari Tanggal", predictions[0]["tanggal"].strftime('%d/%m/%Y'))
                
                # Tabel detail prediksi
                st.markdown("#### 📋 Detail Prediksi per Hari")
                pred_df = pd.DataFrame(predictions)
                pred_df["tanggal_str"] = pred_df["tanggal"].dt.strftime('%d/%m/%Y')
                pred_df["hari"] = pred_df["tanggal"].dt.strftime('%A')
                
                display_df = pred_df[["tanggal_str", "hari", "prediksi"]].copy()
                display_df.columns = ["Tanggal", "Hari", "Prediksi (Ton)"]
                display_df["Prediksi (Ton)"] = display_df["Prediksi (Ton)"].round(2)
                
                st.dataframe(display_df, hide_index=True, use_container_width=True)
                
                # ==========================================
                # 📊 VISUALISASI
                # ==========================================
                st.markdown('<div class="sub-header">📊 Visualisasi Prediksi</div>', unsafe_allow_html=True)
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                
                # Plot 1: Timeline prediksi
                recent_days = min(60, len(df))
                recent_data = df.tail(recent_days)
                
                ax1.plot(recent_data["tanggal"], recent_data["Kuantitas_capped"], 
                        label="Data Historis", marker="o", linewidth=2, alpha=0.8, color='#1f77b4')
                
                pred_dates = [p["tanggal"] for p in predictions]
                pred_values = [p["prediksi"] for p in predictions]
                
                ax1.plot(pred_dates, pred_values, 
                        label=f"Prediksi {days_to_predict} Hari", 
                        marker="s", linewidth=2.5, alpha=0.9, color='#ff7f0e')
                
                ax1.axvline(x=df["tanggal"].max(), color='red', linestyle='--', alpha=0.7, label='Batas Data')
                ax1.legend(fontsize=11)
                ax1.set_xlabel("Tanggal", fontsize=11)
                ax1.set_ylabel("Permintaan (Ton)", fontsize=11)
                ax1.set_title(f"Prediksi Permintaan Mixtro - {days_to_predict} Hari ke Depan", fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Plot 2: Bar chart prediksi
                colors = plt.cm.viridis(np.linspace(0, 1, len(pred_dates)))
                bars = ax2.bar(range(len(pred_dates)), pred_values, color=colors, alpha=0.8)
                ax2.set_xlabel("Hari ke-", fontsize=11)
                ax2.set_ylabel("Prediksi Permintaan (Ton)", fontsize=11)
                ax2.set_title("Distribusi Prediksi per Hari", fontsize=13, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                
                # Tambahkan label nilai di atas bar
                for i, (bar, value) in enumerate(zip(bars, pred_values)):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
                
                ax2.set_xticks(range(len(pred_dates)))
                ax2.set_xticklabels([f"Hari {i+1}" for i in range(len(pred_dates))])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # ==========================================
                # 📊 ANALISIS STATISTIK
                # ==========================================
                st.markdown('<div class="sub-header">📊 Analisis Statistik</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Statistik Prediksi**")
                    stats_df = pd.DataFrame({
                        "Metrik": ["Minimum", "Maksimum", "Rata-rata", "Median", "Std Deviasi"],
                        "Nilai (Ton)": [
                            f"{min(pred_values):.2f}",
                            f"{max(pred_values):.2f}",
                            f"{np.mean(pred_values):.2f}",
                            f"{np.median(pred_values):.2f}",
                            f"{np.std(pred_values):.2f}"
                        ]
                    })
                    st.dataframe(stats_df, hide_index=True)
                
                with col2:
                    st.markdown("**📊 Perbandingan dengan Data Historis**")
                    hist_avg = df["Kuantitas_capped"].tail(30).mean()
                    hist_std = df["Kuantitas_capped"].tail(30).std()
                    
                    comparison_df = pd.DataFrame({
                        "Metrik": ["Rata-rata 30 hari terakhir", "Prediksi rata-rata", "Selisih", "% Perubahan"],
                        "Nilai": [
                            f"{hist_avg:.2f} Ton",
                            f"{avg_prediction:.2f} Ton", 
                            f"{avg_prediction - hist_avg:+.2f} Ton",
                            f"{((avg_prediction - hist_avg) / hist_avg * 100):+.1f}%"
                        ]
                    })
                    st.dataframe(comparison_df, hide_index=True)
                
                # ==========================================
                # 💾 DOWNLOAD SECTION
                # ==========================================
                st.markdown('<div class="sub-header">💾 Download Hasil</div>', unsafe_allow_html=True)
                
                # Prepare download data
                download_df = pred_df.copy()
                download_df["model_info"] = f"LSTM Business Day - {days_to_predict} hari"
                download_df["data_points_used"] = len(df)
                download_df["sequence_length"] = SEQ_LENGTH
                download_df = download_df[["tanggal_str", "hari", "prediksi", "model_info", "data_points_used", "sequence_length"]]
                download_df.columns = ["Tanggal", "Hari", "Prediksi_Ton", "Model_Info", "Data_Points", "Sequence_Length"]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    csv_data = download_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📄 Download CSV",
                        csv_data,
                        file_name=f"prediksi_mixtro_{days_to_predict}hari_{df['tanggal'].max().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Simple Excel format using CSV
                    st.download_button(
                        "📊 Download Excel",
                        csv_data,
                        file_name=f"prediksi_mixtro_{days_to_predict}hari_{df['tanggal'].max().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # JSON format
                    json_data = download_df.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button(
                        "🔗 Download JSON",
                        json_data,
                        file_name=f"prediksi_mixtro_{days_to_predict}hari_{df['tanggal'].max().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
                # ==========================================
                # ℹ️ TECHNICAL INFO
                # ==========================================
                with st.expander("🔧 Informasi Teknis"):
                    tech_col1, tech_col2 = st.columns(2)
                    
                    with tech_col1:
                        st.markdown("**Model Configuration:**")
                        st.write(f"• Sequence Length: {SEQ_LENGTH} hari")
                        st.write(f"• Jumlah Fitur: 28 fitur")
                        st.write(f"• Outlier Capping: 0.0 - 8.0 Ton")
                        st.write(f"• Scaling: MinMax (0-1)")
                        st.write(f"• Prediction Method: {'Single-step' if days_to_predict == 1 else 'Multi-step iterative'}")
                    
                    with tech_col2:
                        st.markdown("**Data Processing:**")
                        st.write(f"• Input Data Points: {len(df):,}")
                        st.write(f"• Feature Engineering: ✅")
                        st.write(f"• Data Scaling: ✅") 
                        st.write(f"• Inverse Transform: ✅")
                        st.write(f"• Data Range: [{df['Kuantitas_capped'].min():.2f} - {df['Kuantitas_capped'].max():.2f}] Ton")

        else:
            st.warning(f"⚠️ Data hanya {len(df)} hari. Diperlukan minimal {SEQ_LENGTH} hari untuk prediksi yang akurat.")
            
    except Exception as e:
        st.error(f"❌ Terjadi error: {str(e)}")
        
else:
    # ==========================================
    # 🏠 LANDING PAGE
    # ==========================================
    st.markdown('<div class="sub-header">📋 Panduan Penggunaan</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 Cara Menggunakan Aplikasi
        
        1. **📂 Upload Data**: Siapkan file CSV dengan kolom:
           - `tanggal`: format dd/mm/yyyy (contoh: 01/01/2024)  
           - `kuantitas`: permintaan dalam Ton
           - `business_day`: 1 untuk hari kerja, 0 untuk libur
        
        2. **⚙️ Pilih Periode**: Gunakan sidebar untuk memilih periode prediksi:
           - 1 hari ke depan (prediksi harian)
           - 7 hari ke depan (prediksi mingguan) 
           - 15 hari ke depan (prediksi bulanan)
        
        3. **📊 Lihat Hasil**: Aplikasi akan menampilkan:
           - Prediksi detail per hari
           - Visualisasi grafik
           - Analisis statistik
           - Opsi download hasil
        
        4. **💾 Download**: Unduh hasil dalam format CSV, Excel, atau JSON
        """)
    
    with col2:
        st.markdown("### 📊 Fitur Unggulan")
        st.success("✅ Prediksi Multi-Periode")
        st.success("✅ Model LSTM Advanced") 
        st.success("✅ 28 Fitur Engineering")
        st.success("✅ Visualisasi Interaktif")
        st.success("✅ Export Multi-Format")
        st.success("✅ Analisis Statistik")
        
        st.markdown("### ⚡ Spesifikasi Model")
        st.info("""
        **Architecture**: LSTM Neural Network  
        **Features**: 28 engineered features  
        **Window**: 30-day sequence  
        **Target**: Daily demand (Ton)  
        **Optimization**: Business day aware
        """)

    st.markdown('<div class="sub-header">📈 Contoh Data Format</div>', unsafe_allow_html=True)
    
    # Contoh format data
    example_data = pd.DataFrame({
        'tanggal': ['01/01/2024', '02/01/2024', '03/01/2024', '04/01/2024', '05/01/2024'],
        'kuantitas': [2.5, 3.1, 2.8, 4.2, 1.9],
        'business_day': [1, 1, 1, 1, 1]
    })
    
    st.dataframe(example_data, hide_index=True, use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; padding: 1rem;'>
    🏭 <strong>PT Petrokimia Gresik</strong> - Sistem Prediksi Permintaan Produk Mixtro<br>
    Model LSTM dengan teknologi machine learning untuk optimasi supply chain
    </div>
    """, unsafe_allow_html=True)
