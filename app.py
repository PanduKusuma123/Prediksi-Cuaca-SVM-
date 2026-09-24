import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Cuaca SVM", page_icon="🌤️", layout="centered")
st.title("🌤️ Aplikasi Prediksi Cuaca (SVM)")
st.write("Masukkan parameter cuaca pada panel di bawah untuk memprediksi tipe cuaca.")

@st.cache_resource
def load_components():
    try:
        model = joblib.load('svm_weather_model.pkl')
        scaler = joblib.load('scaler_weather.pkl')
        
        expected_columns = [
            'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 
            'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 
            'Cloud Cover_cloudy', 'Cloud Cover_overcast', 'Cloud Cover_partly cloudy', 
            'Season_Spring', 'Season_Summer', 'Season_Winter', 
            'Location_inland', 'Location_mountain'
        ]
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("File model atau scaler tidak ditemukan. Pastikan file .pkl berada di folder yang sama.")
        return None, None, None

model, scaler, expected_columns = load_components()

if model is not None:
    st.subheader("Parameter Meteorologi")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input("Suhu (°C)", value=25.0)
        humidity = st.number_input("Kelembapan (%)", value=60.0)
        wind_speed = st.number_input("Kecepatan Angin (km/h)", value=10.0)
        precipitation = st.number_input("Curah Hujan (%)", value=20.0)
    
    with col2:
        pressure = st.number_input("Tekanan Atmosfer (hPa)", value=1010.0)
        uv_index = st.number_input("Indeks UV", value=5.0)
        visibility = st.number_input("Jarak Pandang (km)", value=10.0)

    st.subheader("Kondisi Lingkungan")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        cloud_cover = st.selectbox("Tutupan Awan", ['clear', 'cloudy', 'overcast', 'partly cloudy'])
    with col4:
        season = st.selectbox("Musim", ['Autumn', 'Spring', 'Summer', 'Winter'])
    with col5:
        location = st.selectbox("Lokasi", ['coastal', 'inland', 'mountain'])

    # Tombol Prediksi
    if st.button("Prediksi Cuaca", type="primary"):

        input_data = pd.DataFrame([[
            temperature, humidity, wind_speed, precipitation, 
            pressure, uv_index, visibility, cloud_cover, season, location
        ]], columns=[
            'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 
            'Atmospheric Pressure', 'UV Index', 'Visibility (km)',
            'Cloud Cover', 'Season', 'Location'
        ])

        input_encoded = pd.get_dummies(input_data, columns=['Cloud Cover', 'Season', 'Location'])

        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

        input_scaled = scaler.transform(input_encoded)

        prediction = model.predict(input_scaled)
        
        label_map = {0: 'Cloudy', 1: 'Rainy', 2: 'Snowy', 3: 'Sunny'} 
        hasil_cuaca = label_map.get(prediction[0], "Tidak Diketahui")

        st.success(f"### Hasil Prediksi: **{hasil_cuaca}**")