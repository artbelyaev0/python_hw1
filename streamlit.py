import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Weather Data Analysis")
uploaded_file = st.file_uploader("Choose a CSV file with historical data", type="csv")

city = st.selectbox("Select a city", ("New York", "Moscow", "Berlin", "Cairo", "Dubai", "Beijing"))

api_key = st.text_input("Enter OpenWeatherMap API key", type="password")


if uploaded_file is not None:
    historical_data = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully!")

    city_data = historical_data[historical_data['city'] == city]
    st.write(f"Descriptive statistics for {city}")
    st.write(city_data.describe())

    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(city_data['date']), city_data['temperature'], label='Temperature')
    mean_temp = city_data['temperature'].mean()
    std_temp = city_data['temperature'].std()
    anomalies = city_data[np.abs(city_data['temperature'] - mean_temp) > 2 * std_temp]
    ax.scatter(pd.to_datetime(anomalies['date']), anomalies['temperature'], color='red', label='Anomalies')
    ax.set_title(f"Temperature Time Series and Anomalies for {city}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature")
    ax.legend()
    st.pyplot(fig)

    decomposed = seasonal_decompose(city_data['temperature'], model='additive', period=365)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(decomposed.seasonal)
    ax[0].set_title("Seasonal Component")
    ax[1].plot(decomposed.trend)
    ax[1].set_title("Trend Component")
    ax[2].plot(decomposed.resid)
    ax[2].set_title("Residual Component")
    st.pyplot(fig)

if api_key:
    current_weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(current_weather_url)
        weather_data = response.json()

        if weather_data['cod'] == 200:
            current_temp = weather_data['main']['temp']
            st.write(f"Current temperature in {city}: {current_temp}°C")

            month = datetime.now().month
            seasonal_temps = city_data[pd.to_datetime(city_data['date']).dt.month == month]['temperature']
            mean_seasonal_temp = seasonal_temps.mean()
            if np.abs(current_temp - mean_seasonal_temp) > seasonal_temps.std():
                st.write("The current temperature is considered an anomaly for this season.")
            else:
                st.write("The current temperature is normal for this season.")
        else:
            st.error(f"API Error: {weather_data['message']}")

    except Exception as e:
        st.error("Failed to fetch current weather data. Please check your API key and internet connection.")

else:
    st.warning("Please enter a valid OpenWeatherMap API key to see the current weather data.")