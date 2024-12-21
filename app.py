import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import requests
import aiohttp
import asyncio
from datetime import date


def get_current_temperature(city: str, api_key: str):
    """
    Получение текущений температуры.
    :param city:
    :param api_key:
    :return:
    """
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric',
        'lang': 'en'
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        return temperature, None
    elif response.status_code == 401:
        return None, "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."
    else:
        return None, f"Error {response.status_code}: Unable to fetch weather data for {city}."

def anomalies_detection(data):
    """
    Выявление в данных аномалий и нормальных температур.
    :param data:
    :return:
    """
    window_size = 30

    data['rolling_mean'] = data.groupby(['city', 'season'])['temperature'] \
        .rolling(window=window_size, min_periods=1) \
        .mean().reset_index(level=[0, 1], drop=True)

    std_stats = data.groupby(['city', 'season'])['temperature'].agg(['std', 'min', 'max', 'median']).reset_index()
    mean_stats = data.groupby(['city', 'season'])['rolling_mean'].agg(['mean']).reset_index()

    data = data.merge(std_stats, on=['city', 'season'], how='left', suffixes=('', '_drop'))[
        ['city', 'timestamp', 'temperature', 'season', 'rolling_mean', 'std', 'min', 'max', 'median']]

    data = data.merge(mean_stats, on=['city', 'season'], how='left', suffixes=('', '_drop'))[
        ['city', 'timestamp', 'temperature', 'season', 'rolling_mean', 'mean', 'std', 'min', 'max', 'median']]

    data['is_anomaly'] = (data['temperature'] > (data['rolling_mean'] + 2 * data['std'])) | \
                         (data['temperature'] < (data['rolling_mean'] - 2 * data['std']))

    return data

def plot_temperature_trends(data, city):
    """
    Отображение статистики по историческим данным.
    :param data:
    :param city:
    :return:
    """
    city_data = data[data['city'] == city].copy()

    city_data = anomalies_detection(city_data)

    city_data['days'] = (city_data['timestamp'] - city_data['timestamp'].min()).dt.days

    model = LinearRegression()
    model.fit(city_data[['days']], city_data['temperature'])

    # сохраняем коэффициент тренда
    trend_coef = round(model.coef_[0], 6)

    # предсказание долгосрочного тренда
    city_data['trend'] = model.predict(city_data[['days']])

    plt.figure(figsize=(15, 6))

    plt.plot(city_data['timestamp'], city_data['temperature'], label='Temperature', alpha=0.6)
    plt.plot(city_data['timestamp'], city_data['trend'], label=f'Trend (coefficient = {trend_coef})', color='green', linewidth=2)

    anomaly_points = city_data[city_data['is_anomaly']]
    plt.scatter(
        anomaly_points['timestamp'], anomaly_points['temperature'],
        color='red', label='Anomalies', alpha=0.7, s=1.6
    )

    if trend_coef < 0:
        title_line = f"Temperature dynamics and negative trend for {city}"
    elif trend_coef > 0:
        title_line = f"Temperature dynamics and positive trend for {city}"
    else:
        title_line = f"Temperature dynamics and neutral trend for {city}"

    plt.title(title_line)
    plt.xlabel("Date")
    plt.ylabel("Temperature, °C")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()
    st.pyplot(plt)


def plot_seasonal_boxplot(data, city):
    """
    Построение боксплота для отображения температур по сезонам.
    :param city_data: DataFrame с данными для одного города
    """

    city_data = data[data['city'] == city].copy()

    city_data = anomalies_detection(city_data)

    plt.figure(figsize=(10, 6))

    sns.boxplot(x='season', y='temperature', data=city_data, palette='Set2', hue='season')

    for season in city_data['season'].unique():
        season_data = city_data[city_data['season'] == season]
        mean = season_data['mean'].iloc[0]
        std = season_data['std'].iloc[0]

        # отображение среднего значения с ошибками
        plt.errorbar(
            x=season, y=mean, yerr=std, fmt='o', color='blue', label='Mean ± Std' if season == city_data['season'].unique()[0] else ""
        )

    plt.title(f"Seasonal Temperature Distribution for {city_data['city'].iloc[0]}")
    plt.xlabel("Season")
    plt.ylabel("Temperature, °C")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.show()
    st.pyplot(plt)


# загрузка файла с данными
st.title("Weather Data Analysis")
uploaded_file = st.file_uploader("Upload your historical weather data", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # выбор города
    city = st.selectbox("Select a city", options=data['city'].unique())

    # ввод API-ключа
    api_key = st.text_input("Enter your OpenWeatherMap API key")

    city_data = data[data['city'] == city].reset_index(drop=True)
    city_data = anomalies_detection(city_data)
    city_data = city_data.reset_index(drop=True)

    season_order_dict = {'winter': 1, 'spring': 2, 'summer': 3, 'autumn': 4}
    city_data['№'] = city_data['season'].map(season_order_dict)

    # получение текущей температуры
    if api_key:
        temperature, error = get_current_temperature(city, api_key)

        if temperature is not None:
            city_data['month'] = city_data['timestamp'].dt.month
            current_date = date.today().month

            current_season = city_data[city_data['month'] == current_date]['season'].iloc[0]
            mean_temp = city_data[city_data['season'] == current_season]['mean'].iloc[0]
            std_temp = city_data[city_data['season'] == current_season]['std'].iloc[0]

            if (temperature > (mean_temp + 2 * std_temp)) or (temperature < (mean_temp - 2 * std_temp)):
                st.info(f"The current temperature in {city} is **{temperature}°C**. It's **anomal** for the {current_season} season.")
            else:
                st.info(f"The current temperature in {city} is **{temperature}°C**. It's **normal** for the {current_season} season.")
        else:
            st.error(error)

    # блок с выводом статистики
    st.write(f"Descriptive statistics for {city}:")

    grouped = city_data.groupby(['№', 'season'])
    st.write(grouped['temperature'].describe())

    # визуализация временного ряда температур
    plot_temperature_trends(data, city)

    st.write(city_data.sort_values(by='№')[['season', 'mean', 'std', 'median', 'min', 'max']].drop_duplicates(keep='first').reset_index(drop=True))

    # демонстрация сезонных профилей
    plot_seasonal_boxplot(data, city)
else:
    st.write("Please upload a CSV file to proceed.")