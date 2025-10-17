import requests
import csv
import datetime
import time
import os

CITY = "Kyїв"
LAT, LON = 50.45, 30.52  # координати Києва

def get_weather_for_period(start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        f"&timezone=Europe/Kyiv"
    )

    try:
        response = requests.get(url, timeout=30)
    except requests.exceptions.RequestException as e:
        print(f"Помилка підключення: {e}")
        return None

    if response.status_code != 200:
        print(f"Помилка запиту {response.status_code}: {response.text}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"Не вдалося розпарсити JSON для періоду {start_date} — {end_date}")
        return None

    weather_list = []
    for i, date_str in enumerate(data["daily"]["time"]):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        weather_list.append([
            date_obj,
            date_obj.year,
            date_obj.month,
            date_obj.day,
            date_obj.weekday(),
            data["daily"]["temperature_2m_max"][i],
            data["daily"]["temperature_2m_min"][i],
            data["daily"]["precipitation_sum"][i],
            data["daily"]["windspeed_10m_max"][i],
        ])
    return weather_list

def save_to_file(weather_list, filename="data/open_meteo_weather_2015.csv"):
    os.makedirs("data", exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "date", "year", "month", "day", "weekday",
            "temp_max", "temp_min", "precipitation", "wind_max"
        ])
        writer.writerows(weather_list)

if __name__ == "__main__":
    start_year = 2015
    all_weather = []

    current_date = datetime.date(start_year, 1, 1)
    year_end_date = datetime.date(start_year, 12, 31)

    while current_date <= year_end_date:
        period_end_date = current_date + datetime.timedelta(days=14)
        if period_end_date > year_end_date:
            period_end_date = year_end_date

        start_str = current_date.strftime("%Y-%m-%d")
        end_str = period_end_date.strftime("%Y-%m-%d")

        print(f"Завантаження даних: {start_str} — {end_str}")
        data_chunk = get_weather_for_period(start_str, end_str)

        if data_chunk:
            all_weather.extend(data_chunk)
        else:
            print(f"Пропускаємо період {start_str} — {end_str}")

        current_date = period_end_date + datetime.timedelta(days=1)
        time.sleep(1)  # невелика пауза

    if all_weather:
        save_to_file(all_weather)
        print(f"Погода за рік записана у data/open_meteo_weather.csv")
    else:
        print("Дані не були зібрані.")
