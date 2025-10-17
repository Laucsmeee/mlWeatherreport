# Weather ML — прогнозування опадів у Києві

Цей проєкт збирає історичні погодні дані з Open-Meteo API та застосовує модель машинного навчання для прогнозування кількості опадів.
Основна мета — навчити модель знаходити залежності між температурою, вітром, днем тижня, сезоном та кількістю опадів.

---

# Структура проєкту

```bash
project/
├── data/
│   ├── open_meteo_weather_2015.csv
│   ├── open_meteo_weather_2016.csv
│   ├── ...
│   └── open_meteo_weather.csv
├── get_data.py        # Скрипт для збору історичних погодних даних через Open-Meteo API
├── ml_model.py        # Побудова моделі машинного навчання для прогнозу опадів
├── requirements.txt   # Необхідні бібліотеки
├── LICENSE            # Ліцензія проєкту (MIT, GPL тощо)
└── README.md          # Документація
```
---

# Збір даних (get_data.py)

Скрипт отримує історичні дані про погоду в Києві за допомогою Open-Meteo Archive API.
Дані збираються по 14 днів, щоб уникнути обмежень API.

 Параметри:

- Місто: Київ

- Координати: LAT = 50.45, LON = 30.52

- Джерело: https://open-meteo.com
  - temperature_2m_max — максимальна температура

  - temperature_2m_min — мінімальна температура

  - precipitation_sum — сума опадів

  - windspeed_10m_max — максимальна швидкість вітру

  Запуск:
  ```bash
  python MlWeather.py
  ```
  Результат:
  
  CSV-файли у папці data/,
  наприклад:
  ```bash
  data/open_meteo_weather_2015.csv
  data/open_meteo_weather_2016.csv
  ...
  data/open_meteo_weather.csv
  ```

---

  # 2. Модель машинного навчання (ml_model.py)

  Модуль:

  1.Об’єднує всі CSV у єдиний DataFrame.

  2.Формує нові ознаки (features):

   - temp_mean, temp_diff — середня та різниця температур.

   - temp_wind_interaction, tempdiff_wind_interaction, temp_weekday_interaction — взаємодія факторів.

   - precip_lag1/2/3 — лагові значення опадів.

   - precip_roll3/5/7 — ковзні середні.

   - season, weekday — пори року та дні тижня.

  3. Навчає модель DecisionTreeRegressor (max_depth=5).

  4. Виводить метрики якості:

   - MSE (Mean Squared Error)

   - R² (коефіцієнт детермінації)

---

Приклад виконання
```bash
python MachineLearning.py
```
Вивід прикладу:
```bash
MSE: 19.66054557247043
R²: -0.009644364857032572
```

---

3. Встановлення залежностей

Рекомендується створити віртуальне середовище:
```bash
python -m venv venv
source venv/bin/activate  # або venv\Scripts\activate на Windows
pip install -r requirements.txt
```
Файл requirements.txt:
```bash
pandas
numpy
scikit-learn
requests
```

---

4. Використані технології
   
| Категорія | Технології |
|------------|-------------|
| **Мова** | Python 3.10+ |
| **Робота з даними** | pandas, numpy |
| **Машинне навчання** | scikit-learn |
| **API** | requests, Open-Meteo API |
| **Інше** | csv, datetime, os, time |

---

5. Подальші покращення

- Використати RandomForestRegressor або GradientBoosting

- Додати візуалізацію результатів (matplotlib, seaborn)

- Реалізувати вебінтерфейс (Flask / Streamlit)

- Автоматично оновлювати дані за поточний рік

---

6. Ліцензія
   
This project is licensed under the [MIT License](LICENSE).
