import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
all_data = []

for year in years:
    filename = f"data/open_meteo_weather_{year}.csv" if year != 2025 else "data/open_meteo_weather.csv"
    df = pd.read_csv(filename)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

data['year'] = data['year'].astype(int)
data['month'] = data['month'].astype(int)
data['day'] = data['day'].astype(int)

data['temp_max'] = data['temp_max'].astype(float)
data['temp_min'] = data['temp_min'].astype(float)
data['precipitation'] = data['precipitation'].astype(float)
data['wind_max'] = data['wind_max'].astype(float)

data['temp_mean'] = (data['temp_max'] + data['temp_min']) / 2
data['temp_diff'] = data['temp_max'] - data['temp_min']

data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
data['weekday'] = data['date'].dt.weekday

# фіча взаємодії temp_mean * wind_max
data['temp_wind_interaction'] = data['temp_mean'] * data['wind_max']

# ще приклади
data['tempdiff_wind_interaction'] = data['temp_diff'] * data['wind_max']
data['temp_weekday_interaction'] = data['temp_mean'] * data['weekday']


def get_season(month):
    if month in [12, 1, 2]:
        return 0  # зима
    elif month in [3, 4, 5]:
        return 1  # весна
    elif month in [6, 7, 8]:
        return 2  # літо
    else:
        return 3  # осінь


data['season'] = data['month'].apply(get_season)

# Лаги
data['precip_lag1'] = data['precipitation'].shift(1)
data['precip_lag2'] = data['precipitation'].shift(2)
data['precip_lag3'] = data['precipitation'].shift(3)

# Ковзні середні
data['precip_roll3'] = data['precipitation'].shift(1).rolling(window=3).mean()
data['precip_roll5'] = data['precipitation'].shift(1).rolling(window=5).mean()
data['precip_roll7'] = data['precipitation'].shift(1).rolling(window=7).mean()

# Видаляємо пропуски, що з'явились через shift/rolling
data = data.dropna()

"""print(data.isna().sum())"""
x = data[['year', 'month', 'day', 'temp_max', 'temp_min', 'temp_mean', 'temp_diff',
          'weekday', 'season',
          'precip_lag1', 'precip_lag2', 'precip_lag3',
          'precip_roll3', 'precip_roll5', 'precip_roll7',
          'temp_wind_interaction', 'tempdiff_wind_interaction', 'temp_weekday_interaction']]

y = data['precipitation']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

'''#Маштабування  
scaler = StandardScaler()  
x_train_scaled = scaler.fit_transform(x_train)  
x_test_scaled = scaler.transform(x_test)'''

model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))