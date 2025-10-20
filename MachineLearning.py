import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Data --------------------
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
data['date'] = pd.to_datetime(data[['year','month','day']])
data['weekday'] = data['date'].dt.weekday

# -------------------- Feature Engineering --------------------
data['temp_wind_interaction'] = data['temp_mean'] * data['wind_max']
data['tempdiff_wind_interaction'] = data['temp_diff'] * data['wind_max']
data['temp_weekday_interaction'] = data['temp_mean'] * data['weekday']

def get_season(month):
    if month in [12,1,2]:
        return 0  # Winter
    elif month in [3,4,5]:
        return 1  # Spring
    elif month in [6,7,8]:
        return 2  # Summer
    else:
        return 3  # Autumn

data['season'] = data['month'].apply(get_season)

# Lag features
for lag in range(1,4):
    data[f'precip_lag{lag}'] = data['precipitation'].shift(lag)

# Rolling mean features
for window in [3,5,7]:
    data[f'precip_roll{window}'] = data['precipitation'].shift(1).rolling(window=window).mean()

data = data.dropna()

# -------------------- Train/Test Split --------------------
features = ['year','month','day','temp_max','temp_min','temp_mean','temp_diff',
            'weekday','season',
            'precip_lag1','precip_lag2','precip_lag3',
            'precip_roll3','precip_roll5','precip_roll7',
            'temp_wind_interaction','tempdiff_wind_interaction','temp_weekday_interaction']

X = data[features]
y = data['precipitation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -------------------- Model --------------------
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------- Comparison --------------------
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
comparison['Error'] = abs(comparison['Actual'] - comparison['Predicted'])

# -------------------- Matplotlib Plots --------------------
plt.style.use('ggplot')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,5))

# 1. Actual vs Predicted Precipitation
axes[0].plot(comparison['Actual'].values[:200], label='Actual', color='blue', linewidth=2)
axes[0].plot(comparison['Predicted'].values[:200], label='Predicted', color='orange', linewidth=2, linestyle='--')
axes[0].set_title('Actual vs Predicted Precipitation', fontsize=14)
axes[0].set_xlabel('Day', fontsize=12)
axes[0].set_ylabel('Precipitation (mm)', fontsize=12)
axes[0].legend(fontsize=12)

# 2. Model Error per Day
axes[1].plot(comparison['Error'].values[:200], color='red', linewidth=2)
axes[1].set_title('Absolute Model Error', fontsize=14)
axes[1].set_xlabel('Day', fontsize=12)
axes[1].set_ylabel('Error (mm)', fontsize=12)

plt.tight_layout()
plt.show()

# -------------------- Top Errors --------------------
print("Top-10 Largest Forecast Errors:")
print(comparison.sort_values(by='Error', ascending=False).head(10))
print("\nFirst 10 Predictions:")
print(comparison.head(10))

# -------------------- Metrics --------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMSE: {mse:.3f}")
print(f"R²: {r2:.3f}")
