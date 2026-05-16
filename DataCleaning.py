import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import warnings

#Here I will load the data for the models

file_path = 'GlobalWeatherRepository.csv'

data = pd.read_csv(
    file_path,
    parse_dates=['last_updated'],
    dtype={
        'temperature_celsius': 'float32',
        'feels_like_celsius': 'float32',
        'humidity': 'float32',
        'pressure_mb': 'float32',
        'wind_kph': 'float32',
        'precip_mm': 'float32',
        'visibility_km': 'float32'
    },
    na_values=['', 'N/A', 'null']
)

#printing a bit for check
print('Data shape:', data.shape)
print("\nFirst few rows:")
print(data.head())

#making datetime features
clean_data = data.copy()
clean_data['hour'] = clean_data['last_updated'].dt.hour
clean_data['day_of_week'] = clean_data['last_updated'].dt.dayofweek
clean_data['is_weekend'] = clean_data['day_of_week'].isin([5,6]).astype(int)

# Dealing with missing values
#numeric values
num_cols = clean_data.select_dtypes(include='number').columns
for col in num_cols:
    if clean_data[col].isnull().any():
        clean_data[col] = clean_data[col].fillna(clean_data[col].median())

#category missing values
clean_data['condition_text'] = clean_data['condition_text'].fillna('Unknown')
clean_data['wind_direction'] = clean_data['wind_direction'].fillna('Unknown')

#dealing with duplicates
clean_data = clean_data.drop_duplicates()

#Dealing with temperature outliers and performing clipping
clean_data['temp_c'] = clean_data['temperature_celsius'].clip(lower=-60, upper=60)

clean_data['temp_outlier'] = (
    (data['temperature_celsius'] < -60) |
    (data['temperature_celsius'] > 60)
).astype(int)   # 1 = outlier, 0 = normal

#changing some naming for easy use
clean_data = clean_data.rename(columns={
    'temperature_celsius': 'temp_c_original',   # keep original just in case
    'feels_like_celsius': 'feels_like_c',
    'wind_kph': 'wind_kph',
    'pressure_mb': 'pressure_mb',
    'precip_mm': 'precip_mm',
    'humidity': 'humidity_pct',
    'cloud': 'cloud_pct',
    'visibility_km': 'visibility_km'
})

#setting some data as specifically categories
cat_cols = ['condition_text', 'wind_direction', 'moon_phase']
for col in cat_cols:
    if col in clean_data.columns:
        clean_data[col] = clean_data[col].astype('category')

#checking shape and saving
print("\nCleaned shape:", clean_data.shape)
print("\nMissing values remaining:")
print(clean_data.isnull().sum()[clean_data.isnull().sum() > 0])

print("\nTemperature outliers found:", clean_data['temp_outlier'].sum())

# Save cleaned version
clean_data.to_csv('weather_cleaned.csv', index=False)
clean_data.to_parquet('weather_cleaned.parquet')   # faster for future use

print("\n Cleaned data saved")