import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Load and feature engineering copy pasted from testing_models.py
clean_df = pd.read_csv('weather_cleaned.csv')
df = clean_df.copy()


# --- Reuse the same feature engineering ---
df['temp_humidity_index'] = df['temp_c'] * df['humidity_pct']
df['feels_like_diff'] = df['feels_like_c'] - df['temp_c']
df['heat_index_approx'] = df['temp_c'] + (0.33 * df['humidity_pct'] / 100 * (df['temp_c'] - 14))
df['is_humid'] = (df['humidity_pct'] > 85).astype(int)
df['is_windy'] = (df['wind_kph'] > 20).astype(int)
df['low_visibility'] = (df['visibility_km'] < 5).astype(int)
df['precip_risk'] = df['precip_mm'] * (df['cloud_pct'] / 100)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['distance_from_equator'] = np.abs(df.get('latitude', 0))


feature_cols = ['temp_c', 'feels_like_c', 'humidity_pct', 'wind_kph', 'pressure_mb',
                'precip_mm', 'cloud_pct', 'visibility_km', 'hour', 'is_weekend',
                'temp_outlier', 'temp_humidity_index', 'feels_like_diff',
                'heat_index_approx', 'is_humid', 'is_windy', 'low_visibility',
                'precip_risk', 'hour_sin', 'hour_cos', 'distance_from_equator']


y = df['condition_text']
valid = y.value_counts()[y.value_counts() >= 2].index
df = df[df['condition_text'].isin(valid)].copy()

X = df[feature_cols]
y_encoded = LabelEncoder().fit_transform(df['condition_text'])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2,
                                                    random_state=42, stratify=y_encoded)


#Optuna objective defining

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 6, 18),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)


# optuna tuning run
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)  # Increase to 100+ if you want

print("Best hyperparameters:", study.best_params)
print("Best accuracy:", study.best_value)

# Train final model with best params
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)
final_pred = best_model.predict(X_test)
print("Final Tuned Accuracy:", accuracy_score(y_test, final_pred))