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
import warnings                 #just cleaning up our console output
warnings.filterwarnings('ignore')

#Here I do feature engineering and fit 4 models

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

clean_df = pd.read_csv('weather_cleaned.csv')
print("Creating new features...")
df = clean_df.copy()

# Temperature and humidty
df['temp_humidity_index'] = df['temp_c'] * df['humidity_pct']
df['feels_like_diff'] = df['feels_like_c'] - df['temp_c']
df['heat_index_approx'] = df['temp_c'] + (0.33 * df['humidity_pct'] / 100 * (df['temp_c'] - 14))

# weather, wind, visibilty features
df['is_humid'] = (df['humidity_pct'] > 85).astype(int)
df['is_windy'] = (df['wind_kph'] > 20).astype(int)
df['low_visibility'] = (df['visibility_km'] < 5).astype(int)
df['precip_risk'] = df['precip_mm'] * (df['cloud_pct'] / 100)

# time based features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Here we consider distance from the equator to also be a factor
df['distance_from_equator'] = np.abs(df['latitude'])

# Here I prepare the data for loading into the models and for proper splitting later for test and train sets
target_col = 'condition_text'
feature_cols = ['temp_c', 'feels_like_c', 'humidity_pct', 'wind_kph', 'pressure_mb',
                'precip_mm', 'cloud_pct', 'visibility_km', 'hour', 'is_weekend',
                'temp_outlier', 'temp_humidity_index', 'feels_like_diff',
                'heat_index_approx', 'is_humid', 'is_windy', 'low_visibility',
                'precip_risk', 'hour_sin', 'hour_cos', 'distance_from_equator']

# Here I remove rare cases from our data that are too small so that the train_test_split doesn't run into errors
y = df[target_col]
valid_classes = y.value_counts()[y.value_counts() >= 2].index
df = df[df[target_col].isin(valid_classes)].copy()

X = df[feature_cols].copy()
y = df[target_col]

# encoding category for model use
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# splitting the data for train and test for the models
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"New feature count: {X.shape[1]} | Classes: {len(le.classes_)}")

# Here we're going to creat a function to plug and play our different models for running

results = []
def train_and_evaluate(name, model, use_scaled=False):
    print(f"\nTraining {name}...")
    start = time.time()

    if use_scaled:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    t = time.time() - start
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')

    results.append({'Model': name, 'Accuracy': round(acc, 4),
                    'F1': round(f1, 4), 'Time(s)': round(t, 2)})
    print(f"Done in {t:.1f}s | Acc: {acc:.4f} | F1: {f1:.4f}")


# Now that I've setup our training function we train and compare the results
rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
train_and_evaluate("Random Forest", rf)

xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=12, learning_rate=0.1,
                              device='cuda', tree_method='hist', random_state=42)
train_and_evaluate("XGBoost (GPU)", xgb_model)

logreg = LogisticRegression(max_iter=2000, random_state=42)
train_and_evaluate("Logistic Regression (Scaled)", logreg, use_scaled=True)

lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=12, learning_rate=0.1,
                               random_state=42, verbose=-1)
train_and_evaluate("LightGBM (CPU)", lgb_model)

#Print out the results for assessment
print("\n" + "=" * 60)
print("RESULTS WITH NEW FEATURES")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.sort_values('Accuracy', ascending=False))