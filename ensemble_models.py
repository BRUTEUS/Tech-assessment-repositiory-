import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import time
warnings.filterwarnings('ignore')

#paste from our data loading and feature engineering
df = pd.read_csv('weather_cleaned.csv').copy()

# Reuse our best feature engineering
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

# Prepare data
feature_cols = ['temp_c', 'feels_like_c', 'humidity_pct', 'wind_kph', 'pressure_mb',
                'precip_mm', 'cloud_pct', 'visibility_km', 'hour', 'is_weekend',
                'temp_outlier', 'temp_humidity_index', 'feels_like_diff',
                'heat_index_approx', 'is_humid', 'is_windy', 'low_visibility',
                'precip_risk', 'hour_sin', 'hour_cos', 'distance_from_equator']

target_col = 'condition_text'

# Remove rare classes
valid_classes = df[target_col].value_counts()[df[target_col].value_counts() >= 2].index
df = df[df[target_col].isin(valid_classes)].copy()

X = df[feature_cols]
y = df[target_col]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# setup our individual models
rf = RandomForestClassifier(n_estimators=300, max_depth=18, n_jobs=-1, random_state=42)

xgb_model = xgb.XGBClassifier(
    n_estimators=410,
    max_depth=18,
    learning_rate=0.075,
    subsample=0.75,
    colsample_bytree=0.65,
    gamma=1.0,
    device='cuda',
    tree_method='hist',
    random_state=42
)

logreg = LogisticRegression(max_iter=2000, random_state=42)

# ENSEMBLE 1: Voting Classifier
print("Training Voting Ensemble (Hard Voting)...")
start = time.time()
ensemble_voting = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb_model),
        ('logreg', logreg)
    ],
    voting='hard',        # majority vote
    weights=[2, 3, 1]     # Give more weight to better models
)

ensemble_voting.fit(X_train, y_train)
pred_voting = ensemble_voting.predict(X_test)

print("Voting Ensemble Performance:")
print(f"Accuracy: {accuracy_score(y_test, pred_voting):.4f}")
print(f"F1 Score: {f1_score(y_test, pred_voting, average='weighted'):.4f}")
print(f"Time for running Voting Ensemble: {time.time() - start:.2f}s")

#ENSEMBLE 2: Simple Average (Soft Voting)
print("\nTraining Soft Voting Ensemble...")
start = time.time()

# Get probability predictions
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
logreg.fit(StandardScaler().fit_transform(X_train), y_train)  # scaled for logreg

pred_rf = rf.predict_proba(X_test)
pred_xgb = xgb_model.predict_proba(X_test)
pred_log = logreg.predict_proba(StandardScaler().fit_transform(X_test))

# Weighted average of probabilities
final_pred_proba = (2 * pred_rf + 3 * pred_xgb + 1 * pred_log) / 6
final_pred = np.argmax(final_pred_proba, axis=1)

print("Soft Voting Ensemble Performance:")
print(f"Accuracy: {accuracy_score(y_test, final_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, final_pred, average='weighted'):.4f}")
print(f"Time for running Voting Ensemble: {time.time() - start:.2f}s")

# FEATURE IMPORTANCE from XGBoost
print("\nTop 10 Feature Importances (from XGBoost):")
importance = xgb_model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print(feat_imp.head(10))

#plotting for easy viewing of the feature importance

plt.figure(figsize=(10, 8))
sns.barplot(x=feat_imp['Importance'][:15], y=feat_imp['Feature'][:15])
plt.title('Top 15 Most Important Features - XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300)
plt.show()