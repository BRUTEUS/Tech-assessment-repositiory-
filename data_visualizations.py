import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create folder for saving our plots and graphs
os.makedirs('visualizations', exist_ok=True)

# load our data
df = pd.read_csv('weather_cleaned.csv')

print("Dataset shape:", df.shape)

# Temperature and precipitation plots

plt.figure(figsize=(15, 10))

# 1. Temperature Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['temp_c'], kde=True, bins=50, color='orange')
plt.title('Temperature Distribution (°C)')
plt.xlabel('Temperature (°C)')

# 2. Top 10 Hottest Countries
plt.subplot(2, 2, 2)
country_temp = df.groupby('country')['temp_c'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=country_temp.values, y=country_temp.index, palette='Reds_d')
plt.title('Top 10 Hottest Countries (Avg Temp)')
plt.xlabel('Average Temperature (°C)')

# 3. Precipitation Distribution
plt.subplot(2, 2, 3)
sns.histplot(df['precip_mm'], kde=True, bins=50, color='skyblue')
plt.title('Precipitation Distribution (mm)')
plt.xlabel('Precipitation (mm)')

# 4. Temperature vs Humidity
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='temp_c', y='humidity_pct', alpha=0.5, color='teal')
plt.title('Temperature vs Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')

plt.tight_layout()
plt.savefig('visualizations/temperature_precip.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=['number']).columns
corr = df[numeric_cols].corr()

sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Air Quality and common conditions
plt.figure(figsize=(12, 6))
air_quality = df.groupby('country')['air_quality_us-epa-index'].mean().sort_values(ascending=False).head(12)
sns.barplot(x=air_quality.values, y=air_quality.index, palette='viridis')
plt.title('Worst Air Quality by Country (US EPA Index)')
plt.xlabel('Average Air Quality Index')
plt.tight_layout()
plt.savefig('visualizations/air_quality_by_country.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
top_conditions = df['condition_text'].value_counts().head(10)
sns.barplot(x=top_conditions.values, y=top_conditions.index, palette='Blues_d')
plt.title('Top 10 Most Common Weather Conditions')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('visualizations/common_conditions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll visualizations successfully saved in the 'visualizations/' folder")