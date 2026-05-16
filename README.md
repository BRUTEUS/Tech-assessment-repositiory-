# Tech-assessment-repositiory-
Here is all the code for the tech assessment and how to execute.

## Global Weather Forecasting & Analysis Assessment

## Data Scientist / Analyst Role at PM Accelerator

## Assessment Overview

This assessment performs comprehensive analysis and forecasting on the Global Weather Repository dataset from Kaggle. The goal was to predict weather conditions (`condition_text`) across 48 classes while extracting meaningful global climate insights.

**Final Best Performance**:  
**XGBoost + Ensemble** achieved **~85.6% accuracy** on multi-class weather prediction.


## Key Highlights

- Cleaned and engineered 21+ features
- Built and compared multiple ML models with GPU acceleration
- Performed hyperparameter tuning using Optuna
- Created model ensemble (Voting Classifier)
- Conducted extensive EDA and visualizations
- Analyzed feature importance and geographical patterns


## Project Structure

- DataCleaning.py
- testing_models.py
- data_visualizations.py
- optuna_XGboost.py
- visualizations
- requirements.txt
- README.md
- weather_cleaned.csv


## Technologies Used

- Language: Python 3.11
- Data: pandas, numpy
- Visualization: matplotlib, seaborn
- ML: scikit-learn, XGBoost (GPU), LightGBM
- Optimization: Optuna
- Hardware: NVIDIA RTX 3090 Ti (CUDA enabled)

## Methodology

1. Data Cleaning — Handled missing values, outliers, and data types
2. Feature Engineering — Created interaction terms, cyclical time features (`hour_sin`, `hour_cos`), binary flags, and geographical features
3. EDA — Distributions, correlations, geographical analysis, air quality insights
4. Modeling — Random Forest, XGBoost, Logistic Regression
5. Hyperparameter Tuning — Optuna (50 trials)
6. Ensemble — Weighted Voting Classifier
7. Evaluation — Accuracy + F1 Score

## Key Insights

- Humidity, cloud cover, and temperature-humidity interactions are the strongest predictors of weather conditions.
- Countries closer to the equator show distinct weather patterns.
- Air quality shows strong correlation with certain weather conditions.
- Cyclical encoding of time features improved model performance.

## Results Summary

| Model                    | Accuracy | F1 Score |
|--------------------------|----------|----------|
| XGBoost (Tuned)          | 0.8559   | 0.8417   |
| Random Forest            | 0.8367   | 0.8200   |
| Ensemble (hardVoting)    | 0.8503   | 0.8376   |
| Ensemble (softVoting)    | 0.8516   | 0.8371   |

**Top Features ** (from XGBoost):
1        cloud_pct    0.289015
2  low_visibility    0.183030
3    visibility_km    0.088349
4        precip_mm    0.073477
5     precip_risk    0.059178
6        hour_cos    0.052238
7        is_humid    0.044072
8     humidity_pct    0.032918
9             hour    0.025034
10        hour_sin    0.018073

## How to Reproduce

```bash
git clone <repo-url>
cd Weather-Forecasting-Assessment

pip install -r requirements.txt

python data_cleaning.py
python data_visualizations.py
python final_ensemble.py

## explained
Download files from github link
Download the GlobalWeatherRepository.csv from kaggle into your project folder
Pip install –r requirements.txt
Run in order of DataCleaning.py, testing_models.py, optuna_XGboost.py, ensemble_models.py, then data_visualizations.py
