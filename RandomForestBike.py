import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
os.chdir(r"C:\Users\AARINI\Downloads")
bike_train = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")   # has casual, registered, count
bike_test  = pd.read_csv(r"C:\Users\AARINI\Downloads\test.csv")    # unlabeled -- to predict on

def engineer(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour']      = df['datetime'].dt.hour
    df['day']       = df['datetime'].dt.day
    df['month']     = df['datetime'].dt.month
    df['year']      = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    return df

bike_train = engineer(bike_train)
bike_test  = engineer(bike_test)

# Feature isolation & dummy encoding
# (casual/registered/count are TARGETS -- excluded from features)
feature_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                 'humidity', 'windspeed', 'hour', 'day', 'month', 'year', 'dayofweek']
cat_cols = ['season', 'weather', 'holiday', 'workingday']

x1 = pd.get_dummies(bike_train[feature_cols], columns=cat_cols, drop_first=True)
x_test_final = pd.get_dummies(bike_test[feature_cols], columns=cat_cols, drop_first=True)
x_test_final = x_test_final.reindex(columns=x1.columns, fill_value=0)  # align dummy columns
# RMSE evaluation function
def rmse_log(test_y, predicted_y):
    t1 = np.expm1(test_y)
    t2 = np.expm1(predicted_y)
    rmse_test = np.sqrt(mean_squared_error(t1, t2))
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    return {'RMSE-test from model': rmse_test, 'Base RMSE': rmse_base}

# Hyperparameter grid (random search space)
n_estimators      = [int(x) for x in np.linspace(10, 600, num=15)]
max_depth         = [int(x) for x in np.linspace(10, 110, num=10)]
min_samples_split = list(np.arange(100, 1100, 100))
min_samples_leaf  = [1, 2, 4, 10, 20]
max_features       = ['sqrt', 'log2', None]

random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'max_features': max_features
}
# Reusable pipeline: train/test split -> baseline RF -> RandomizedSearchCV
# -> refit tuned model on FULL train.csv for final prediction
def build_model(target_col, label):
    y1 = bike_train.filter([target_col], axis=1)
    y2 = np.log1p(y1.values.ravel())

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        x1, y2, test_size=0.3, random_state=3)
    print(f"\n===== {label} =====")
    print("Shapes (X_train, X_test, y_train, y_test):",
          X_train.shape, X_test.shape, y_train_log.shape, y_test_log.shape)

    # ---- Baseline Random Forest ----
    rf = RandomForestRegressor(n_estimators=220, max_depth=87, random_state=3)
    model_rf1 = rf.fit(X_train, y_train_log)
    preds_test = rf.predict(X_test)

    print(f"\n--- {label}: Baseline Evaluation Metrics ---")
    print(rmse_log(y_test_log, preds_test))
    print(f"Train R^2: {model_rf1.score(X_train, y_train_log):.4f}")
    print(f"Test R^2:  {model_rf1.score(X_test, y_test_log):.4f}")

    # ---- Hyperparameter tuning ----
    rf_base = RandomForestRegressor(random_state=3)
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=random_grid,
        n_iter=25,
        cv=3,
        verbose=1,
        random_state=3,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error'
    )
    rf_random.fit(X_train, y_train_log)
    best_rf = rf_random.best_estimator_

    print(f"\n--- {label}: Best Hyperparameters ---")
    print(rf_random.best_params_)

    best_preds_test = best_rf.predict(X_test)
    print(f"\n--- {label}: Tuned Evaluation Metrics ---")
    print(rmse_log(y_test_log, best_preds_test))
    print(f"Tuned Train R^2: {best_rf.score(X_train, y_train_log):.4f}")
    print(f"Tuned Test R^2:  {best_rf.score(X_test, y_test_log):.4f}")

    # ---- Refit tuned hyperparameters on FULL train.csv for final prediction ----
    final_model = RandomForestRegressor(random_state=3, **rf_random.best_params_)
    final_model.fit(x1, y2)
    return final_model

# Build separate models for casual and registered
# (count = casual + registered, so predicting them separately
#  and summing avoids the target-leakage problem)

model_casual     = build_model('casual', 'CASUAL')
model_registered = build_model('registered', 'REGISTERED')

# Predict on the real test.csv (no ground truth available there --
# these are the actual submission-style predictions)
pred_casual     = np.expm1(model_casual.predict(x_test_final)).clip(min=0)
pred_registered = np.expm1(model_registered.predict(x_test_final)).clip(min=0)
pred_count      = pred_casual + pred_registered

output = pd.DataFrame({
    'datetime': bike_test['datetime'],
    'casual': pred_casual.round().astype(int),
    'registered': pred_registered.round().astype(int),
    'count': pred_count.round().astype(int)
})
output.to_csv('test_predictions_tuned.csv', index=False)
print("\nSaved predictions to test_predictions_tuned.csv")
print(output.head())