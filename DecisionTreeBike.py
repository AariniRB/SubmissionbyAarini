import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score

os.chdir(r"C:\Users\AARINI\Downloads")

data = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")
bike_test = pd.read_csv(r"C:\Users\AARINI\Downloads\test.csv")

data['datetime'] = pd.to_datetime(data['datetime'])
data['hour'] = data['datetime'].dt.hour
data['month'] = data['datetime'].dt.month
data['year'] = data['datetime'].dt.year
data['dayofweek'] = data['datetime'].dt.dayofweek

bike_test['datetime'] = pd.to_datetime(bike_test['datetime'])
bike_test['hour'] = bike_test['datetime'].dt.hour
bike_test['month'] = bike_test['datetime'].dt.month
bike_test['year'] = bike_test['datetime'].dt.year
bike_test['dayofweek'] = bike_test['datetime'].dt.dayofweek

# Model Choice
"""
'count' is a continuous numeric variable (total hourly rentals), so a
Decision Tree Regressor is used here, not a Decision Tree Classifier.
A Random Forest is simply an ensemble of many such trees, so this is the
single-tree version of that same idea -- easier to interpret and visualize,
but usually less accurate than the forest.
"""
cols_to_drop = ['datetime', 'count', 'casual', 'registered']
data2 = data.drop(columns=cols_to_drop)

new_data = pd.get_dummies(data2, columns=['season', 'weather', 'holiday', 'workingday'],
                           drop_first=True)

features = list(new_data.columns)
print(features)

x = new_data[features].values
y = data['count'].values

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3)

# Training & Evaluation
"""
max_depth is limited to 6 -- an unrestricted Decision Tree Regressor will
grow until every leaf is pure, essentially memorizing the training data
(severe overfitting). Limiting depth keeps the tree both more generalizable
and small enough to actually visualize in the next step.
"""
dt_regressor = DecisionTreeRegressor(max_depth=6, random_state=3)
dt_regressor.fit(train_x, train_y)

predictions = dt_regressor.predict(test_x)

rmse = np.sqrt(mean_squared_error(test_y, predictions))
r2 = r2_score(test_y, predictions)

print('Test RMSE:', rmse)
print('Test R^2:', r2)

base_pred = np.repeat(np.mean(test_y), len(test_y))
base_rmse = np.sqrt(mean_squared_error(test_y, base_pred))
print('Base RMSE (predicting the mean):', base_rmse)

# Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_regressor, feature_names=features, filled=True, max_depth=3,
          fontsize=8, rounded=True)
plt.title("Decision Tree Regressor Structure (top 3 levels shown)")
plt.show()

importances = pd.Series(dt_regressor.feature_importances_, index=features)
importances = importances.sort_values(ascending=False)
print('\nTop 5 feature importances:')
print(importances.head(5))

# Observation
"""
The very first split in the tree is on 'hour', confirming what the Random
Forest also found -- time of day is by far the strongest single predictor
of rental demand, more influential than weather, season, or temperature.
"""

# Predicting on test.csv
final_dt = DecisionTreeRegressor(max_depth=6, random_state=3)
final_dt.fit(x, y)

test_data2 = bike_test.drop(columns=['datetime'])
new_test_data = pd.get_dummies(test_data2, columns=['season', 'weather', 'holiday', 'workingday'],
                                drop_first=True)
new_test_data = new_test_data.reindex(columns=features, fill_value=0)

test_predictions = final_dt.predict(new_test_data.values).clip(min=0)

output = pd.DataFrame({
    'datetime': bike_test['datetime'],
    'predicted_count': test_predictions.round().astype(int)
})
output.to_csv('test_predictions_decision_tree.csv', index=False)
print(output.head())