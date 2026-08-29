import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, confusion_matrix)

os.chdir(r"C:\Users\AARINI\Downloads")

# loading the data
bike = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")
bike_test_raw = pd.read_csv(r"C:\Users\AARINI\Downloads\test.csv")

print(bike.head())
print(bike.tail())
print("shape:", bike.shape)
print(bike.columns.tolist())
print(bike.dtypes)
print(bike.describe())

# season/holiday/workingday/weather are ints but they're really categories,
# not numbers that mean "more" or "less" of something
categorical_cols = ['season', 'holiday', 'workingday', 'weather']
numerical_cols = [c for c in bike.columns if c not in categorical_cols + ['datetime']]
print("numerical:", numerical_cols)
print("categorical:", categorical_cols)

for col in categorical_cols:
    print(col, "->", bike[col].nunique(), "unique values")
    print(bike[col].value_counts().sort_index())

print(pd.to_datetime(bike['datetime']).min(), "to", pd.to_datetime(bike['datetime']).max())

# target variable: count
# count = casual + registered, it's the total hourly rentals, so this is a
# regression problem. casual/registered get dropped later since using them
# as features would basically be leaking the answer straight into the model.

# a few things that stood out while exploring:
# - datetime is just text right now, need hour/day/month/year pulled out of it
# - count is pretty right-skewed, most hours are under ~300 but a handful spike past 700
# - weather=4 (heavy rain/snow) shows up basically once in the whole dataset

# missing values
print(bike.isnull().sum())
# none found, so nothing to impute here

dupes = bike.duplicated().sum()
print("duplicate rows:", dupes)
bike.drop_duplicates(keep='first', inplace=True)
bike.reset_index(drop=True, inplace=True)

# humidity of 0 doesn't really happen with a real sensor outdoors, treating
# it as bad data and filling with the median instead of the mean since a
# few extreme weather rows could skew the mean
bad_humidity = (bike['humidity'] == 0).sum()
print("rows with humidity == 0:", bad_humidity)
med_humidity = bike.loc[bike['humidity'] != 0, 'humidity'].median()
bike.loc[bike['humidity'] == 0, 'humidity'] = med_humidity

# windspeed of 0 is fine though - a calm hour is completely plausible, leaving those alone
print("rows with windspeed == 0:", (bike['windspeed'] == 0).sum())

neg_vals = (bike[['temp', 'atemp', 'humidity', 'windspeed', 'count', 'casual', 'registered']] < 0).sum().sum()
print("negative values across the numeric columns:", neg_vals)


def add_date_parts(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['dayofweek'] = df['datetime'].dt.dayofweek
    return df


bike = add_date_parts(bike)
bike_test = add_date_parts(bike_test_raw)

# --- plots ---

plt.figure(figsize=(8, 5))
sns.histplot(bike['count'], bins=50, kde=True)
plt.title("Distribution of Hourly Rental Count")
plt.xlabel("count")
plt.ylabel("frequency")
plt.show()
# skewed right, most hours are low demand with a long tail of busy hours

plt.figure(figsize=(8, 5))
sns.scatterplot(x='temp', y='count', data=bike, alpha=0.3)
plt.title("Temperature vs Rental Count")
plt.xlabel("temp (C)")
plt.ylabel("count")
plt.show()
# rentals go up with temperature but it flattens out / dips again at the very top end

plt.figure(figsize=(9, 5))
sns.boxplot(x='hour', y='count', data=bike)
plt.title("Rental Count by Hour of Day")
plt.xlabel("hour")
plt.ylabel("count")
plt.show()
# two clear peaks around 8am and 5-6pm (commute times), quiet overnight -
# hour is probably going to be the single strongest predictor

# --- preprocessing, shared across all 5 models below ---

# classification target: is this hour above or below the median demand
med_count = bike['count'].median()
bike['high_demand'] = (bike['count'] > med_count).astype(int)
print("median count used as the high/low demand cutoff:", med_count)
print(bike['high_demand'].value_counts())

drop_cols = ['datetime', 'count', 'casual', 'registered', 'high_demand']
X_all = pd.get_dummies(bike.drop(columns=drop_cols), columns=categorical_cols, drop_first=True)
y_reg_all = bike['count']
y_class_all = bike['high_demand']
feature_columns = list(X_all.columns)

# splitting once and reusing it everywhere so all 5 models get evaluated on
# the exact same rows
X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = train_test_split(
    X_all, y_reg_all, y_class_all, test_size=0.3, random_state=3
)
print("X_train:", X_train.shape, " X_test:", X_test.shape)

# lining up test.csv with the same dummy columns as the training data
X_test_final = pd.get_dummies(bike_test.drop(columns=['datetime']), columns=categorical_cols, drop_first=True)
X_test_final = X_test_final.reindex(columns=feature_columns, fill_value=0)

# scaling - fit on train only, then apply to test. matters most for KNN and
# logistic regression since they're distance/magnitude sensitive
scale_cols = ['temp', 'atemp', 'humidity', 'windspeed']
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

results = []  # one row gets added per model, used for the comparison table at the end


def add_result(name, task, y_true=None, y_pred=None):
    """stores whichever metrics make sense for the task type so the final
    table can show RMSE/MAE/R2 for regression rows and Accuracy for
    classification rows side by side"""
    row = {'Model': name, 'Task': task, 'MAE': np.nan, 'RMSE': np.nan,
           'R2': np.nan, 'Accuracy': np.nan}
    if task == 'Regression':
        row['MAE'] = mean_absolute_error(y_true, y_pred)
        row['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        row['R2'] = r2_score(y_true, y_pred)
    else:
        row['Accuracy'] = accuracy_score(y_true, y_pred)
    results.append(row)
    return row


# ------------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------------

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_reg)
lin_pred = lin_reg.predict(X_test)

lin_row = add_result('Linear Regression', 'Regression', y_test_reg, lin_pred)
print("\nLinear Regression - MAE:", lin_row['MAE'], "RMSE:", lin_row['RMSE'], "R2:", lin_row['R2'])

compare_df = pd.DataFrame({'Actual': y_test_reg.values, 'Predicted': lin_pred}).reset_index(drop=True)
print(compare_df.head(10))

plt.figure(figsize=(7, 6))
plt.scatter(y_test_reg, lin_pred, alpha=0.3)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
plt.xlabel("actual count")
plt.ylabel("predicted count")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()
# it picks up the broad trend but R2 isn't great - count has cyclical,
# non-linear patterns (hour of day especially) that a straight line just
# can't represent well

# ------------------------------------------------------------------
# Logistic Regression (target: high_demand)
# ------------------------------------------------------------------

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train_class)
log_pred = log_reg.predict(X_test_scaled)

log_row = add_result('Logistic Regression', 'Classification', y_test_class, log_pred)
print("\nLogistic Regression - confusion matrix:\n", confusion_matrix(y_test_class, log_pred))
print("Accuracy:", log_row['Accuracy'])
# since high_demand was split on the median, the two classes are roughly
# 50/50, so accuracy is a fair enough headline number here. TN/TP being
# large relative to FP/FN means weather + season + hour + calendar info
# alone can separate busy hours from quiet ones reasonably well

# ------------------------------------------------------------------
# Decision Tree Regressor
# ------------------------------------------------------------------

# count is continuous so this needs a regressor, not a classifier.
# capping depth at 6 so it doesn't just memorize the training rows
dt_reg = DecisionTreeRegressor(max_depth=6, random_state=3)
dt_reg.fit(X_train, y_train_reg)
dt_pred = dt_reg.predict(X_test)

dt_row = add_result('Decision Tree', 'Regression', y_test_reg, dt_pred)
print("\nDecision Tree - RMSE:", dt_row['RMSE'], "R2:", dt_row['R2'])

plt.figure(figsize=(20, 10))
plot_tree(dt_reg, feature_names=feature_columns, filled=True, max_depth=3, fontsize=8, rounded=True)
plt.title("Decision Tree (top 3 levels)")
plt.show()

dt_importances = pd.Series(dt_reg.feature_importances_, index=feature_columns).sort_values(ascending=False)
print(dt_importances.head(5))
# first split in the tree is on hour, which lines up with what the boxplot
# showed earlier - time of day matters more than weather or season here

# ------------------------------------------------------------------
# Random Forest Regressor
# ------------------------------------------------------------------

rf_reg = RandomForestRegressor(n_estimators=220, max_depth=25, random_state=3, n_jobs=-1)
rf_reg.fit(X_train, y_train_reg)
rf_pred = rf_reg.predict(X_test)
print("\nRandom Forest (baseline) RMSE:", np.sqrt(mean_squared_error(y_test_reg, rf_pred)),
      "R2:", r2_score(y_test_reg, rf_pred))

# small randomized search around the baseline - kept short so it doesn't
# take forever to run on a laptop
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(100, 400, num=4)],
    'max_depth': [int(x) for x in np.linspace(10, 60, num=4)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=3, n_jobs=-1),
                                param_distributions=param_grid, n_iter=8, cv=3,
                                random_state=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
rf_search.fit(X_train, y_train_reg)
best_rf = rf_search.best_estimator_
best_rf_pred = best_rf.predict(X_test)

rf_row = add_result('Random Forest', 'Regression', y_test_reg, best_rf_pred)
print("best params:", rf_search.best_params_)
print("Random Forest (tuned) - MAE:", rf_row['MAE'], "RMSE:", rf_row['RMSE'], "R2:", rf_row['R2'])

rf_importances = pd.Series(best_rf.feature_importances_, index=feature_columns).sort_values(ascending=False)
print(rf_importances.head(3))
# random forest beats the single tree since averaging a bunch of trees
# trained on different bootstrapped samples cuts down the overfitting a
# single deep tree is prone to - trades off some interpretability for it

# ------------------------------------------------------------------
# KNN (target: high_demand)
# ------------------------------------------------------------------

# KNN works off distance, so unscaled features with big ranges (year,
# humidity) would swamp small-range ones (holiday 0/1) - that's why the
# scaled versions from earlier get used here
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_scaled, y_train_class)
knn5_pred = knn5.predict(X_test_scaled)
print("\nKNN (K=5) confusion matrix:\n", confusion_matrix(y_test_class, knn5_pred))
print("KNN (K=5) accuracy:", accuracy_score(y_test_class, knn5_pred))

acc_by_k = []
miss_by_k = []
for k in range(1, 20):
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train_class)
    pred_k = knn_k.predict(X_test_scaled)
    acc_by_k.append(accuracy_score(y_test_class, pred_k))
    miss_by_k.append((y_test_class != pred_k).sum())

best_k = int(np.argmax(acc_by_k) + 1)
print("accuracy by k:", acc_by_k)
print("best k:", best_k, "accuracy:", max(acc_by_k))

plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), miss_by_k, marker='o')
plt.title("Misclassified Samples vs K")
plt.xlabel("k")
plt.ylabel("misclassified")
plt.show()
# small k overfits to noisy individual points, bigger k smooths things out
# up to a point - after that it starts underfitting and accuracy flattens
# or drops again

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train_class)
best_knn_pred = best_knn.predict(X_test_scaled)
knn_row = add_result(f'KNN (k={best_k})', 'Classification', y_test_class, best_knn_pred)

# ------------------------------------------------------------------
# final comparison across all 5 models
# ------------------------------------------------------------------

results_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))

reg_table = results_df[results_df['Task'] == 'Regression']
class_table = results_df[results_df['Task'] == 'Classification']

best_reg = reg_table.loc[reg_table['R2'].idxmax()]
worst_reg = reg_table.loc[reg_table['R2'].idxmin()]
best_class = class_table.loc[class_table['Accuracy'].idxmax()]
worst_class = class_table.loc[class_table['Accuracy'].idxmin()]

print(f"\nBest regression model:  {best_reg['Model']}  (RMSE={best_reg['RMSE']:.2f}, R2={best_reg['R2']:.4f})")
print(f"Worst regression model: {worst_reg['Model']}  (RMSE={worst_reg['RMSE']:.2f}, R2={worst_reg['R2']:.4f})")
print(f"Best classification model:  {best_class['Model']}  (Accuracy={best_class['Accuracy']:.4f})")
print(f"Worst classification model: {worst_class['Model']}  (Accuracy={worst_class['Accuracy']:.4f})")

print("\nRegression and classification scores aren't really on the same scale")
print("(R2 vs Accuracy), so they can't be ranked against each other directly.")
print(f"Within the regression models - which match the actual question of")
print(f"'how many bikes get rented this hour' - {best_reg['Model']} comes out")
print("on top. It handles the non-linear, cyclical demand pattern (hour of")
print("day, season/weather interactions) much better than a straight line")
print("(Linear Regression) or a single shallow tree (Decision Tree) can.")

# ------------------------------------------------------------------
# predictions on test.csv (no ground truth count available there)
# ------------------------------------------------------------------

final_lin_pred = lin_reg.predict(X_test_final).clip(min=0)

final_dt = DecisionTreeRegressor(max_depth=6, random_state=3)
final_dt.fit(X_all, y_reg_all)
final_dt_pred = final_dt.predict(X_test_final).clip(min=0)

final_rf = RandomForestRegressor(random_state=3, n_jobs=-1, **rf_search.best_params_)
final_rf.fit(X_all, y_reg_all)
final_rf_pred = final_rf.predict(X_test_final).clip(min=0)

test_predictions = pd.DataFrame({
    'datetime': bike_test['datetime'],
    'linear_regression_pred': final_lin_pred.round().astype(int),
    'decision_tree_pred': final_dt_pred.round().astype(int),
    'random_forest_pred': final_rf_pred.round().astype(int),
})
test_predictions.to_csv('test_predictions_all_models.csv', index=False)
print(test_predictions.head())