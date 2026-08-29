import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

os.chdir(r"C:\Users\AARINI\Downloads")

bike_train = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")
bike_test = pd.read_csv(r"C:\Users\AARINI\Downloads\test.csv")

data = bike_train.copy()

print(data.info())
print('Data columns with null values:\n', data.isnull().sum())
print('No missing values found.')

summary_num = data.describe()
print(summary_num)

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

"""
Target Formulation:
'count' is a continuous variable (total hourly rentals), so it cannot be
used directly as a classification target -- Logistic Regression predicts
discrete classes, not continuous quantities.

A sensible binary target is defined instead: whether a given hour is a
"high demand" hour or not, based on whether count is above or below the
median count across the training data. This mirrors how SalStat was
defined as <=50k vs >50k in the income example -- a continuous quantity
converted into a meaningful binary split.
high_demand = 1  if count  > median(count)
high_demand = 0  if count <= median(count)
"""
median_count = data['count'].median()
print('Median count used as threshold:', median_count)

data['high_demand'] = (data['count'] > median_count).astype(int)
print(data['high_demand'].value_counts())

demand_plot = sns.countplot(x=data['high_demand'])

cols_to_drop = ['datetime', 'count', 'casual', 'registered']
data2 = data.drop(columns=cols_to_drop)

new_data = pd.get_dummies(data2, columns=['season', 'weather', 'holiday', 'workingday'],
                           drop_first=True)

columns_list = list(new_data.columns)
print(columns_list)

features = list(set(columns_list) - set(['high_demand']))
print(features)

y = new_data['high_demand'].values
print(y)

x = new_data[features].values
print(x)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=3)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

logistic = LogisticRegression(max_iter=1000)

logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

prediction = logistic.predict(test_x)
print(prediction)

conf_matrix = confusion_matrix(test_y, prediction)
print(conf_matrix)

acc_score = accuracy_score(test_y, prediction)
print(acc_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())
"""
TN (True Negative)  = correctly predicted low-demand hours
TP (True Positive)  = correctly predicted high-demand hours
FP (False Positive) = predicted high-demand, but was actually low-demand
FN (False Negative) = predicted low-demand, but was actually high-demand

Accuracy tells us the overall proportion of hours the model classified
correctly (high vs low demand). Because the target was built from a
median split, the two classes are roughly balanced (~50/50), so accuracy
is a reasonably fair metric here (it would be misleading on an imbalanced
target, but that is not the situation in this case).

If TN and TP are both large relative to FP and FN, the model is doing a
good job separating high-demand hours (e.g. commute times) from
low-demand hours (e.g. late night) using weather, season, hour and
calendar features alone.
"""

print('\nApplying trained model to test.csv (predicting high_demand class only')
print('-- test.csv has no count/casual/registered columns, so only the')
print('binary high_demand label can be produced, not actual counts):')

test_data2 = bike_test.drop(columns=['datetime'])
new_test_data = pd.get_dummies(test_data2, columns=['season', 'weather', 'holiday', 'workingday'],
                                drop_first=True)
new_test_data = new_test_data.reindex(columns=features, fill_value=0)
new_test_data_scaled = scaler.transform(new_test_data.values)

test_predictions = logistic.predict(new_test_data_scaled)

output = pd.DataFrame({
    'datetime': bike_test['datetime'],
    'predicted_high_demand': test_predictions
})
output.to_csv('test_predictions_logistic.csv', index=False)
print(output.head())
