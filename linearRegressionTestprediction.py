import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

os.chdir(r"C:\Users\AARINI\Downloads")

train = pd.read_csv(r"C:\Users\AARINI\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\AARINI\Downloads\test.csv")

train.drop_duplicates(keep='first', inplace=True)

train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['dayofweek'] = train['datetime'].dt.dayofweek

test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['dayofweek'] = test['datetime'].dt.dayofweek

test_datetime = test['datetime']

feature_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                 'humidity', 'windspeed', 'hour', 'year', 'month', 'dayofweek']
cat_cols = ['season', 'weather', 'year', 'month', 'dayofweek', 'hour']

train_feat = train[feature_cols].dropna().reset_index(drop=True)
train_targets = train.loc[train_feat.index, ['casual', 'registered', 'count']].reset_index(drop=True)
test_feat = test[feature_cols].reset_index(drop=True)

combined = pd.concat([train_feat, test_feat], keys=['train', 'test'])
combined_dummies = pd.get_dummies(combined, columns=cat_cols, drop_first=True, dtype=int)

X_train = combined_dummies.xs('train').astype(float).reset_index(drop=True)
X_test = combined_dummies.xs('test').astype(float).reset_index(drop=True)

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test, has_constant='add')
X_test_const = X_test_const.reindex(columns=X_train_const.columns, fill_value=0)

def fit_and_predict(target_name):
    y_log = np.log1p(train_targets[target_name])
    model = sm.OLS(y_log, X_train_const).fit()
    pred_log = model.predict(X_test_const)
    pred = np.expm1(pred_log)
    pred = pred.clip(lower=0)
    print(target_name, "R-squared:", model.rsquared)
    return pred

pred_casual = fit_and_predict('casual')
pred_registered = fit_and_predict('registered')
pred_count = pred_casual + pred_registered

results = pd.DataFrame({
    'datetime': test_datetime,
    'casual': pred_casual.round().astype(int),
    'registered': pred_registered.round().astype(int),
    'count': pred_count.round().astype(int),
})

results.to_csv(r"C:\Users\AARINI\Downloads\test_predictions.csv", index=False)
print(results.head())